"""Qwen as a second annotator on dataset-labeled records.

Usage: python -m spanjudge.annotate [--config configs/mix.yaml] [--ids FILE]

Runs after the judge stage and writes data_dir/annotations.jsonl (one row
per record id); records.jsonl is not changed. `report` applies the rows.

Label: when Qwen's own label agrees with the dataset label (scores within
one point), the dataset label stays. When they disagree, Qwen reasons in a
think block over both labels (`adjudicate.j2`) and its verdict is the label.
IFEval labels come from code checks and are not adjudicated.

Spans: the human spans for the final label (if any) plus Qwen's spans for
that label that do not overlap them and pass a counterfactual test. On a
relabeled record the human spans no longer fit, so all spans are Qwen's
tested ones.
"""

import argparse
import asyncio
import hashlib
import json
import os
from typing import Any

import yaml

from spanjudge import llm
from spanjudge import pipeline
from spanjudge import prompts
from spanjudge import schema

_CHECKED_SOURCES = frozenset({"ifeval"})  # labels from code checks


def _disagree(r: schema.Record) -> bool:
    """Whether Qwen's label and the dataset label disagree."""
    a, b = r.qwen.label, r.gold.label
    if r.question.type == "score" and a.isdigit() and b.isdigit():
        return abs(int(a) - int(b)) >= 2
    return a != b


async def _adjudicate(
    cfg: dict[str, Any], client: llm.Client, r: schema.Record
) -> dict[str, Any]:
    """Qwen's reasoned verdict between the dataset label and its own."""
    pair = [r.gold.label, r.qwen.label]
    if hashlib.sha1(r.id.encode()).digest()[0] % 2:  # no position bias
        pair.reverse()
    q = r.question
    prompt = prompts.render(
        "adjudicate",
        context=r.context,
        response=r.response,
        question=q.text,
        labels=q.labels,
        first=pair[0],
        second=pair[1],
    )
    out = await client.complete(
        "adjudicate",
        llm.chat_prefix(prompt, think=True),
        max_tokens=cfg.get("adjudicate_max_tokens", 8000),
    )
    answer = out["text"].rsplit("</think>", 1)[-1]
    parsed = pipeline._parse_json(answer[answer.find("{") :])  # noqa: SLF001
    label = (parsed or {}).get("label")
    return {
        "label": label if label in q.labels else None,
        "reason": (parsed or {}).get("reason", ""),
        "shown": pair,
        "finish": out.get("finish"),
    }


async def _spans(
    cfg: dict[str, Any],
    client: llm.Client,
    r: schema.Record,
    label: str,
    base: list[schema.Span],
) -> tuple[list[schema.Span], list[schema.Span]]:
    """Base spans plus Qwen's tested spans for `label`: (kept, dropped)."""
    q = r.question
    if label == schema.UNRELATED or label in q.empty_labels:
        return [], []
    if r.qwen.label == label:
        cands = r.qwen.spans
    else:
        _, again, _ = await pipeline._judge(  # noqa: SLF001
            client, r, r.response, given_label=label
        )
        cands = []
        if again:
            cands, _, _ = schema.locate_quotes(
                pipeline._quotes(again),  # noqa: SLF001
                r.response,
                origin="qwen_given_label",
            )
    new = [
        s
        for s in cands
        if not any(s.start < b.end and s.end > b.start for b in base)
    ]
    if not new:
        return base, []
    spans = schema.merge(
        [s.model_copy() for s in base] + [s.model_copy() for s in new],
        r.response,
    )
    s0 = pipeline.support(r.dist, q, label)
    tests, _, _, _ = await pipeline._test_spans(  # noqa: SLF001
        cfg, client, r, spans, label, s0
    )
    size = len(r.response.strip())
    kept, dropped = [], []
    for s, t in zip(spans, tests, strict=True):
        s.test = t
        is_base = any(s.start == b.start and s.end == b.end for b in base)
        whole = size > 200 and s.end - s.start > 0.9 * size
        if is_base or (t and t.kept and not whole):
            kept.append(s)
        else:
            dropped.append(s)
    return kept, dropped


async def _annotate(
    cfg: dict[str, Any], client: llm.Client, r: schema.Record
) -> dict[str, Any]:
    gold = r.gold
    row: dict[str, Any] = {"id": r.id, "adjudication": None}
    label = gold.label
    if r.source not in _CHECKED_SOURCES and _disagree(r):
        verdict = await _adjudicate(cfg, client, r)
        row["adjudication"] = verdict
        if verdict["label"] is not None:
            label = verdict["label"]
    row["label"] = label
    row["label_from"] = (
        "dataset+qwen" if label == gold.label else "qwen_adjudicated"
    )
    if label != gold.label:
        base = []
    elif gold.spans_annotated:
        base = [s.model_copy(update={"test": None}) for s in gold.spans]
    else:  # label-only source: its spans are already Qwen's tested ones
        row.update(spans=None, dropped=None)
        return row
    kept, dropped = await _spans(cfg, client, r, label, base)
    row["spans"] = [s.model_dump() for s in kept]
    row["dropped"] = [s.model_dump() for s in dropped]
    return row


def _wanted(r: schema.Record) -> bool:
    return (
        r.related
        and r.gold is not None
        and pipeline._leave_out(r) is None  # noqa: SLF001
    )


async def annotate(cfg: dict[str, Any], ids: set[str] | None = None) -> None:
    """Annotates every wanted record not yet in annotations.jsonl."""
    data = cfg["data_dir"]
    path = f"{data}/annotations.jsonl"
    pipeline._drop_partial_tail(path)  # noqa: SLF001
    done = {row["id"] for row in pipeline._jsonl(path)}  # noqa: SLF001
    records = [
        schema.Record(**row)
        for row in pipeline._jsonl(f"{data}/records.jsonl")  # noqa: SLF001
        if row["id"] not in done and (ids is None or row["id"] in ids)
    ]
    todo = [r for r in records if _wanted(r)]
    print(f"annotate: {len(todo)} records ({len(done)} done before)")
    client = llm.Client(cfg)
    queue: asyncio.Queue = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)
    counts = {"done": len(done), "errors": 0}
    total = len(done) + len(todo)

    async def worker(f: Any) -> None:
        while not queue.empty():
            r = queue.get_nowait()
            try:
                row = await _annotate(cfg, client, r)
            except Exception as e:  # noqa: BLE001 - keep the run going
                counts["errors"] += 1
                print(f"annotate: {r.id} failed: {e!r}")
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            counts["done"] += 1
            if counts["done"] % 10 == 0 or queue.empty():
                pipeline._progress(  # noqa: SLF001
                    cfg, stage="annotate", total=total, **counts
                )

    with open(path, "a") as f:
        await asyncio.gather(*(worker(f) for _ in range(cfg["item_workers"])))
    pipeline._progress(
        cfg, stage="annotate", total=total, **counts
    )  # noqa: SLF001


def main() -> None:
    """Runs the annotate stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/mix.yaml")
    parser.add_argument("--ids", help="file with one record id per line")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["data_dir"], exist_ok=True)
    ids = None
    if args.ids:
        with open(args.ids) as f:
            ids = {line.strip() for line in f if line.strip()}
    asyncio.run(annotate(cfg, ids))


if __name__ == "__main__":
    main()
