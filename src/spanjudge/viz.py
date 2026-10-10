"""Viewer for the span-judging data.

Usage: python -m spanjudge.viz [--port 8777] [--config configs/mix.yaml]
Reads what the pipeline appended since the last request, so it works during a
run. Records show a sample per (source, final label).
"""

import argparse
import json
import pathlib
from typing import Any

import fastapi
from fastapi import responses
import uvicorn
import yaml

from spanjudge import pipeline
from spanjudge import prompts

_HERE = pathlib.Path(__file__).parent


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


class _Tail:
    """Reads only the complete lines appended to a JSONL file since last time.

    Keeps a byte offset per record id so one record can be fetched without
    reading the whole file, and a small sample per (source, label) to show.
    """

    def __init__(self, path: pathlib.Path, per_label: int):
        self.path = path
        self.per_label = per_label
        self.pos = 0
        self.offsets: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.sample: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def update(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            while True:
                start = f.tell()
                line = f.readline()
                if not line.endswith(b"\n"):  # missing or still being written
                    break
                self.pos = f.tell()
                if not line.strip():
                    continue
                row = json.loads(line)
                self.offsets[row["id"]] = start
                src = row["source"]
                self.counts[src] = self.counts.get(src, 0) + 1
                key = (src, row["final"]["label"] if row["related"] else "~")
                bucket = self.sample.setdefault(key, [])
                if len(bucket) < self.per_label:
                    row.pop("raw", None)  # loads per record
                    bucket.append(row)

    def get(self, item: str) -> dict[str, Any]:
        if item not in self.offsets:
            return {}
        with open(self.path, "rb") as f:
            f.seek(self.offsets[item])
            return json.loads(f.readline())

    def shown(self) -> list[dict[str, Any]]:
        return [r for key in sorted(self.sample) for r in self.sample[key]]


class _Ledger:
    """Running totals over the ledger, read incrementally."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.pos = 0
        self.totals = {"calls": 0, "tokens_in": 0, "tokens_out": 0}

    def update(self) -> dict[str, int]:
        if self.path.exists():
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                for line in f:
                    if not line.endswith(b"\n"):
                        break
                    self.pos += len(line)
                    entry = json.loads(line)
                    self.totals["calls"] += 1
                    self.totals["tokens_in"] += entry["in"]
                    self.totals["tokens_out"] += entry["out"]
        return self.totals


def _steps(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The pipeline steps, with the template text of each LLM call."""
    tau = cfg["tau"]
    return [
        {
            "title": "1. Build a balanced mix",
            "llm": "Qwen writes responses for rubrichub, infobench, ifeval",
            "prompt": None,
            "note": "13 sources, equal share each (smaller ones give all"
            " they have), sampled over (question family, label). Every"
            f" source also gets {int(cfg['unrelated_share'] * 100)}%"
            " unrelated items whose question cannot apply. Details:"
            " docs/data_mix.md.",
        },
        {
            "title": "2. Judge: label + spans",
            "llm": "completion, greedy, top-20 log-probs on the first token",
            "prompt": prompts.source("judge"),
            "prefill": pipeline.PREFILL,
            "note": "The reply is forced to start with the prefill, so the"
            " first generated token is the label and its log-probabilities"
            " give the label distribution. Final label: the dataset's when"
            " it has one. Final spans: human spans when annotated; else"
            " Qwen's (asked again with the label given when its own label"
            " differs).",
        },
        {
            "title": "3. Rewrite each span",
            "llm": "chat completion, greedy",
            "prompt": prompts.source("rewrite"),
            "tasks": {
                name: prompts.source(name)
                for name in (
                    "counter_label",
                    "counter_score",
                    "paraphrase_label",
                    "paraphrase_score",
                )
            },
            "note": "A counterfactual that should change the answer and a"
            " paraphrase that should not (for problems and scores it must"
            " keep the error). Refusals, empty rewrites and copies are"
            " rejected.",
        },
        {
            "title": "4. Score the rewritten responses",
            "llm": "judge prompt of step 2, 1 token",
            "prompt": None,
            "note": "Support = p(label) (scores: how far the expected score is"
            " below 5). Per span: only this span rewritten (leave-one-out),"
            " only this span paraphrased (control), every other span"
            " rewritten (leave-one-in); and all spans rewritten."
            f" necessary: paraphrase moves support < {tau} and the"
            f" counterfactual drops it > {tau}. sufficient: support with the"
            f" others rewritten beats all rewritten by > {tau}. Qwen spans"
            " that are neither are dropped; human spans always stay.",
        },
        {
            "title": "5. Look for missed evidence (Qwen spans only)",
            "llm": "judge prompt of step 2, full reply",
            "prompt": None,
            "note": f"If rewriting all spans leaves support within {tau},"
            " the judge runs on the rewritten response; new spans outside"
            " the rewritten passages are added and steps 3-4 run again.",
        },
        {
            "title": "6. Specificity and export",
            "llm": "judge prompt of step 2, 1 token",
            "prompt": None,
            "note": "For Qwen-written sources, 2 other questions of the same"
            " prompt are scored before and after the rewrites. Export:"
            " train.jsonl with label, spans and where each came from;"
            " unrelated items Qwen disputes are left out.",
        },
    ]


def make_app(cfg: dict[str, Any]) -> fastapi.FastAPI:
    """Creates the viewer app for the config's data directory."""
    data = pathlib.Path(cfg["data_dir"])
    app = fastapi.FastAPI()
    records = _Tail(data / "records.jsonl", per_label=2)
    ledger = _Ledger(data / "ledger.jsonl")

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return (_HERE / "viz.html").read_text()

    @app.get("/api/item/{item}")
    def item(item: str) -> dict[str, Any]:
        records.update()
        return records.get(item)

    @app.get("/api/steps")
    def steps() -> list[dict[str, Any]]:
        return _steps(cfg)

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        return _json(data / "progress.json") or {}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        records.update()
        return {
            "progress": _json(data / "progress.json") or {},
            "report": _json(data / "report.json"),
            "records": records.shown(),  # a sample per (source, label)
            "counts": records.counts,
            "rejects": _jsonl(data / "rejects.jsonl")[-200:],
            "llm": ledger.update(),
        }

    return app


def main() -> None:
    """Serves the viewer."""
    parser = argparse.ArgumentParser(description="Span-judging data viewer")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--config", default="configs/mix.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    uvicorn.run(make_app(cfg), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
