"""Span-judging data: build items, judge, test spans with rewrites, export.

Usage:
  python -m spanjudge.pipeline build   # dataset rows (+ Qwen responses)
  python -m spanjudge.pipeline judge   # label + spans, counterfactual tests
  python -m spanjudge.pipeline report  # summary + training export
  python -m spanjudge.pipeline all
"""

import argparse
import asyncio
import collections
from collections.abc import Callable
import html
import json
import math
import os
import random
import re
import time
from typing import Any

import yaml

from spanjudge import datasets
from spanjudge import llm
from spanjudge import prompts
from spanjudge import schema

# The judge's reply is forced to start here, so the first generated token is
# the label and its log-probabilities give the label distribution.
PREFILL = '{"label": "'
_REFUSAL = re.compile(
    r"^\W*(I cannot|I can't|I can not|I won't|I will not|I'm sorry|I am sorry"
    r"|Sorry|As an AI|I'm unable|I am unable|I must decline)",
    re.IGNORECASE,
)


def _jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _drop_partial_tail(path: str) -> None:
    """Cuts a last line left half-written by a killed run."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def _write_jsonl(path: str, rows: list[Any]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def _progress(cfg: dict[str, Any], **fields: Any) -> None:
    with open(f"{cfg['data_dir']}/progress.json", "w") as f:
        json.dump({**fields, "time": time.time()}, f)


# --- build -------------------------------------------------------------------


def _stratified(
    items: list[schema.Item],
    n: int,
    key: Callable[[schema.Item], Any],
    rng: random.Random,
) -> list[schema.Item]:
    """Takes n items round-robin over strata, so small strata are kept."""
    pools: dict[Any, list[schema.Item]] = collections.defaultdict(list)
    for it in items:
        pools[key(it)].append(it)
    for pool in pools.values():
        rng.shuffle(pool)
    out: list[schema.Item] = []
    strata = sorted(pools, key=str)
    while len(out) < n and any(pools.values()):
        for s in strata:
            if pools[s] and len(out) < n:
                out.append(pools[s].pop())
    return out


def _water_fill(
    available: dict[str, int],
    total: int,
    weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Shares in proportion to weight; sources with fewer items give all.

    Args:
      available: Candidate items per source.
      total: Items wanted over all sources.
      weights: Relative share per source (default 1; 0 leaves it out).

    Returns:
      Items to take per source.
    """
    weights = weights or {}
    take = dict.fromkeys(available, 0)
    left = total
    open_ = {s for s in available if available[s] and weights.get(s, 1) > 0}
    while left > 0 and open_:
        unit = left / sum(weights.get(s, 1) for s in open_)
        given = 0
        for s in sorted(open_):
            share = max(1, int(unit * weights.get(s, 1)))
            add = min(share, available[s] - take[s], left - given)
            take[s] += add
            given += add
            if take[s] >= available[s]:
                open_.discard(s)
            if given >= left:
                break
        left -= given
    return take


async def _respond(
    cfg: dict[str, Any], client: llm.Client, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Qwen answers each prompt as asked and (mostly) carelessly too.

    Returns:
      The prompt rows (each once per variant) and (variant, response) pairs.
    """

    def careless_ok(r: dict[str, Any]) -> bool:
        # A "repeat the request" prompt would repeat the hidden note too.
        return not any("repeat the request" in q for q in r["questions"])

    jobs = [(r, "normal") for r in rows] + [
        (r, "careless") for r in rows if careless_ok(r)
    ]
    outs = await asyncio.gather(
        *(
            client.complete(
                "respond",
                llm.chat_prefix(
                    r["prompt"] + (datasets.CARELESS if v == "careless" else "")
                ),
                cfg["respond_max_tokens"],
            )
            for r, v in jobs
        )
    )
    return [r for r, _ in jobs], [
        (v, o["text"].strip()) for (_, v), o in zip(jobs, outs, strict=True)
    ]


def _related_total(cfg: dict[str, Any]) -> int:
    return round(cfg["total"] * (1 - cfg["unrelated_share"]))


def _rubric_split(
    cfg: dict[str, Any], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """RubricHub prompts to answer, and spare ones for unrelated questions."""
    rubric = datasets.rubric_prompts(cfg["rubrichub_scan"])
    rng.shuffle(rubric)
    k = cfg["rubrichub_prompts"]
    return rubric[:k], rubric[k:]


def _loaders(
    rng: random.Random, raw: str
) -> dict[str, Callable[[], list[schema.Item]]]:
    """Candidate builders for the sources with dataset responses."""
    return {
        "ragtruth": lambda: datasets.ragtruth(rng),
        "mqm": lambda: datasets.mqm(rng),
        "qa_feedback": lambda: datasets.qa_feedback(rng, raw),
        "fava": lambda: datasets.fava(rng),
        "mushroom": lambda: datasets.mushroom(rng),
        "prm800k": lambda: datasets.prm800k(rng),
        "processbench": lambda: datasets.processbench(rng),
        "helpsteer2": lambda: datasets.helpsteer2(rng),
        "ultrafeedback": lambda: datasets.ultrafeedback(rng),
        "beavertails": lambda: datasets.beavertails(rng),
        "aegis": lambda: datasets.aegis(rng),
    }


def _pick(
    cfg: dict[str, Any],
    cands: list[schema.Item],
    n: int,
    pool: list[tuple[str, datasets.Family, frozenset[str]]],
    rng: random.Random,
) -> list[schema.Item]:
    """Takes n related items over (family, label), plus unrelated ones."""
    chosen = _stratified(
        cands,
        n,
        lambda it: (it.meta["family"], it.gold.label if it.gold else ""),
        rng,
    )
    ratio = cfg["unrelated_share"] / (1 - cfg["unrelated_share"])
    return chosen + [
        datasets.unrelated(rng.choice(chosen), pool, rng)
        for _ in range(round(len(chosen) * ratio))
    ]


def _quotas(cfg: dict[str, Any]) -> dict[tuple[str, bool], int] | None:
    """Items to judge per (source, related) so the export can fill its share.

    The share from `source_weights` is divided by the fraction of the
    source's records judged so far that `_leave_out` keeps, plus 3% slack.
    None when the config has no weights (judge everything).
    """
    if not cfg.get("source_weights"):
        return None
    data = cfg["data_dir"]
    with open(f"{data}/build.json") as f:
        candidates = json.load(f)["candidates"]
    target = _water_fill(candidates, _related_total(cfg), cfg["source_weights"])
    ratio = cfg["unrelated_share"] / (1 - cfg["unrelated_share"])
    seen: collections.Counter = collections.Counter()
    kept: collections.Counter = collections.Counter()
    for row in _jsonl(f"{data}/records.jsonl"):
        r = schema.Record(**row)
        seen[r.source, r.related] += 1
        kept[r.source, r.related] += _leave_out(r) is None
    quotas = {}
    for source, n in target.items():
        for related, want in ((True, n), (False, n * ratio)):
            key = (source, related)
            rate = kept[key] / seen[key] if seen[key] >= 50 else 1.0
            quotas[key] = math.ceil(want / max(rate, 0.3) * 1.03)
    return quotas


async def topup(cfg: dict[str, Any]) -> None:
    """Appends items to sources that are below their weighted share.

    For a mix that was built (and partly judged) with other weights: the
    items already there are kept, so nothing judged is lost; `report` then
    samples every source down to its share. Only short sources with dataset
    responses are loaded again (generated sources would need new
    responses), and a new item never repeats a (response, question
    family) that is already in items.jsonl.
    """
    data = cfg["data_dir"]
    items = [schema.Item(**r) for r in _jsonl(f"{data}/items.jsonl")]
    with open(f"{data}/build.json") as f:
        built = json.load(f)
    rng = random.Random(cfg["seed"] + 1)
    loaders = _loaders(rng, f"{data}/raw")
    loaded: dict[str, list[schema.Item]] = {}
    weights = cfg.get("source_weights") or {}
    for source, load in loaders.items():  # a source added after the build
        if source not in built["candidates"] and weights.get(source, 1) > 0:
            loaded[source] = load()
            built["candidates"][source] = len(loaded[source])
            print(f"topup: new source {source}, {len(loaded[source])} items")
    with open(f"{data}/build.json", "w") as f:
        json.dump(built, f, indent=1)
    quotas = _quotas(cfg) or {}
    target = {s: n for (s, rel), n in quotas.items() if rel}
    have = collections.Counter(it.source for it in items if it.related)
    short = {
        s: n - have[s]
        for s, n in target.items()
        if n > have[s] and s in loaders
    }
    print("topup: target", target, "\ntopup: adding", short)
    _, spare = _rubric_split(cfg, random.Random(cfg["seed"]))  # as in build
    pool = datasets.unrelated_pool([q for p in spare for q in p["questions"]])
    seen = {(it.response, it.meta["family"]) for it in items if it.related}
    ids = {it.id for it in items}
    added: list[schema.Item] = []
    for source, n in short.items():
        cands = []
        for it in loaded.get(source) or loaders[source]():
            key = (it.response, it.meta["family"])
            if key not in seen and it.id not in ids:
                seen.add(key)  # one wording per (response, family)
                cands.append(it)
        new = []
        for it in _pick(cfg, cands, n, pool, rng):
            if it.id not in ids:  # an unrelated question drawn twice
                ids.add(it.id)
                new.append(it)
        added += new
        print(f"topup: {source} +{len(new)} items")
    rng.shuffle(added)
    with open(f"{data}/items.jsonl", "a") as f:
        for it in added:
            f.write(it.model_dump_json() + "\n")
    built["topup"] = {
        "target_related": target,
        "added": dict(collections.Counter(it.source for it in added)),
    }
    with open(f"{data}/build.json", "w") as f:
        json.dump(built, f, indent=1)
    print(f"topup: {len(added)} items appended")


def _to_shares(
    cfg: dict[str, Any], rows: list[tuple[schema.Record, dict[str, Any]]]
) -> list[tuple[schema.Record, dict[str, Any]]]:
    """Samples exportable rows down to each source's weighted share.

    Related rows are shared out by `source_weights` over what each source
    has (stratified over family and label); each source keeps unrelated
    rows in proportion to its related ones. Order is kept.
    """
    rng = random.Random(cfg["seed"])
    by_source: dict[tuple[str, bool], list[int]] = collections.defaultdict(list)
    for i, (r, _) in enumerate(rows):
        by_source[(r.source, r.related)].append(i)
    available = {s: len(v) for (s, rel), v in by_source.items() if rel}
    take = _water_fill(
        available, _related_total(cfg), cfg.get("source_weights")
    )
    ratio = cfg["unrelated_share"] / (1 - cfg["unrelated_share"])
    keep: set[int] = set()
    for (source, related), idx in by_source.items():
        if related:
            strata: dict[Any, list[int]] = collections.defaultdict(list)
            for i in idx:
                r = rows[i][0]
                strata[(r.meta.get("family"), r.final.label)].append(i)
            for v in strata.values():
                rng.shuffle(v)
            n, chosen = take.get(source, 0), []
            while len(chosen) < n and any(strata.values()):
                for k in sorted(strata, key=str):
                    if strata[k] and len(chosen) < n:
                        chosen.append(strata[k].pop())
        else:
            n = round(take.get(source, 0) * ratio)
            chosen = rng.sample(idx, min(n, len(idx)))
        keep.update(chosen)
    return [row for i, row in enumerate(rows) if i in keep]


async def build(cfg: dict[str, Any]) -> None:
    """Writes items.jsonl: a balanced mix of related and unrelated items.

    Every source's candidates are built first; then each source gets a
    share of the related items in proportion to `source_weights` (sources
    with fewer candidates give all of theirs), sampled round-robin over
    (question family, label). Each source then gets unrelated items in
    proportion to its related ones.
    """
    rng = random.Random(cfg["seed"])
    raw = f"{cfg['data_dir']}/raw"
    client = llm.Client(cfg)
    rubric, spare = _rubric_split(cfg, rng)
    generated = {
        "rubrichub": rubric,
        "infobench": datasets.infobench_prompts(),
        "ifeval": datasets.ifeval_prompts(),
    }
    candidates: dict[str, list[schema.Item]] = {}
    for name, rows in generated.items():
        rng.shuffle(rows)
        rows = rows[: cfg.get("max_generated_prompts") or len(rows)]
        prompt_rows, responses = await _respond(cfg, client, rows)
        candidates[name] = datasets.generated_items(
            name, prompt_rows, responses, cfg["questions_per_response"], rng
        )
        print(f"build: {name} {len(candidates[name])} candidates")
    for name, load in _loaders(rng, raw).items():
        candidates[name] = load()
        print(f"build: {name} {len(candidates[name])} candidates")

    take = _water_fill(
        {s: len(v) for s, v in candidates.items()},
        _related_total(cfg),
        cfg.get("source_weights"),
    )
    pool = datasets.unrelated_pool([q for p in spare for q in p["questions"]])
    items: list[schema.Item] = []
    for source, cands in candidates.items():
        items += _pick(cfg, cands, take[source], pool, rng)
    rng.shuffle(items)
    _write_jsonl(f"{cfg['data_dir']}/items.jsonl", items)
    counts = collections.Counter(it.source for it in items)
    with open(f"{cfg['data_dir']}/build.json", "w") as f:
        json.dump(
            {
                "candidates": {s: len(v) for s, v in candidates.items()},
                "related_taken": take,
                "items_per_source": counts,
            },
            f,
            indent=1,
        )
    print(f"build: {len(items)} items", dict(counts))


# --- judge and test ----------------------------------------------------------


def _judge_prefix(
    item: schema.Item,
    response: str,
    question: schema.Question | None = None,
    given_label: str | None = None,
) -> str:
    q = question or item.question
    prompt = prompts.render(
        "judge",
        context=item.context,
        response=response,
        question=q.text,
        labels=q.labels,
        span_rule=q.span_rule,
        given_label=given_label,
    )
    return llm.chat_prefix(prompt) + PREFILL


def label_dist(first: list[list], labels: list[str]) -> dict[str, float] | None:
    """Label probabilities from the first label token's top log-probs.

    A token counts for the one label it is a prefix of ("ye" -> yes).

    Args:
      first: (token, logprob) pairs of the first generated token.
      labels: The question's label ids.

    Returns:
      Probability per label, renormalized, or None when the label tokens
      carry less than half of the probability mass.
    """
    probs = dict.fromkeys(labels, 0.0)
    for token, logprob in first:
        t = token.strip().strip('"')
        hits = [lab for lab in labels if t and lab.startswith(t)]
        if len(hits) == 1:
            probs[hits[0]] += math.exp(logprob)
    mass = sum(probs.values())
    if mass < 0.5:
        return None
    return {k: v / mass for k, v in probs.items()}


def support(dist: dict[str, float], q: schema.Question, label: str) -> float:
    """How strongly the judge holds the answer.

    p(label) for boolean questions. For score questions the spans are what
    lowers the score, so support is how far the expected score (over 1..5)
    is below 5, from 0 to 1.
    """
    if q.type != "score":
        return dist[label]
    levels = {k: v for k, v in dist.items() if k.isdigit()}
    mass = sum(levels.values()) or 1.0
    expected = sum(int(k) * v for k, v in levels.items()) / mass
    return (5 - expected) / 4


async def _dist(
    client: llm.Client,
    item: schema.Item,
    response: str,
    question: schema.Question | None = None,
) -> dict[str, float] | None:
    """The judge's label distribution, from one forward pass."""
    q = question or item.question
    out = await client.complete(
        "score", _judge_prefix(item, response, q), max_tokens=1, top=20
    )
    return label_dist(out["first"], list(q.labels))


def _replace(
    text: str, spans: list[schema.Span], news: list[str]
) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Replaces sorted, disjoint spans.

    Returns:
      The new text and, per span, (old start, old end, new start, new end).
    """
    out, regions, last, pos = [], [], 0, 0
    for span, new in zip(spans, news, strict=True):
        out.append(text[last : span.start])
        pos += span.start - last
        regions.append((span.start, span.end, pos, pos + len(new)))
        out.append(new)
        pos += len(new)
        last = span.end
    out.append(text[last:])
    return "".join(out), regions


def _keeps_errors(q: schema.Question, label: str) -> bool:
    """Whether the spans show a shortcoming a paraphrase must not fix.

    True for score questions and problem questions (spans are errors), and
    for requirement questions answered "no" (spans fall short).
    """
    return q.type == "score" or "no" in q.empty_labels or label == "no"


async def _rewrite(
    client: llm.Client,
    item: schema.Item,
    span: schema.Span,
    task: str,
    allow_same: bool = False,
) -> str | None:
    """One rewrite of a span.

    Args:
      client: LLM client.
      item: The item the span belongs to.
      span: The span to rewrite.
      task: The rewrite instruction.
      allow_same: Accept a rewrite identical to the span (a paraphrase of a
        name or a single word often has to be).

    Returns:
      The new text, or None for a refusal, empty text, or (unless allowed)
      a copy of the span.
    """
    prompt = prompts.render(
        "rewrite",
        context=item.context,
        response=item.response,
        question=item.question.text,
        passage=span.text,
        task=task,
    )
    out = await client.complete(
        "rewrite",
        llm.chat_prefix(prompt),
        max_tokens=min(3000, 200 + len(span.text)),
    )
    text = out["text"].strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    refusal = _REFUSAL.match(text) and not _REFUSAL.match(span.text)
    if not text or refusal or out.get("finish") == "length":
        return None  # a refusal span may be rewritten as a refusal
    if len(text) > max(3 * len(span.text), len(span.text) + 120):
        return None  # a rewrite that grows this much is not a local edit
    if " ".join(text.split()) == " ".join(span.text.split()):
        return span.text if allow_same else None
    return text


def _is_addition(old: str, new: str) -> bool:
    """Whether a rewrite keeps the old text and only adds to it.

    A requirement that fails by omission is "fixed" by adding the missing
    content anywhere, so such a passage is not evidence (omissions are out
    of scope).
    """
    flat_old, flat_new = " ".join(old.split()), " ".join(new.split())
    return flat_old in flat_new and len(flat_new) > 1.3 * len(flat_old) + 20


def _tasks(
    q: schema.Question, label: str, span: schema.Span
) -> tuple[str, str]:
    """The counterfactual and paraphrase instructions for one span.

    The span's reason (from Qwen or the human annotation) tells the rewriter
    what the problem is; without it, "fix the problem" gets copied as is.
    """
    reason = span.reason.strip().rstrip(".")
    if q.type == "score":
        counter = prompts.render("counter_score", reason=reason)
    else:
        counter = prompts.render(
            "counter_label",
            label=label,
            description=q.labels[label],
            reason=reason,
        )
    para = "paraphrase_score" if _keeps_errors(q, label) else "paraphrase_label"
    return counter, prompts.render(para)


async def _test_spans(
    cfg: dict[str, Any],
    client: llm.Client,
    item: schema.Item,
    spans: list[schema.Span],
    label: str,
    s0: float,
) -> tuple[list[schema.SpanTest | None], float | None, str | None, list]:
    """Leave-one-out / leave-one-in counterfactual tests of a span set.

    Args:
      cfg: Config (uses `tau`).
      client: LLM client.
      item: The item.
      spans: Sorted, disjoint spans of the response.
      label: The final label the spans should support.
      s0: Support for that label on the original response.

    Returns:
      A test per span (None when a rewrite or a score failed), support and
      argmax label with every span rewritten, and each span's counterfactual
      (the original text where the rewrite failed).
    """
    tau, q, text = cfg["tau"], item.question, item.response
    tasks = [_tasks(q, label, s) for s in spans]
    counters = await asyncio.gather(
        *(
            _rewrite(client, item, s, t[0])
            for s, t in zip(spans, tasks, strict=True)
        )
    )
    paras = await asyncio.gather(
        *(
            _rewrite(client, item, s, t[1], allow_same=True)
            for s, t in zip(spans, tasks, strict=True)
        )
    )
    news = [
        c if c is not None else s.text
        for c, s in zip(counters, spans, strict=True)
    ]
    variants = [_replace(text, spans, news)[0]]  # all rewritten
    for k, s in enumerate(spans):
        variants.append(_replace(text, [s], [news[k]])[0])
        variants.append(_replace(text, [s], [paras[k] or s.text])[0])
        others = [j for j in range(len(spans)) if j != k]
        variants.append(
            _replace(
                text, [spans[j] for j in others], [news[j] for j in others]
            )[0]
        )
    dists = await asyncio.gather(*(_dist(client, item, v) for v in variants))
    if dists[0] is None:
        return [None] * len(spans), None, None, news
    s_all = support(dists[0], q, label)
    label_all = max(dists[0], key=lambda k: dists[0][k])
    tests: list[schema.SpanTest | None] = []
    for k in range(len(spans)):
        d_c, d_p, d_o = dists[1 + 3 * k : 4 + 3 * k]
        if None in (counters[k], paras[k], d_c, d_p, d_o) or _is_addition(
            spans[k].text, counters[k]
        ):
            tests.append(None)
            continue
        s_c, s_p, s_o = (support(d, q, label) for d in (d_c, d_p, d_o))
        control_ok = abs(s0 - s_p) < tau
        necessary = control_ok and s0 - s_c > tau
        sufficient = s_o - s_all > tau
        tests.append(
            schema.SpanTest(
                counter=counters[k],
                paraphrase=paras[k],
                s_counter=s_c,
                s_paraphrase=s_p,
                s_others_rewritten=s_o,
                control_ok=control_ok,
                necessary=necessary,
                sufficient=sufficient,
                kept=necessary or sufficient,
            )
        )
    return tests, s_all, label_all, news


async def _judge(
    client: llm.Client,
    item: schema.Item,
    response: str,
    given_label: str | None = None,
) -> tuple[str, dict | None, dict | None]:
    """One judge call: (raw reply, parsed JSON or None, label distribution)."""
    out = await client.complete(
        "judge",
        _judge_prefix(item, response, given_label=given_label),
        max_tokens=2000,
        top=20,
    )
    raw = PREFILL + out["text"]
    return (
        raw,
        _parse_json(raw),
        label_dist(out["first"], list(item.question.labels)),
    )


def _parse_json(raw: str) -> dict | None:
    """Parses the judge's JSON, tolerating stray backslashes (LaTeX)."""
    body = raw[: raw.rfind("}") + 1]
    for text in (body, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", body)):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _quotes(parsed: dict) -> list[tuple[str, str]]:
    return [
        (str(s.get("text", "")), str(s.get("reason", "")))
        for s in parsed.get("spans") or []
        if isinstance(s, dict)
    ]


async def _missed_spans(
    client: llm.Client,
    item: schema.Item,
    spans: list[schema.Span],
    news: list[str],
    label: str,
) -> list[schema.Span]:
    """Asks the judge for evidence left after every span was rewritten.

    Only new spans outside the rewritten passages are kept, mapped back to
    offsets in the original response.
    """
    rewritten, regions = _replace(item.response, spans, news)
    _, parsed, _ = await _judge(client, item, rewritten)
    if not parsed or parsed.get("label") != label:
        return []
    found, _, _ = schema.locate_quotes(
        _quotes(parsed), rewritten, origin="qwen_round2"
    )
    out = []
    for s in found:
        if any(s.start < ne and s.end > ns for _, _, ns, ne in regions):
            continue
        shift = sum(
            (oe - os_) - (ne - ns)
            for os_, oe, ns, ne in regions
            if ne <= s.start
        )
        start, end = s.start + shift, s.end + shift
        if item.response[start:end] == s.text:
            out.append(s.model_copy(update={"start": start, "end": end}))
    return out


def _reject(item: schema.Item, reason: str, raw: str) -> dict[str, Any]:
    return {"id": item.id, "source": item.source, "reason": reason, "raw": raw}


async def _judge_item(
    cfg: dict[str, Any], client: llm.Client, item: schema.Item
) -> schema.Record | dict[str, Any]:
    """Judges one item and tests the final spans; a record or a reject.

    Final label: the human (or dataset) label when there is one, else
    Qwen's. Final spans: human spans when annotated; otherwise Qwen's spans
    (asked again with the label given when Qwen's own label differs), kept
    only if they pass a counterfactual test.
    """
    q, gold = item.question, item.gold
    raw, parsed, dist = await _judge(client, item, item.response)
    human = gold is not None and gold.spans_annotated
    if dist is None:
        return _reject(item, "label_mass", raw)
    if parsed is None or parsed.get("label") not in q.labels:
        if gold is None:  # labeled items stay: Qwen's answer is only reported
            reason = "bad_json" if parsed is None else "bad_label"
            return _reject(item, reason, raw)
        parsed = {"label": max(dist, key=lambda k: dist[k]), "spans": []}
    label = parsed["label"]
    spans, unlocated, ambiguous = schema.locate_quotes(
        _quotes(parsed), item.response
    )
    if label == schema.UNRELATED or label in q.empty_labels:
        spans = []
    qwen = schema.Annotation(label=label, spans=spans)
    if gold is None:
        final = qwen
    elif gold.spans_annotated:
        final = gold
    else:  # label only: Qwen's spans must support the dataset's label
        final_spans = spans if label == gold.label else []
        empty = gold.label == schema.UNRELATED or gold.label in q.empty_labels
        if label != gold.label and not empty:
            _, again, _ = await _judge(
                client, item, item.response, given_label=gold.label
            )
            if again:
                final_spans, _, _ = schema.locate_quotes(
                    _quotes(again), item.response, origin="qwen_given_label"
                )
        final = schema.Annotation(
            label=gold.label,
            spans=[] if empty else final_spans,
            spans_annotated=False,
        )
    record = schema.Record(
        **item.model_dump(),
        dist=dist,
        qwen=qwen,
        raw=raw,
        unlocated=unlocated,
        ambiguous=ambiguous,
        final=final,
        rounds=1,
    )
    if not final.spans or final.label == schema.UNRELATED:
        return record
    s0 = support(dist, q, final.label)
    record.support = s0
    test_spans = [s.model_copy() for s in final.spans]
    tests, s_all, label_all, news = await _test_spans(
        cfg, client, item, test_spans, final.label, s0
    )
    if not human and s_all is not None and s0 - s_all <= cfg["tau"]:
        # Rewriting every span left the answer standing: evidence was missed.
        extra = await _missed_spans(client, item, test_spans, news, final.label)
        if extra:
            test_spans = schema.merge(test_spans + extra, item.response)
            tests, s_all, label_all, news = await _test_spans(
                cfg, client, item, test_spans, final.label, s0
            )
        record.rounds = 2
    for span, test in zip(test_spans, tests, strict=True):
        span.test = test
    record.s_all_rewritten, record.label_all_rewritten = s_all, label_all
    if human:  # human spans stay; their tests are recorded with them
        record.final = final.model_copy(update={"spans": test_spans})
    else:  # Qwen's spans must pass a test to stay
        size = len(item.response.strip())

        def good(s: schema.Span) -> bool:
            # Marking (almost) the whole response is not evidence.
            whole = size > 200 and s.end - s.start > 0.9 * size
            return bool(s.test and s.test.kept) and not whole

        record.final = final.model_copy(
            update={"spans": [s for s in test_spans if good(s)]}
        )
        record.dropped = [s for s in test_spans if not good(s)]
    if item.siblings:
        rewritten = _replace(item.response, test_spans, news)[0]
        before = await asyncio.gather(
            *(_dist(client, item, item.response, sq) for sq in item.siblings)
        )
        after = await asyncio.gather(
            *(_dist(client, item, rewritten, sq) for sq in item.siblings)
        )
        record.siblings_after = [
            schema.Sibling(
                question=sq.text, p_yes_before=b["yes"], p_yes_after=a["yes"]
            )
            for sq, b, a in zip(item.siblings, before, after, strict=True)
            if a and b
        ]
    return record


async def judge(cfg: dict[str, Any]) -> None:
    """Judges every item not yet done; appends to records/rejects.jsonl.

    A fixed pool of workers takes items in order, so records arrive steadily
    and a stopped run resumes where it left off (finished items are skipped;
    items that failed with an exception are retried). With `source_weights`
    a source stops once it has enough records for its share (`_quotas`).
    """
    data = cfg["data_dir"]
    rec_path, rej_path = f"{data}/records.jsonl", f"{data}/rejects.jsonl"
    for path in (rec_path, rej_path):
        _drop_partial_tail(path)
    done = {r["id"] for r in _jsonl(rec_path)}
    rejects = [r for r in _jsonl(rej_path) if r["reason"] != "error"]
    done |= {r["id"] for r in rejects}
    _write_jsonl_dicts(rej_path, rejects)
    items = [schema.Item(**r) for r in _jsonl(f"{data}/items.jsonl")]
    todo = [it for it in items if it.id not in done]
    quotas = _quotas(cfg)
    if quotas:  # items past their source's quota wait in items.jsonl
        have = collections.Counter(
            (r["source"], r["related"]) for r in _jsonl(rec_path)
        )
        kept_todo = []
        for it in todo:
            key = (it.source, it.related)
            if have[key] < quotas.get(key, 0):
                have[key] += 1
                kept_todo.append(it)
        print(f"judge: {len(todo) - len(kept_todo)} items over quota skipped")
        items = [it for it in items if it.id in done] + kept_todo
        todo = kept_todo
    client = llm.Client(cfg)
    queue: asyncio.Queue = asyncio.Queue()
    for it in todo:
        queue.put_nowait(it)
    counts = {"done": len(items) - len(todo), "rejected": len(rejects)}

    async def worker(rec_f: Any, rej_f: Any) -> None:
        while not queue.empty():
            item = queue.get_nowait()
            try:
                result = await _judge_item(cfg, client, item)
            except Exception as e:  # noqa: BLE001 - keep the run going
                result = _reject(item, "error", repr(e))
            if isinstance(result, schema.Record):
                rec_f.write(result.model_dump_json() + "\n")
                rec_f.flush()
            else:
                counts["rejected"] += 1
                rej_f.write(json.dumps(result) + "\n")
                rej_f.flush()
            counts["done"] += 1
            if counts["done"] % 10 == 0 or queue.empty():
                _progress(cfg, stage="judge", total=len(items), **counts)

    with open(rec_path, "a") as rec_f, open(rej_path, "a") as rej_f:
        _progress(cfg, stage="judge", total=len(items), **counts)
        await asyncio.gather(
            *(worker(rec_f, rej_f) for _ in range(cfg["item_workers"]))
        )
    _progress(cfg, stage="judge", total=len(items), **counts)
    print(f"judge: {counts} of {len(items)}", client.stats)


def _write_jsonl_dicts(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --- report and export -------------------------------------------------------


def _share(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def _char_f1(records: list[schema.Record]) -> dict[str, float] | None:
    """Character-level P/R/F1 of Qwen's spans against human spans."""
    tp = fp = fn = 0
    for r in records:
        gold = {i for s in r.gold.spans for i in range(s.start, s.end)}
        pred = {i for s in r.qwen.spans for i in range(s.start, s.end)}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    if not tp + fp + fn:
        return None
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"p": round(p, 3), "r": round(rec, 3), "f1": round(f1, 3)}


def _source_report(
    rs: list[schema.Record], rejected: int, tau: float
) -> dict[str, Any]:
    """Summary numbers for the records of one source."""
    rel = [r for r in rs if r.related]
    unrel = [r for r in rs if not r.related]
    human = [r for r in rel if r.gold and r.gold.spans_annotated]
    labelled = [r for r in rel if r.gold]
    tested = [s for r in rel for s in r.final.spans + r.dropped]
    tests = [s.test for s in tested if s.test]
    flips = [r for r in rel if r.label_all_rewritten]
    sib = [
        abs(x.p_yes_after - x.p_yes_before)
        for r in rel
        for x in r.siblings_after
    ]
    return {
        "related": len(rel),
        "unrelated": len(unrel),
        "rejected": rejected,
        "final_labels": dict(
            collections.Counter(r.final.label for r in rel).most_common()
        ),
        "qwen_label_agrees_with_dataset": _share(
            [r.qwen.label == r.gold.label for r in labelled]
        ),
        "qwen_span_char_vs_human": _char_f1(human) if human else None,
        "unrelated_answered_unrelated": _share(
            [r.qwen.label == schema.UNRELATED for r in unrel]
        ),
        "records_with_final_spans": _share([bool(r.final.spans) for r in rel]),
        "spans_tested": len(tested),
        "rewrite_failed": sum(s.test is None for s in tested),
        "control_ok": _share([t.control_ok for t in tests]),
        "necessary": _share([t.necessary for t in tests]),
        "sufficient": _share([t.sufficient for t in tests]),
        "kept": _share([t.kept for t in tests]),
        "label_flips_all_rewritten": _share(
            [r.label_all_rewritten != r.final.label for r in flips]
        ),
        "support_drop_all_rewritten_gt_tau": _share(
            [
                r.support - r.s_all_rewritten > tau
                for r in flips
                if r.support is not None
            ]
        ),
        "second_round": _share([r.rounds == 2 for r in rel]),
        "spans_given_label": sum(
            s.origin == "qwen_given_label" for s in tested
        ),
        "spans_round2": sum(s.origin == "qwen_round2" for s in tested),
        "sibling_mean_abs_change": (
            round(sum(sib) / len(sib), 4) if sib else None
        ),
    }


def _export_row(r: schema.Record) -> dict[str, Any]:
    """One training example: input, typed answer, evidence spans."""
    return {
        "id": r.id,
        "source": r.source,
        "context": r.context,
        "response": r.response,
        "question": r.question.text,
        "question_type": r.question.type,
        "labels": r.question.labels,
        "label": r.final.label,
        "label_from": (
            "dataset"
            if r.gold
            else ("construction" if not r.related else "qwen")
        ),
        "qwen_dist": r.dist,
        "spans": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in r.final.spans
        ],
        "spans_from": (
            "human" if r.gold and r.gold.spans_annotated else "qwen_tested"
        ),
        "related": r.related,
    }


# Sources whose prompts can ask for math (documents and translations do not).
_CHAT_SOURCES = frozenset(
    {"rubrichub", "infobench", "ifeval", "helpsteer2", "ultrafeedback", "aegis"}
)
_MATH_PROMPT = re.compile(
    r"\b(solve|equation|calculate|compute|prove|integral|derivative|"
    r"simplify|how many|what is \d)|\d\s*[-+*/^=]\s*\d",
    re.I,
)
_CARELESS_LEAK = re.compile(
    r"answer hastily|silently ignore|ignore one of the explicit", re.I
)


def _whole_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widens a range to whole words (Mu-SHROOM spans often cut words)."""
    return datasets._word_bounds(text, start, end)  # noqa: SLF001


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """Export-time cleanup that keeps span offsets valid.

    Strips a trailing chat end token from the response, unescapes HTML in
    the context, and widens Mu-SHROOM spans to whole words.
    """
    text = row["response"]
    for token in ("<|im_end|>", "</s>"):
        if text.rstrip().endswith(token):
            text = text.rstrip()[: -len(token)]
    spans = []
    for s in row["spans"]:
        a, b = s["start"], min(s["end"], len(text))
        if row["source"] == "mushroom":
            a, b = _whole_words(text, a, b)
        if a < b:
            spans.append({"start": a, "end": b, "text": text[a:b]})
    context = html.unescape(row["context"])
    context = re.sub(r"<br\s*/?>", "\n", context, flags=re.I)
    return {**row, "response": text, "context": context, "spans": spans}


def _leave_out(r: schema.Record) -> str | None:
    """Why a record is not exported, or None to export it.

    - The response repeats the hidden "careless" note.
    - An unrelated requirement question that Qwen thinks applies
      (p(unrelated) < 0.5): a requirement can be generic enough to fit.
    - An unrelated math question on a prompt that asks for math (solve,
      calculate, an equation...): the response is itself a solution.
    - A label that needs evidence but no tested span survived (e.g. "yes"
      for a problem question, a score below 5, or a requirement that fails
      only by omission). Human-spanned records always stay.
    """
    q = r.question
    if _CARELESS_LEAK.search(r.response):
        return "careless_note_leaked"
    if not r.related:
        pool = r.meta.get("unrelated_pool")
        if pool == "requirement" and r.dist.get(schema.UNRELATED, 0) < 0.5:
            return "unrelated_disputed"
        chat = r.source in _CHAT_SOURCES
        if pool == "math" and chat and _MATH_PROMPT.search(r.context):
            return "unrelated_applies"  # a math question on a math answer
        return None
    if r.gold is not None and r.gold.spans_annotated:
        return None
    label = r.final.label
    needs_spans = not (
        q.spans_optional or label == schema.UNRELATED or label in q.empty_labels
    )
    if needs_spans and not r.final.spans:
        return "no_tested_span"
    return None


def report(cfg: dict[str, Any]) -> dict[str, Any]:
    """Writes report.json and the training file train.jsonl.

    Unrelated items get the label "unrelated" with no spans in the export
    (true by construction). Records are left out for the reasons in
    `_leave_out`; the counts are in report.json.
    """
    data = cfg["data_dir"]
    records = [schema.Record(**r) for r in _jsonl(f"{data}/records.jsonl")]
    rejects = _jsonl(f"{data}/rejects.jsonl")
    out: dict[str, Any] = {
        "records": len(records),
        "rejects": dict(collections.Counter(r["reason"] for r in rejects)),
    }
    for source in sorted({r.source for r in records}):
        out[source] = _source_report(
            [r for r in records if r.source == source],
            sum(x.get("source") == source for x in rejects),
            cfg["tau"],
        )
    with open(f"{data}/report.json", "w") as f:
        json.dump(out, f, indent=1)
    left_out: collections.Counter = collections.Counter()

    # FAVA has some outputs twice with different annotations: a (response,
    # question family) seen with conflicting labels is left out entirely,
    # and exact repeats are exported once.
    def same_question(r: schema.Record) -> tuple:
        # Human datasets ask one family in several wordings; elsewhere every
        # requirement is its own question.
        human = r.gold is not None and r.gold.spans_annotated
        return (r.response, r.meta.get("family") if human else r.question.text)

    labels_seen: dict[tuple, set] = collections.defaultdict(set)
    for r in records:
        if r.related:
            labels_seen[same_question(r)].add(r.final.label)
    exported_keys: set[tuple] = set()
    rows: list[tuple[schema.Record, dict[str, Any]]] = []
    for r in records:
        reason = _leave_out(r)
        conflict = r.related and len(labels_seen[same_question(r)]) > 1
        if not reason and conflict:
            reason = "conflicting_duplicate"
        if not reason and (r.response, r.question.text) in exported_keys:
            reason = "duplicate"
        if reason:
            left_out[f"{r.source}:{reason}"] += 1
            continue
        exported_keys.add((r.response, r.question.text))
        row = _export_row(r)
        if not r.related:
            row.update(label=schema.UNRELATED, spans=[])
        rows.append((r, _clean_row(row)))
    shared = _to_shares(cfg, rows)
    over = collections.Counter(r.source for r, _ in rows)
    over.subtract(r.source for r, _ in shared)
    weights = cfg.get("source_weights") or {}
    left_out.update(
        {
            f"{s}:{'weight_0' if weights.get(s, 1) == 0 else 'over_share'}": n
            for s, n in over.items()
            if n
        }
    )
    with open(f"{data}/train.jsonl", "w") as f:
        for _, row in shared:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    exported = len(shared)
    out["left_out"] = dict(left_out.most_common())
    out["exported"] = exported
    out["exported_per_source"] = dict(
        collections.Counter(r.source for r, _ in shared).most_common()
    )
    with open(f"{data}/report.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    """Runs the chosen stage(s)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stage", choices=["build", "topup", "judge", "report", "all"]
    )
    parser.add_argument("--config", default="configs/mix.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["data_dir"], exist_ok=True)
    if args.stage in ("build", "all"):
        asyncio.run(build(cfg))
    if args.stage == "topup":
        asyncio.run(topup(cfg))
    if args.stage in ("judge", "all"):
        asyncio.run(judge(cfg))
    if args.stage in ("report", "all"):
        report(cfg)
    _progress(cfg, stage=args.stage, done=1, total=1, finished=True)


if __name__ == "__main__":
    main()
