"""Synthetic evidence spans: loaders for the sources without human spans.

Usage:
  PYTHONPATH=src .venv/bin/python -m spanjudge.synth preview --n 5
  PYTHONPATH=src .venv/bin/python -m spanjudge.synth pilot \
      --out data/synth/pilot.jsonl
  PYTHONPATH=src .venv/bin/python -m spanjudge.synth review \
      --out data/synth/pilot.jsonl   (writes pilot_review.md next to it)
  PYTHONPATH=src .venv/bin/python -m spanjudge.synth generate \
      --out data/synth/spans.jsonl   (resumable; progress in <out>.log)
  PYTHONPATH=src .venv/bin/python -m spanjudge.synth filter \
      --out data/synth/spans.jsonl   (writes spans_filtered / spans_dropped)

A row is a document (named parts, e.g. Prompt and Response), a question,
its answers (id -> description) and the correct answer. Qwen3.6-27B gets
the correct answer and marks the document text that leads to it
(prompts/evidence.j2). Rows whose answer needs no evidence (WANLI
"undetermined", citation "insufficient") get no spans.

The pilot compares two checks of the spans, always for the gold answer:
  vote: 4 samples at temperature 0.7; a character is evidence when at least
    3 of the 4 samples mark it.
  counterfactual: on the temperature-0 spans. Support is Qwen's p(gold)
    (prompts/score.j2, no answer given). A span is necessary when its
    rewrite (prompts/counterfactual.j2) lowers support by more than tau and
    its paraphrase (prompts/paraphrase.j2) does not move it by tau; it is
    sufficient when, with every other span rewritten, support stays more
    than tau above the all-rewritten support.
p(gold) is recorded, never used to drop a row: the gold label is final.

The full run (`generate`) uses the vote, which won the pilot review: 8
samples at temperature 0.7 from one request, characters marked by at least
5 of 8, span edges snapped outward to word boundaries, and each span's
segments with how many samples marked them (soft labels). When the vote is
empty the temperature-0 spans are used (snapped). No counterfactual test.

Sources: WANLI and the customer-control and citation-control families
(Open-Jev-v1.1) and Aegis 2.0. HelpSteer2 is not here: its 1-5 scores are
holistic, with no locatable evidence (it trains the decision adapter only).
"""

import argparse
import asyncio
import gzip
import itertools
import json
import random
import re
from typing import Any

import datasets as hf
import openai

from spanjudge import llm
from spanjudge import prompts

_OPEN_JEV = (
    "data/external/open-jev-v1.1/raw/community-hard-mix-v2-redistributable/"
    "train.jsonl.gz"
)
_SERVERS = ("http://127.0.0.1:8011/v1", "http://127.0.0.1:8012/v1")
_NO_EVIDENCE = {"undetermined", "insufficient"}
_PILOT = {  # source -> {label: rows}; None spreads rows over all labels
    "wanli": {"supports": 250, "contradicts": 250},
    "aegis": {"yes": 150, "no": 150},
    "customer_control": None,
    "citation_control": {"supported": 50, "contradicted": 50},
}
_PILOT_ANY = 100  # rows for a source with no per-label counts

_UNSAFE = {
    "yes": "Yes: the response is unsafe (harmful content or harmful help).",
    "no": "No: the response is safe.",
}


def row(
    source: str,
    rid: str,
    parts: list[tuple[str, str]],
    question: str,
    labels: dict[str, str],
    label: str,
) -> dict[str, Any]:
    """One row; parts are (name, text) sections of the document."""
    return {
        "id": f"{source}:{rid}",
        "source": source,
        "parts": [{"name": n, "text": t} for n, t in parts],
        "question": question,
        "labels": labels,
        "label": label,
        "needs_spans": label not in _NO_EVIDENCE,
    }


def _open_jev(prefix: str) -> list[dict[str, Any]]:
    with gzip.open(_OPEN_JEV, "rt") as f:
        return [r for r in map(json.loads, f) if r["source"].startswith(prefix)]


def wanli() -> list[dict[str, Any]]:
    """Premise + claim -> supports / contradicts / undetermined (human)."""
    ids = ("supports", "contradicts", "undetermined")
    out = []
    for r in _open_jev("wanli"):
        # The three options, in their shuffled order, by their verb.
        by_id = {next(i for i in ids if i in o): o for o in r["options"]}
        gold = r["options"][r["target"].index(max(r["target"]))]
        out.append(
            row(
                "wanli",
                r["id"],
                [("Premise", r["state"]["premise"])],
                f'Claim: "{r["state"]["claim"]}"\n'
                "Treat the premise as given. Does it support the claim, "
                "contradict it, or leave it undetermined?",
                {i: by_id[i] for i in ids},
                next(i for i in ids if i in gold),
            )
        )
    return out


def customer_control() -> list[dict[str, Any]]:
    """Customer conversation + a yes/no requirement (rule-derived)."""
    out = []
    for r in _open_jev("customer-control"):
        if max(r["target"]) < 1.0:  # the 90 deliberately uncertain rows
            continue
        gold = r["options"][r["target"].index(1.0)]
        out.append(
            row(
                "customer_control",
                r["id"],
                [("Conversation", r["state"])],
                r["question"],
                {o: o.capitalize() for o in r["options"]},
                gold,
            )
        )
    return out


def citation_control() -> list[dict[str, Any]]:
    """Policy source + claim -> supported / contradicted / insufficient."""
    out = []
    for r in _open_jev("citation-control"):
        st = r["state"]
        labels = dict(o.split(": ", 1) for o in r["options"])
        gold = r["options"][r["target"].index(1.0)].split(": ", 1)[0]
        quote = (
            f'\nQuoted from the source: "{st["quote"]}"' if st["quote"] else ""
        )
        out.append(
            row(
                "citation_control",
                r["id"],
                [("Source", st["source"])],
                f'Claim: "{st["claim"]}"{quote}\n{r["question"]}',
                labels,
                gold,
            )
        )
    return out


def aegis() -> list[dict[str, Any]]:
    """Prompt + response -> is the response unsafe? (dataset label).

    Aegis ids name the prompt; one prompt can have several responses (the
    model's, labelled by the LLM jury, and templated refusals added by
    NVIDIA's refusal augmentation), so a row's id adds a hash of its
    response. Exact repeats of a prompt + response are kept once.
    """
    import hashlib

    out, seen = [], set()
    for split in ("train", "validation", "test"):
        ds = hf.load_dataset(
            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split
        )
        for r in ds:
            response = (r["response"] or "").strip()
            if not response:
                continue
            rid = (
                f"{r['id']}-{hashlib.sha256(response.encode()).hexdigest()[:8]}"
            )
            if rid in seen:
                continue
            seen.add(rid)
            x = row(
                "aegis",
                rid,
                [("Prompt", f"User: {r['prompt']}"), ("Response", response)],
                "Is the response unsafe?",
                _UNSAFE,
                "yes" if r["response_label"] == "unsafe" else "no",
            )
            x["label_source"] = r["response_label_source"]
            out.append(x)
    return out


def _nimble_parts(state: Any) -> list[tuple[str, str]]:
    """Document sections for a Nimble state (text, statements or fields)."""
    if isinstance(state, str):
        return [("Document", state)]
    if isinstance(state, list):
        return [("Statements", _nimble_text(state))]
    parts = []
    for key, value in state.items():
        if key == "questions":  # a nested schema, not document text
            continue
        if key == "state" and not isinstance(value, str):
            parts += _nimble_parts(value)
            continue
        parts.append((key.replace("_", " ").title(), _nimble_text(value)))
    return [(n, t) for n, t in parts if t.strip()]


def _nimble_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            (
                f"{v['speaker']}: {v['text']}"
                if isinstance(v, dict) and {"speaker", "text"} <= set(v)
                else _nimble_text(v)
            )
            for v in value
        )
    return json.dumps(value, ensure_ascii=False)


def nimble(split: str) -> list[dict[str, Any]]:
    """Bespoke Labs' Nimble data (github.com/bespokelabsai/nimble, data/).

    Every row gets spans. Choice: the criteria keys; noul: true/false;
    score: the levels 0.. with their rubric text (target is the level).
    """
    out = []
    with open(f"data/external/nimble/{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            q = r["input"]["questions"]["decision"]
            criteria, target = q["criteria"], r["reference"]["target"]
            if q["type"] == "score":
                labels = {str(i): d for i, d in enumerate(criteria)}
                gold = str(target)
            elif q["type"] == "noul":
                labels = {k.lower(): v for k, v in criteria.items()}
                gold = str(target).lower()
            else:
                labels, gold = dict(criteria), target
            x = row(
                f"nimble_{split}",
                r["id"],
                _nimble_parts(r["input"]["state"]),
                q["instructions"],
                labels,
                gold,
            )
            x.update(
                needs_spans=True,
                kind=q["type"],
                domain=r["domain"],
                family=r["family"],
                variant=r["variant"],
            )
            out.append(x)
    return out


LOADERS = {
    "wanli": wanli,
    "aegis": aegis,
    "customer_control": customer_control,
    "citation_control": citation_control,
}
# Not part of the default synthetic set: run with --sources.
ALL_LOADERS = {
    **LOADERS,
    "nimble_train": lambda: nimble("train"),
    "nimble_eval": lambda: nimble("eval"),
}


_QUOTES = str.maketrans(
    {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
)


def _norm(text: str) -> tuple[str, list[int]]:
    """Text with unified quotes and single spaces, and each char's origin."""
    out, origin = [], []
    for i, ch in enumerate(text.translate(_QUOTES)):
        if ch.isspace():
            if out and out[-1] == " ":
                continue
            ch = " "
        out.append(ch)
        origin.append(i)
    return "".join(out), origin


def locate(r: dict[str, Any], span: dict[str, Any]) -> dict[str, Any] | None:
    """Finds a span in its part; None when it is missing or ambiguous.

    The span text is "<before><quote>QUOTE</quote><after>"; without the tags
    (a span at the edge of the text) the whole text is the quote. The
    before/after context picks between repeated quotes.
    """
    text = span.get("text", "")
    m = re.search(r"(.*?)<quote>(.*?)</quote>(.*)", text, re.S)
    before, quote, after = m.groups() if m else ("", text, "")
    if not quote.strip():
        return None
    for part in r["parts"]:
        if span.get("section") != part["name"]:
            continue
        doc, origin = _norm(part["text"])
        b, q, a = (_norm(x)[0] for x in (before, quote, after))
        q = q.strip()
        hits = [x.start() for x in re.finditer(re.escape(q), doc)]
        if len(hits) > 1:  # use the context to choose
            hits = [
                h
                for h in hits
                if doc[:h].endswith(b.lstrip())
                and doc[h + len(q) :].startswith(a.rstrip())
            ] or hits
        if len(hits) != 1:
            return None
        start = origin[hits[0]]
        end = origin[hits[0] + len(q) - 1] + 1
        return {
            "part": part["name"],
            "start": start,
            "end": end,
            "text": part["text"][start:end],
            "reason": span.get("reason", ""),
        }
    return None


def _schema(r: dict[str, Any]) -> dict[str, Any]:
    """JSON schema of the answer (structured output: always parsable)."""
    span = {
        "type": "object",
        "properties": {
            "section": {"enum": [p["name"] for p in r["parts"]]},
            "text": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["section", "text", "reason"],
    }
    return {
        "type": "object",
        "properties": {"spans": {"type": "array", "items": span}},
        "required": ["spans"],
    }


async def propose(
    clients: Any, sem: asyncio.Semaphore, r: dict[str, Any], temp: float
) -> dict[str, Any]:
    """Qwen's evidence spans for the row's correct answer."""
    user = prompts.render(
        "evidence",
        parts=r["parts"],
        question=r["question"],
        labels=r["labels"],
        label=r["label"],
    )
    async with sem:
        out = await next(clients).completions.create(
            model="judge",
            prompt=llm.chat_prefix(user),
            max_tokens=1500,
            temperature=temp,
            extra_body={"structured_outputs": {"json": _schema(r)}},
        )
    raw = out.choices[0].text
    try:
        spans = json.loads(raw)["spans"]
    except (ValueError, KeyError):  # cut off at max_tokens
        return {**r, "raw": raw, "spans": [], "unlocated": None}
    located = [locate(r, s) for s in spans]
    return {
        **r,
        "raw": raw,
        "spans": [s for s in located if s],
        "missing": [
            s["text"] for s, x in zip(spans, located, strict=True) if x is None
        ],
        "unlocated": sum(s is None for s in located),
    }


class Qwen:
    """Client over the vLLM servers, with a call counter.

    Each request goes to the server with the fewest of this client's
    requests in flight, so a slower server (less free GPU memory) gets fewer.
    """

    def __init__(self, concurrency: int):
        self.servers = [  # long timeout: 8 samples can queue for a while
            openai.AsyncOpenAI(base_url=u, api_key="none", timeout=1800)
            for u in _SERVERS
        ]
        self.clients = itertools.cycle(self.servers)  # pilot code
        self.busy = [0] * len(self.servers)
        self.sem = asyncio.Semaphore(concurrency)
        self.calls = 0

    async def create(self, **kwargs: Any) -> Any:
        """One completions request to the least busy server."""
        i = min(range(len(self.servers)), key=self.busy.__getitem__)
        self.busy[i] += 1
        try:
            return await self.servers[i].completions.create(**kwargs)
        finally:
            self.busy[i] -= 1

    async def complete(self, user: str, **kwargs: Any) -> Any:
        """One completion of a user message (no thinking)."""
        async with self.sem:
            self.calls += 1
            out = await self.create(
                model="judge", prompt=llm.chat_prefix(user), **kwargs
            )
        return out.choices[0]


_REWRITE = {
    "type": "object",
    "properties": {"rewrite": {"type": "string"}},
    "required": ["rewrite"],
}


async def score(
    q: Qwen, r: dict[str, Any], parts: list[dict]
) -> dict[str, float] | None:
    """Qwen's answer distribution for the row's question over `parts`."""
    ids = list(r["labels"])
    letters = [chr(ord("A") + i) for i in range(len(ids))]
    user = prompts.render(
        "score",
        parts=parts,
        question=r["question"],
        options=list(zip(letters, r["labels"].values(), strict=True)),
    )
    out = await q.complete(user, max_tokens=1, temperature=0.0, logprobs=20)
    top = out.logprobs.top_logprobs[0] if out.logprobs else {}
    mass = dict.fromkeys(ids, 0.0)
    for tok, lp in top.items():
        letter = tok.strip()
        if letter in letters:
            mass[ids[letters.index(letter)]] += 2.718281828459045**lp
    total = sum(mass.values())
    return {k: v / total for k, v in mass.items()} if total > 0.5 else None


def replace(parts: list[dict], edits: list[tuple]) -> list[dict]:
    """Parts with (part, start, end, new text) edits applied."""
    out = []
    for p in parts:
        text = p["text"]
        mine = sorted(
            (e for e in edits if e[0] == p["name"]), key=lambda e: -e[1]
        )
        for _, start, end, new in mine:
            text = text[:start] + new + text[end:]
        out.append({"name": p["name"], "text": text})
    return out


def disjoint(spans: list[dict]) -> list[dict]:
    """Spans sorted by position, later overlapping ones dropped."""
    out = []
    for s in sorted(spans, key=lambda s: (s["part"], s["start"])):
        if out and out[-1]["part"] == s["part"] and s["start"] < out[-1]["end"]:
            continue
        out.append(s)
    return out


def vote(r: dict[str, Any], samples: list[list[dict]], need: int) -> list:
    """Spans of characters marked by at least `need` samples."""
    out = []
    for p in r["parts"]:
        count = [0] * len(p["text"])
        for spans in samples:
            marked = set()
            for s in spans:
                if s["part"] == p["name"]:
                    marked.update(range(s["start"], s["end"]))
            for i in marked:
                count[i] += 1
        i, text = 0, p["text"]
        while i < len(text):
            if count[i] < need:
                i += 1
                continue
            j = i
            while j < len(text) and count[j] >= need:
                j += 1
            a, b = i, j  # trim spaces at the edges
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if b > a:
                out.append(
                    {"part": p["name"], "start": a, "end": b, "text": text[a:b]}
                )
            i = j
    return out


async def rewrite(
    q: Qwen, r: dict[str, Any], s: dict[str, Any], kind: str
) -> str | None:
    """A counterfactual or paraphrase of span s; None when it failed."""
    user = prompts.render(
        kind,
        parts=r["parts"],
        question=r["question"],
        answer=r["labels"][r["label"]],
        section=s["part"],
        span=s["text"],
    )
    out = await q.complete(
        user,
        max_tokens=max(200, 3 * len(s["text"])),
        temperature=0.0,
        extra_body={"structured_outputs": {"json": _REWRITE}},
    )
    try:
        new = json.loads(out.text)["rewrite"].strip()
    except (ValueError, KeyError):
        return None
    if not new or (kind == "counterfactual" and new == s["text"]):
        return None
    return new


async def counterfactual(
    q: Qwen, r: dict[str, Any], spans: list[dict], s0: float, tau: float
) -> list[dict]:
    """Necessary / sufficient tests of each span (support = p(gold))."""
    if not spans:
        return []
    counters, paras = await asyncio.gather(
        asyncio.gather(*(rewrite(q, r, s, "counterfactual") for s in spans)),
        asyncio.gather(*(rewrite(q, r, s, "paraphrase") for s in spans)),
    )
    edit = [
        (s["part"], s["start"], s["end"], c if c is not None else s["text"])
        for s, c in zip(spans, counters, strict=True)
    ]
    variants = [replace(r["parts"], edit)]  # every span rewritten
    for k, s in enumerate(spans):
        variants.append(replace(r["parts"], [edit[k]]))
        para = paras[k] if paras[k] is not None else s["text"]
        variants.append(
            replace(r["parts"], [(s["part"], s["start"], s["end"], para)])
        )
        variants.append(replace(r["parts"], edit[:k] + edit[k + 1 :]))
    dists = await asyncio.gather(*(score(q, r, v) for v in variants))
    sup = [None if d is None else d[r["label"]] for d in dists]
    tests = []
    for k, s in enumerate(spans):
        s_all, s_c, s_p, s_o = sup[0], *sup[1 + 3 * k : 4 + 3 * k]
        if None in (counters[k], paras[k], s_all, s_c, s_p, s_o):
            tests.append({**s, "tested": False})
            continue
        necessary = abs(s0 - s_p) < tau and s0 - s_c > tau
        sufficient = s_o - s_all > tau
        tests.append(
            {
                **s,
                "tested": True,
                "counter": counters[k],
                "paraphrase": paras[k],
                "support": {
                    "counter": s_c,
                    "paraphrase": s_p,
                    "others": s_o,
                    "all": s_all,
                },
                "necessary": necessary,
                "sufficient": sufficient,
                "kept": necessary or sufficient,
            }
        )
    return tests


async def pilot_row(q: Qwen, r: dict[str, Any], tau: float) -> dict[str, Any]:
    """Both checks on one row."""
    dist, greedy, *samples = await asyncio.gather(
        score(q, r, r["parts"]),
        propose(q.clients, q.sem, r, 0.0),
        *(propose(q.clients, q.sem, r, 0.7) for _ in range(4)),
    )
    q.calls += 5  # propose() calls the servers directly
    s0 = None if dist is None else dist[r["label"]]
    spans = disjoint(greedy["spans"])
    tests = [] if s0 is None else await counterfactual(q, r, spans, s0, tau)
    return {
        **{k: r[k] for k in ("id", "source", "parts", "question", "labels")},
        "label": r["label"],
        "p_gold": s0,
        "dist": dist,
        "greedy": tests or spans,
        "samples": [x["spans"] for x in samples],
        "vote": vote(r, [x["spans"] for x in samples], 3),
    }


def _overlap(a: list[dict], b: list[dict]) -> float | None:
    """Share of a's characters that b also marks."""
    ca = {(s["part"], i) for s in a for i in range(s["start"], s["end"])}
    cb = {(s["part"], i) for s in b for i in range(s["start"], s["end"])}
    return len(ca & cb) / len(ca) if ca else None


def summarize(results: list[dict[str, Any]]) -> None:
    """Per source: span counts, test outcomes and agreement of the checks."""
    by: dict[str, list] = {}
    for x in results:
        by.setdefault(x["source"], []).append(x)
    head = (
        f"{'source':17}{'rows':>5}{'p<.2':>6}{'greedy':>7}{'tested':>7}"
        f"{'nec':>6}{'suff':>6}{'kept':>6}{'vote':>6}{'agree':>7}"
    )
    print(head)
    for name, xs in sorted(by.items()):
        g = [s for x in xs for s in x["greedy"]]
        t = [s for s in g if s.get("tested")]
        low = sum(x["p_gold"] is not None and x["p_gold"] < 0.2 for x in xs)
        ov = [
            o for x in xs if (o := _overlap(x["greedy"], x["vote"])) is not None
        ]

        def pct(k: int, n: int) -> str:
            return f"{k / n:5.0%}" if n else "    -"

        print(
            f"{name:17}{len(xs):5}{pct(low, len(xs)):>6}"
            f"{len(g) / len(xs):7.2f}{pct(len(t), len(g)):>7}"
            f"{pct(sum(s['necessary'] for s in t), len(t)):>6}"
            f"{pct(sum(s['sufficient'] for s in t), len(t)):>6}"
            f"{pct(sum(s['kept'] for s in t), len(t)):>6}"
            f"{sum(len(x['vote']) for x in xs) / len(xs):6.2f}"
            f"{(sum(ov) / len(ov) if ov else 0):7.0%}"
        )
    print(
        "greedy/vote = spans per row; tested/nec/suff/kept = shares of greedy"
        " spans; agree = share of greedy span characters the vote also marks;"
        " p<.2 = rows where Qwen gives the gold answer < 0.2 (kept anyway)."
    )


async def pilot(out: str, seed: int, concurrency: int, tau: float) -> None:
    """Runs both checks on the pilot rows and writes them to `out`."""
    import time

    rng = random.Random(seed)
    picks = []
    for name, counts in _PILOT.items():
        rows = [x for x in LOADERS[name]() if x["needs_spans"]]
        rng.shuffle(rows)
        if counts is None:
            picks += rows[:_PILOT_ANY]
            continue
        for label, n in counts.items():
            picks += [x for x in rows if x["label"] == label][:n]
    q = Qwen(concurrency)
    start = time.time()
    results = await asyncio.gather(*(pilot_row(q, r, tau) for r in picks))
    elapsed = time.time() - start
    with open(out, "w") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summarize(results)
    print(
        f"{len(results)} rows, {q.calls} calls, {elapsed / 60:.1f} min "
        f"({q.calls / elapsed:.1f} calls/s); wrote {out}"
    )


def snap(text: str, start: int, end: int) -> tuple[int, int]:
    """Widens [start, end) so it neither starts nor ends inside a word."""
    while (
        0 < start < len(text)
        and text[start - 1].isalnum()
        and text[start].isalnum()
    ):
        start -= 1
    while (
        0 < end < len(text) and text[end - 1].isalnum() and text[end].isalnum()
    ):
        end += 1
    return start, end


def snapped(r: dict[str, Any], spans: list[dict]) -> list[dict]:
    """Spans snapped to word boundaries, merged where they now overlap."""
    out = []
    for s in sorted(spans, key=lambda s: (s["part"], s["start"])):
        text = next(p["text"] for p in r["parts"] if p["name"] == s["part"])
        a, b = snap(text, s["start"], s["end"])
        if out and out[-1]["part"] == s["part"] and a <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], b)
            out[-1]["text"] = text[out[-1]["start"] : out[-1]["end"]]
            continue
        out.append({"part": s["part"], "start": a, "end": b, "text": text[a:b]})
    return out


def segments(r: dict[str, Any], samples: list[list[dict]]) -> list[list]:
    """[part, start, end, samples marking it] for every marked stretch."""
    out = []
    for p in r["parts"]:
        count = [0] * len(p["text"])
        for spans in samples:
            marked = set()
            for s in spans:
                if s["part"] == p["name"]:
                    marked.update(range(s["start"], s["end"]))
            for i in marked:
                count[i] += 1
        i = 0
        while i < len(count):
            if not count[i]:
                i += 1
                continue
            j = i
            while j < len(count) and count[j] == count[i]:
                j += 1
            out.append([p["name"], i, j, count[i]])
            i = j
    return out


async def sample(
    q: Qwen, r: dict[str, Any], n: int, temp: float, think: int = 0
) -> tuple[list[list[dict]], list[dict]]:
    """Located span samples (n of them) for the gold answer, one request.

    Also returns each sample's raw response: its text, finish reason and
    located spans (None where a quoted span could not be found).

    With think > 0, each sample first reasons freely (up to `think` tokens,
    at temperature temp, n samples in one request), then writes its JSON
    under the grammar, continuing its own reasoning at temperature 0. The
    servers run without a reasoning parser, so the two steps are separate
    requests; prefix caching makes the second one cheap.
    """
    user = prompts.render(
        "evidence",
        parts=r["parts"],
        question=r["question"],
        labels=r["labels"],
        label=r["label"],
    )
    if think:
        head = llm.chat_prefix(user, think=True)
        async with q.sem:
            q.calls += 1
            thought = await q.create(
                model="judge",
                prompt=head,
                max_tokens=think,
                temperature=temp,
                n=n,
                stop=["</think>"],
            )

        async def answer(c: Any) -> Any:
            async with q.sem:
                q.calls += 1
                out = await q.create(
                    model="judge",
                    prompt=head + c.text.rstrip() + "\n</think>\n\n",
                    max_tokens=1500,
                    temperature=0.0,
                    extra_body={"structured_outputs": {"json": _schema(r)}},
                )
            return out.choices[0]

        choices = await asyncio.gather(*map(answer, thought.choices))
        thinking = [(c.text, c.finish_reason) for c in thought.choices]
    else:
        async with q.sem:
            q.calls += 1
            out = await q.create(
                model="judge",
                prompt=llm.chat_prefix(user),
                max_tokens=1500,
                temperature=temp,
                n=n,
                extra_body={"structured_outputs": {"json": _schema(r)}},
            )
        choices, thinking = out.choices, [None] * n
    samples, responses = [], []
    for choice, thought_ in zip(choices, thinking, strict=True):
        try:
            spans = json.loads(choice.text)["spans"]
        except (ValueError, KeyError):  # cut off at max_tokens
            spans = []
        located = [locate(r, s) for s in spans]
        samples.append(disjoint([x for x in located if x]))
        responses.append(
            {
                "text": choice.text,
                "finish_reason": choice.finish_reason,
                "located": located,
                **(
                    {"thinking": thought_[0], "thinking_finish": thought_[1]}
                    if thought_
                    else {}
                ),
            }
        )
    return samples, responses


async def generate_row(
    q: Qwen, r: dict[str, Any], n: int, need: int, think: int = 0
) -> dict[str, Any]:
    """Final spans for one row (empty when the answer needs none)."""
    keep = ("id", "source", "parts", "question", "labels", "label")
    base = {k: r[k] for k in keep}
    if not r["needs_spans"]:
        return {**base, "spans": [], "method": "none", "segments": []}
    samples, responses = await sample(q, r, n, 0.7, think)
    spans, method, greedy_response = (
        snapped(r, vote(r, samples, need)),
        "vote",
        None,
    )
    if not spans:
        (greedy,), (greedy_response,) = await sample(q, r, 1, 0.0, think)
        spans, method = snapped(r, greedy), "greedy_fallback"
    return {
        **base,
        "spans": spans,
        "method": method,
        "n_samples": n,
        "segments": segments(r, samples),
        "responses": responses,  # the n raw samples (temperature 0.7)
        "greedy_response": greedy_response,  # temperature 0, if used
    }


async def generate(
    out: str,
    seed: int,
    concurrency: int,
    n: int,
    need: int,
    sources: list[str] | None = None,
    think: int = 0,
):
    """Writes final spans for every row of the sources; resumable."""
    import pathlib
    import time

    done = set()
    if pathlib.Path(out).exists():
        with open(out) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    loaders = [ALL_LOADERS[name] for name in sources or LOADERS]
    rows = [x for load in loaders for x in load() if x["id"] not in done]
    random.Random(seed).shuffle(rows)  # sources interleaved in the output
    q = Qwen(concurrency)
    log = open(out + ".log", "a")
    start, finished, total = time.time(), 0, len(rows)
    print(f"{len(done)} rows already done, {total} to go", file=log, flush=True)

    async def one(r: dict[str, Any]) -> dict[str, Any] | None:
        for attempt in range(3):
            try:
                return await generate_row(q, r, n, need, think)
            except openai.APIError as e:  # retry transient server errors
                print(
                    f"retry {attempt + 1} {r['id']}: {e}", file=log, flush=True
                )
                await asyncio.sleep(10 * (attempt + 1))
        return None

    with open(out, "a") as f:
        for fut in asyncio.as_completed([one(r) for r in rows]):
            x = await fut
            finished += 1
            if x is not None:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                f.flush()
            if finished % 500 == 0 or finished == total:
                rate = finished / (time.time() - start)
                eta = (total - finished) / rate / 3600
                print(
                    f"{time.strftime('%H:%M:%S')} {finished}/{total} rows, "
                    f"{rate:.2f} rows/s, {q.calls} requests, ETA {eta:.1f} h",
                    file=log,
                    flush=True,
                )
    print("done", file=log, flush=True)


# customer-control questions whose "no" means something is absent: there is
# no text to point at, so those rows become no-span rows.
_ABSENCE = (
    "The user is explicitly asking for a refund or credit",
    "The user describes specific steps to reproduce the issue",
)


def filter_reason(x: dict[str, Any]) -> str | None:
    """Why a generated row is dropped, or None to keep it.

    Chosen against Sonnet reviews of 116 rated and 28 low-agreement rows
    (data/synth/final_review/): a stretch marked by at least 6 of 8 samples
    dropped no good row, unlike a pairwise-IoU threshold, which mostly
    removed good rows whose samples differed only at the span edges.
    """
    if x["method"] == "none":
        return None
    spans = x["spans"]
    if not spans:
        return "empty"
    if x["method"] == "greedy_fallback":
        size = sum(len(p["text"]) for p in x["parts"])
        if sum(s["end"] - s["start"] for s in spans) / max(size, 1) > 0.5:
            return "fallback_covers_most"
    if not any(count >= 6 for *_, count in x["segments"]):
        return "no_core_6_of_8"
    if x["source"] == "aegis" and all(s["part"] == "Prompt" for s in spans):
        return "prompt_only"
    if x["source"] == "citation_control" and all(
        len(s["text"].split()) <= 3 for s in spans
    ):
        return "bare_value"
    return None


def apply_filter(path: str) -> None:
    """Writes <path>_filtered.jsonl and <path>_dropped.jsonl (with reasons)."""
    import collections

    stem = path.rsplit(".", 1)[0]
    kept = dropped = 0
    reasons: collections.Counter = collections.Counter()
    with (
        open(path) as f,
        open(stem + "_filtered.jsonl", "w") as fk,
        open(stem + "_dropped.jsonl", "w") as fd,
    ):
        for line in f:
            x = json.loads(line)
            if (
                x["source"] == "customer_control"
                and x["label"] == "no"
                and x["question"].startswith(_ABSENCE)
            ):
                x = {**x, "spans": [], "method": "none", "absence": True}
                reasons["absence_to_no_span"] += 1
            why = filter_reason(x)
            if why:
                reasons[why] += 1
                dropped += 1
                fd.write(
                    json.dumps({**x, "dropped": why}, ensure_ascii=False) + "\n"
                )
            else:
                kept += 1
                fk.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"kept {kept}, dropped {dropped}; {dict(reasons)}")


def review(path: str) -> str:
    """Writes a readable Markdown review of pilot results; returns its path."""
    with open(path) as f:
        results = [json.loads(line) for line in f]
    out = path.rsplit(".", 1)[0] + "_review.md"
    lines = ["# Synthetic span pilot: review", ""]
    for x in sorted(results, key=lambda x: (x["source"], x["label"])):
        p = "-" if x["p_gold"] is None else f"{x['p_gold']:.2f}"
        lines += [
            f"## {x['id']}",
            f"**Answer:** {x['label']} &nbsp; **Qwen p(answer):** {p}",
            "",
            f"**Question:** {x['question']}",
            "",
        ]
        for part in x["parts"]:
            text, more = part["text"][:3000], len(part["text"]) - 3000
            cut = f"\n[... {more} more characters]" if more > 0 else ""
            lines += [f"**[{part['name']}]**", "```", text + cut, "```"]
        lines.append("**Spans (temperature 0) and counterfactual test:**")
        for s in x["greedy"]:
            if not s.get("tested"):
                lines.append(f"- `{s['text']}` ({s['part']}): not tested")
                continue
            verdict = "KEPT" if s["kept"] else "DROPPED"
            why = [
                w
                for w, v in (
                    ("necessary", s["necessary"]),
                    ("sufficient", s["sufficient"]),
                )
                if v
            ]
            sup = s["support"]
            lines += [
                f"- **{verdict}** ({', '.join(why) or 'neither'}) "
                f"`{s['text']}` ({s['part']})",
                f"  - counterfactual: `{s['counter']}` -> p(answer) "
                f"{sup['counter']:.2f}",
                f"  - paraphrase: `{s['paraphrase']}` -> p(answer) "
                f"{sup['paraphrase']:.2f}",
                f"  - others rewritten {sup['others']:.2f}, all rewritten "
                f"{sup['all']:.2f}",
            ]
        if not x["greedy"]:
            lines.append("- (no spans)")
        vote = "; ".join(f"`{v['text']}`" for v in x["vote"]) or "(none)"
        lines += ["", f"**Vote (3 of 4 samples):** {vote}", "", "---", ""]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


async def preview(n: int, seed: int, concurrency: int) -> None:
    """Prints n rows per source, spread over the answers, with spans."""
    rng = random.Random(seed)
    picks = []
    for load in LOADERS.values():
        rows = [x for x in load() if x["needs_spans"]]
        by_label: dict[str, list] = {}
        for x in rows:
            by_label.setdefault(x["label"], []).append(x)
        cycle = itertools.cycle(sorted(by_label))
        chosen = []
        while len(chosen) < n:
            pool = by_label[next(cycle)]
            chosen.append(pool.pop(rng.randrange(len(pool))))
        picks += chosen
    clients = itertools.cycle(
        [openai.AsyncOpenAI(base_url=u, api_key="none") for u in _SERVERS]
    )
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(propose(clients, sem, r, 0.0) for r in picks)
    )
    for r in results:
        print(f"\n### {r['id']}  answer={r['label']}")
        print(f"Q: {r['question']}")
        for p in r["parts"]:
            text, more = p["text"][:1500], len(p["text"]) - 1500
            cut = f" ... [{more} more chars not shown]" if more > 0 else ""
            print(f"[{p['name']}] {text!r}{cut}")
        for s in r["spans"]:
            print(f"  SPAN ({s['part']}): {s['text']!r} -- {s['reason']}")
        if r["unlocated"]:
            print(
                f"  ({r['unlocated']} span(s) could not be found in the text)"
            )
        if r["unlocated"] is None:
            print(f"  (unparsable output: {r['raw']!r})")


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stage", choices=["preview", "pilot", "review", "generate", "filter"]
    )
    parser.add_argument("--out", default="data/synth/pilot.jsonl")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=480)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--need", type=int, default=5, help="votes to keep")
    parser.add_argument(
        "--sources", nargs="*", help="generate: loaders to run (default: all)"
    )
    parser.add_argument(
        "--think",
        type=int,
        default=0,
        help="generate: reasoning tokens per sample before the spans (0: none)",
    )
    args = parser.parse_args()
    if args.stage == "preview":
        asyncio.run(preview(args.n, args.seed, args.concurrency))
    elif args.stage == "review":
        print(review(args.out))
    elif args.stage == "filter":
        apply_filter(args.out)
    elif args.stage == "generate":
        asyncio.run(
            generate(
                args.out,
                args.seed,
                args.concurrency,
                args.samples,
                args.need,
                args.sources,
                args.think,
            )
        )
    else:
        asyncio.run(pilot(args.out, args.seed, args.concurrency, args.tau))


if __name__ == "__main__":
    main()
