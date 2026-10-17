"""Viewer: what each Reason+Long-12k dataset looks like, a few rows each.

Usage:
  PYTHONPATH=src .venv/bin/python -m spanjudge.data_viz
  PYTHONPATH=src .venv/bin/python -m spanjudge.data_viz --per 3 --seed 7 --port 8791
  PYTHONPATH=src .venv/bin/python -m spanjudge.data_viz \
      --data data/splits/reason_long_12k/train.jsonl

Picks --per rows per dataset with a seeded RNG (for proofwriter, contractnli
and musique, two of them are rows with native evidence spans) and serves one
HTML page: the datasets with row counts, each dataset's description and its
stats from manifest.json (options, label distribution, tokens), then the
example rows (question, options with the gold one highlighted, document parts
in scrollable boxes, evidence spans marked).
"""

import argparse
import collections
import html
import json
import pathlib
import random
from typing import Any

import fastapi
from fastapi import responses
import uvicorn

_DEFAULT_DATA = "data/splits/reason_long_12k/train.jsonl"
# Span-carrying sources: this many of the --per rows come from the span pool.
_SPAN_SOURCES = {"proofwriter", "contractnli", "musique"}
_SPAN_ROWS = 2
# Option lists at least this long are collapsed (the gold option stays visible).
_COLLAPSE_AT = 60
# Label distributions longer than this are collapsed after the top entries.
_TOP_LABELS = 12

# source -> (title, description); descriptions follow docs/data.md.
DATASETS: dict[str, tuple[str, str]] = {
    "clinc150": (
        "CLINC150",
        "Short user messages, each labelled with one of 151 intents across 10 "
        "domains, including out-of-scope (`oos`). The option ids are the intent "
        "names.",
    ),
    "massive": (
        "MASSIVE (en-US)",
        "Voice-assistant utterances labelled with one of 60 intents. The option "
        "ids are the intent names.",
    ),
    "bitext": (
        "Bitext",
        "E-commerce customer-support messages labelled with one of 27 intents. "
        "Messages keep the release's `{{...}}` placeholders.",
    ),
    "reclor": (
        "ReClor",
        "Logical-reasoning argument passages with a 4-option question about the "
        "argument. Research-only license.",
    ),
    "logiqa2": (
        "LogiQA 2.0",
        "Short logical-reasoning passages (civil-service style) with a 4-option "
        "question.",
    ),
    "cosmosqa": (
        "CosmosQA",
        "Commonsense reading comprehension: a short everyday context with a "
        "question and 4 options.",
    ),
    "piqa": (
        "PIQA",
        "Physical commonsense: a goal and two candidate solutions; pick the one "
        "that better achieves the goal.",
    ),
    "anli": (
        "αNLI (ART)",
        "Abductive reasoning: two observations and two hypotheses; pick the one "
        "that best explains what happened between them.",
    ),
    "proofwriter": (
        "ProofWriter (OWA, depth 5)",
        "A theory of facts and rules, and a statement; is it true, false, or "
        "unknown given the theory? True and false rows carry proof-step sentences "
        "as spans.",
    ),
    "contractnli": (
        "ContractNLI",
        "A long contract and a hypothesis statement: entailed, contradicted, or "
        "not mentioned. Entailed and contradicted rows carry the human evidence "
        "spans.",
    ),
    "musique": (
        "MuSiQue (full v1.0)",
        "Twenty Wikipedia paragraphs and a multi-hop question; yes or no, are the "
        "paragraphs enough to answer it? Answerable rows carry the supporting "
        "paragraphs as spans.",
    ),
    "quality": (
        "QuALITY (v1.0.1, HTML-stripped)",
        "Long story articles with a 4-option reading question; the gold answer is "
        "the annotators' majority.",
    ),
}

_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1f23;--muted:#6b6f76;--card:#ffffff;--line:#e2e0da;
--mark:#ffe58a;--markfg:#1d1f23;--gold:#d8f5e1;--goldline:#1f7a3d;--chip:#f0efea;
--code:#f4f3ef;--warn:#8a5a00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
--bg:#15171b;--fg:#e6e6e6;--muted:#9aa1ac;--card:#1e2127;--line:#343842;
--mark:#8a6a00;--markfg:#fff6d0;--gold:#173a25;--goldline:#7fdca0;--chip:#2a2e36;
--code:#22252c;--warn:#f0c674}}
:root[data-theme="dark"]{--bg:#15171b;--fg:#e6e6e6;--muted:#9aa1ac;--card:#1e2127;
--line:#343842;--mark:#8a6a00;--markfg:#fff6d0;--gold:#173a25;--goldline:#7fdca0;
--chip:#2a2e36;--code:#22252c;--warn:#f0c674}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
display:grid;grid-template-columns:250px minmax(0,1fr)}
.side{position:sticky;top:0;height:100vh;overflow:auto;padding:16px;
border-right:1px solid var(--line)}
.side h1{font-size:16px;margin:0 0 4px}
.side p{margin:0 0 12px;color:var(--muted);font-size:13px}
.side ol{list-style:none;padding:0;margin:0}
.side li{display:flex;justify-content:space-between;gap:8px;margin:2px 0}
.side a{color:var(--fg);text-decoration:none;padding:3px 6px;border-radius:4px;flex:1}
.side a:hover{background:var(--chip)}
.side .n{color:var(--muted);font-variant-numeric:tabular-nums;font-size:13px;padding-top:3px}
main{padding:20px 28px;min-width:0}
h2{margin:0 0 4px;font-size:21px}
h2 .src{color:var(--muted);font-weight:normal;font-size:14px;font-family:ui-monospace,monospace}
h4{margin:14px 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
section.ds{padding:8px 0 24px;margin-bottom:24px;border-bottom:2px solid var(--line)}
.desc{margin:6px 0 12px;max-width:80ch}
.stats{display:flex;flex-wrap:wrap;gap:8px 18px;margin:0 0 6px;font-size:14px}
.stats b{font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0}
.chip{background:var(--chip);border-radius:4px;padding:1px 7px;font-size:13px;
font-family:ui-monospace,monospace}
.chip b{font-family:system-ui,sans-serif;margin-left:4px}
.muted{color:var(--muted)}
.ex{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 16px 16px;margin:14px 0;min-width:0}
.ex header{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px}
.ex code{font-size:12px;color:var(--muted);word-break:break-all}
.badge{border-radius:4px;padding:1px 7px;font-size:12px;border:1px solid var(--line)}
.badge.gold{background:var(--gold);border-color:var(--goldline);color:var(--fg)}
.badge.spans{color:var(--warn);border-color:var(--warn)}
.question{white-space:pre-wrap;word-break:break-word;background:var(--code);
border-radius:6px;padding:8px 10px;margin:0}
ul.opts{list-style:none;padding:0;margin:4px 0}
ul.opts li{padding:3px 8px;border-radius:4px;margin:2px 0;word-break:break-word}
ul.opts li.gold{background:var(--gold);box-shadow:inset 4px 0 0 var(--goldline)}
.oid{display:inline-block;min-width:3em;font-family:ui-monospace,monospace;color:var(--muted)}
.gtag{font-size:12px;color:var(--goldline);font-weight:600;margin-left:6px}
.goldline{padding:6px 10px;border-radius:4px;margin:6px 0;background:var(--gold);
box-shadow:inset 4px 0 0 var(--goldline);word-break:break-word}
details{margin:4px 0}
summary{cursor:pointer;color:var(--muted);font-size:13px}
.part{margin:8px 0}
.pname{font-size:13px;font-weight:600;margin-bottom:3px}
.ptext{max-height:320px;overflow:auto;white-space:pre-wrap;word-break:break-word;
background:var(--code);border:1px solid var(--line);border-radius:6px;
padding:8px 10px;font-size:13px;line-height:1.45}
mark{background:var(--mark);color:var(--markfg);border-radius:2px;padding:0 1px}
@media (max-width:800px){
body{grid-template-columns:minmax(0,1fr)}
.side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
main{padding:14px 16px}}
"""


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def pick_rows(
    rows: list[dict[str, Any]], per: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Seeded sample of `per` rows for each dataset, in DATASETS order."""
    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    picked: dict[str, list[dict[str, Any]]] = {}
    for src in DATASETS:
        pool = by_src[src]
        chosen: list[dict[str, Any]] = []
        if src in _SPAN_SOURCES:
            with_spans = [r for r in pool if r["spans"]]
            chosen = rng.sample(with_spans, min(_SPAN_ROWS, per, len(with_spans)))
        taken = {r["id"] for r in chosen}
        rest = [r for r in pool if r["id"] not in taken]
        chosen += rng.sample(rest, min(per - len(chosen), len(rest)))
        picked[src] = chosen
    return picked


def _marked(text: str, spans: list[dict[str, Any]]) -> str:
    """Escaped part text with each span wrapped in <mark>."""
    out: list[str] = []
    pos = 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        if sp["start"] < pos:  # overlapping span; the release has none
            continue
        out.append(_esc(text[pos : sp["start"]]))
        out.append(f"<mark>{_esc(text[sp['start'] : sp['end']])}</mark>")
        pos = sp["end"]
    out.append(_esc(text[pos:]))
    return "".join(out)


def _options(row: dict[str, Any]) -> str:
    gold = row["label"]
    opts = row["labels"]
    items = ""
    for k, v in opts.items():
        if k == gold:
            items += (
                f'<li class="gold"><span class="oid">{_esc(k)}</span>'
                f"{_esc(v)}<span class=\"gtag\">gold</span></li>"
            )
        else:
            items += f'<li><span class="oid">{_esc(k)}</span>{_esc(v)}</li>'
    if len(opts) < _COLLAPSE_AT:
        return f'<ul class="opts">{items}</ul>'
    gold_text = opts.get(gold, "")
    return (
        f'<div class="goldline"><span class="oid">{_esc(gold)}</span>'
        f"{_esc(gold_text)}<span class=\"gtag\">gold</span></div>"
        f"<details><summary>All {len(opts)} options</summary>"
        f'<ul class="opts">{items}</ul></details>'
    )


def _example(idx: int, row: dict[str, Any]) -> str:
    spans_by_part: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for sp in row["spans"]:
        spans_by_part[sp["part"]].append(sp)
    badges = f'<span class="badge gold">gold {_esc(row["label"])}</span>'
    if row["spans"]:
        badges += f'<span class="badge spans">{len(row["spans"])} evidence spans</span>'
    parts = ""
    for p in row["parts"]:
        n_sp = len(spans_by_part.get(p["name"], []))
        note = f" · {n_sp} span(s)" if n_sp else ""
        parts += (
            f'<div class="part"><div class="pname">{_esc(p["name"])} '
            f'<span class="muted">{len(p["text"]):,} chars{note}</span></div>'
            f'<div class="ptext">'
            f'{_marked(p["text"], spans_by_part.get(p["name"], []))}</div></div>'
        )
    return (
        f'<article class="ex"><header><b>#{idx}</b> <code>{_esc(row["id"])}</code>'
        f"{badges}</header>"
        f"<h4>Question</h4><div class=\"question\">{_esc(row['question'])}</div>"
        f"<h4>Options ({len(row['labels'])})</h4>{_options(row)}"
        f"<h4>Document parts</h4>{parts}</article>"
    )


def _labels(dist: dict[str, int], total: int) -> str:
    ranked = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))

    def chip(k: str, v: int) -> str:
        return f'<span class="chip">{_esc(k)}<b>{v} ({100 * v / total:.1f}%)</b></span>'

    head = "".join(chip(k, v) for k, v in ranked[:_TOP_LABELS])
    if len(ranked) <= _TOP_LABELS:
        return f'<div class="chips">{head}</div>'
    rest = "".join(chip(k, v) for k, v in ranked[_TOP_LABELS:])
    return (
        f'<div class="chips">{head}</div><details><summary>all '
        f"{len(ranked)} labels</summary><div class=\"chips\">{rest}</div></details>"
    )


def render_page(
    picked: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    counts: dict[str, int],
    data: str,
    per: int,
    seed: int,
) -> str:
    """The one HTML page: sidebar of datasets, then the sampled rows."""
    nav = ""
    body = ""
    for src, (title, desc) in DATASETS.items():
        stats = manifest["datasets"][src]
        tok = stats["train_tokens"]
        nav += (
            f'<li><a href="#{src}">{_esc(title)}</a>'
            f'<span class="n">{counts[src]:,}</span></li>'
        )
        span_note = f"train rows with spans: {stats['train_with_spans']}"
        body += (
            f'<section class="ds" id="{src}"><h2>{_esc(title)} '
            f'<span class="src">{src}</span></h2>'
            f'<p class="desc">{_esc(desc)}</p>'
            f'<div class="stats"><span>rows in file: <b>{counts[src]:,}</b></span>'
            f'<span>options: <b>{stats["options"]}</b></span>'
            f'<span>tokens: median <b>{tok["median"]:,}</b> · max <b>{tok["max"]:,}</b>'
            f"</span><span>{_esc(span_note)}</span></div>"
            f'<h4>Label distribution (train split, manifest)</h4>'
            f"{_labels(stats['train_labels'], sum(stats['train_labels'].values()))}"
            f"<h4>Examples ({len(picked[src])})</h4>"
        )
        body += "".join(_example(i + 1, r) for i, r in enumerate(picked[src]))
        body += "</section>"
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Reason+Long-12k data</title>"
        f"<style>{_CSS}</style></head><body>"
        f'<nav class="side"><h1>Reason+Long-12k</h1>'
        f"<p>{len(DATASETS) * per} examples: {per} per dataset, seed {seed}. "
        f"Source: <code>{_esc(data)}</code></p><ol>{nav}</ol></nav>"
        f"<main>{body}</main></body></html>"
    )


def make_app(data: str, per: int, seed: int) -> fastapi.FastAPI:
    """The viewer app: the sample is drawn once, at startup."""
    with open(data) as f:
        rows = [json.loads(line) for line in f]
    manifest_path = pathlib.Path(data).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    counts = collections.Counter(r["source"] for r in rows)
    picked = pick_rows(rows, per, seed)
    page = render_page(picked, manifest, counts, data, per, seed)
    app = fastapi.FastAPI()

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return page

    return app


def main() -> None:
    """Serves the viewer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=_DEFAULT_DATA)
    parser.add_argument("--per", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    app = make_app(args.data, args.per, args.seed)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
