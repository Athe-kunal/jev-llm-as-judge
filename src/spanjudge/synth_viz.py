"""Viewer for the synthetic span pilot: browse rows, checks and reviews.

Usage:
  PYTHONPATH=src .venv/bin/python -m spanjudge.synth_viz \
      --data data/synth/pilot.jsonl [--port 8779]

Shows each row's document with its spans (temperature-0 spans colored by
the counterfactual verdict, vote spans underlined), the per-span rewrites
and p(gold) numbers, and lets the reviewer mark a row good / bad / unsure
with a note. Reviews are appended to <data>_reviews.jsonl (the latest entry
per row wins).
"""

import argparse
import json
import pathlib
import time
from typing import Any

import fastapi
from fastapi import responses
import pydantic
import uvicorn

_HERE = pathlib.Path(__file__).parent


def _chars(spans: list[dict]) -> set:
    return {(s["part"], i) for s in spans for i in range(s["start"], s["end"])}


def _flags(x: dict[str, Any]) -> list[str]:
    """Tags used by the filters."""
    tested = [s for s in x["greedy"] if s.get("tested")]
    flags = []
    if any(not s["kept"] for s in tested):
        flags.append("cf_dropped")
    if tested and not any(s["kept"] for s in tested):
        flags.append("no_kept")
    if x["p_gold"] is not None and x["p_gold"] < 0.2:
        flags.append("p_low")
    vote = _chars(x["vote"])
    for s in tested:
        cs = _chars([s])
        in_vote = len(cs & vote) / len(cs) >= 0.5
        if in_vote != s["kept"]:
            flags.append("disagree")
            break
    if not x["greedy"]:
        flags.append("no_spans")
    return flags


class Review(pydantic.BaseModel):
    """One reviewer verdict."""

    id: str
    verdict: str  # good | bad | unsure
    note: str = ""


def make_app(data: pathlib.Path) -> fastapi.FastAPI:
    """The viewer app for one pilot file."""
    rows = [json.loads(line) for line in data.read_text().splitlines() if line]
    rows.sort(key=lambda x: (x["source"], x["label"], x["id"]))
    for x in rows:
        x["flags"] = _flags(x)
    reviews_path = data.with_name(data.stem + "_reviews.jsonl")
    reviews: dict[str, dict] = {}
    if reviews_path.exists():
        for line in reviews_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                reviews[r["id"]] = r
    app = fastapi.FastAPI()

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return (_HERE / "synth_viz.html").read_text()

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        labels: dict[str, list] = {}
        for x in rows:
            labels.setdefault(x["source"], [])
            if x["label"] not in labels[x["source"]]:
                labels[x["source"]].append(x["label"])
        return {"data": str(data), "rows": len(rows), "labels": labels}

    @app.get("/api/list")
    def listing(
        source: str = "", label: str = "", flag: str = "", review: str = ""
    ) -> dict[str, Any]:
        keep = []
        for i, x in enumerate(rows):
            if source and x["source"] != source:
                continue
            if label and x["label"] != label:
                continue
            if flag and flag not in x["flags"]:
                continue
            verdict = reviews.get(x["id"], {}).get("verdict", "")
            if review == "none" and verdict:
                continue
            if review and review != "none" and verdict != review:
                continue
            keep.append(i)
        return {"idx": keep}

    @app.get("/api/row/{i}")
    def row(i: int) -> dict[str, Any]:
        x = rows[i]
        return {"i": i, "row": x, "review": reviews.get(x["id"])}

    @app.post("/api/review")
    def save(r: Review) -> dict[str, Any]:
        entry = {**r.model_dump(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        reviews[r.id] = entry
        with open(reviews_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        counts: dict[str, int] = {}
        for v in reviews.values():
            counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        return {"saved": True, "counts": counts}

    @app.get("/api/progress")
    def progress() -> dict[str, Any]:
        counts: dict[str, int] = {}
        for v in reviews.values():
            counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        return {"counts": counts, "total": len(rows)}

    return app


def main() -> None:
    """Serves the viewer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/synth/pilot.jsonl")
    parser.add_argument("--port", type=int, default=8779)
    args = parser.parse_args()
    app = make_app(pathlib.Path(args.data))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
