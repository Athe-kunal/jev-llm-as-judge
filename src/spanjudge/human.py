"""Loaders for the human-span datasets, in the synthetic rows' format.

Usage (counts per dataset, subset and answer):
  PYTHONPATH=src .venv/bin/python -m spanjudge.human

A row: id, source, subset (used to balance sampling), group (rows that share
a document; splits keep a group on one side), parts (named document
sections), question, labels (answer id -> description), label (gold),
spans (part, start, end, text: human evidence; empty for answers with no
evidence such as "no problem") and needs_spans.

Datasets (licences allow sharing with citation): RAGTruth, MQM, QA-Feedback,
Mu-SHROOM, ProcessBench, PRM800K, FEVER (ERASER), HotpotQA (yes/no
questions), ContractNLI and HateXplain. Questions about a response come in
a few wordings each; the wording is picked with the row's seeded RNG.
"""

import ast
import collections
import hashlib
import json
import random
import re
from typing import Any

import datasets as hf
from huggingface_hub import hf_hub_download

_EXTERNAL = "data/external"
_MQM_PENALTY = {"minor": 0.5, "major": 2.0, "critical": 4.0}
_LANGUAGES = {"en": "English", "de": "German", "ru": "Russian", "zh": "Chinese"}


def _problem(problem: str, wordings: tuple[str, ...]) -> dict[str, Any]:
    """A yes/no question about a problem in the response."""
    return {
        "wordings": wordings,
        "labels": {
            "yes": f"Yes: the response has {problem}.",
            "no": f"No: the response does not have {problem}.",
        },
    }


UNSUPPORTED = _problem(
    "information that conflicts with the document or is not supported by it",
    (
        "Does the response contain information that conflicts with the"
        " document or is not supported by it?",
        "Is any part of the response not backed up by the provided document?",
        "Does the answer include claims that the source text does not"
        " support?",
    ),
)
CONFLICT = _problem(
    "statements that the document contradicts",
    (
        "Does the response state anything that the document contradicts?",
        "Does any claim in the response conflict with the provided text?",
    ),
)
BASELESS = _problem(
    "facts that are not in the document",
    (
        "Does the response add facts that the document does not contain?",
        "Does the answer introduce information that cannot be found in the"
        " source?",
    ),
)
MAJOR_ERROR = _problem(
    "a major or critical translation error",
    (
        "Does the translation contain a major or critical error?",
        "Is there a serious mistake in this translation (one that changes or"
        " loses the meaning)?",
    ),
)
IRRELEVANT = _problem(
    "irrelevant or repeated information",
    (
        "Does the answer include information that is irrelevant to the"
        " question or repeated?",
        "Does the response contain off-topic or redundant content?",
    ),
)
UNGROUNDED = _problem(
    "facts that are wrong or not supported by the passages",
    (
        "Does the answer contain facts that are wrong or not supported by the"
        " passages?",
        "Is any statement in the response unverifiable or misgrounded given"
        " the passages?",
    ),
)
ANY_ERROR = _problem(
    "irrelevant, repeated, incoherent, wrong or unverifiable content",
    (
        "Does the answer have any problem: irrelevant, repeated, incoherent,"
        " wrong or unverifiable content?",
        "Is there anything in the response that should be removed or"
        " corrected?",
    ),
)
HALLUCINATION = _problem(
    "false or made-up content",
    (
        "Does the response contain false or made-up information?",
        "Does the answer hallucinate any facts?",
        "Is any part of the response factually incorrect or invented?",
    ),
)
MATH_ERROR = _problem(
    "a wrong step",
    (
        "Does the solution contain a wrong step?",
        "Is there a mistake in any step of this solution?",
        "Does any step contain an error in reasoning or calculation?",
    ),
)
FIRST_MATH_ERROR = {
    **MATH_ERROR,
    "wordings": (
        "Does the solution go wrong somewhere? Mark the first wrong step.",
        "Is any step incorrect? If so, mark the earliest one.",
    ),
}
TRANSLATION = {
    "wordings": (
        "How accurate and fluent is the response as a translation of the"
        " source text?",
        "Rate the quality of this translation.",
        "How good is the translation in the response?",
    ),
    "labels": {
        "1": "Very poor translation quality.",
        "2": "Poor translation quality.",
        "3": "Fair translation quality.",
        "4": "Good translation quality, with minor problems.",
        "5": "Excellent translation quality: nothing to improve.",
    },
}


def _hash(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[
        :16
    ]


def _row(
    source: str,
    key: Any,
    subset: str,
    group: Any,
    parts: list[tuple[str, str]],
    question: str,
    labels: dict[str, str],
    label: str,
    spans: list[tuple[str, int, int]],
) -> dict[str, Any]:
    """One row; spans are (part name, start, end) in that part's text."""
    texts = dict(parts)
    located = sorted(
        {(p, a, b) for p, a, b in spans if 0 <= a < b <= len(texts[p])}
    )
    return {
        "id": f"{source}:{_hash(key, question)}",
        "source": source,
        "subset": subset,
        "group": f"{source}:{_hash(group)}",
        "parts": [{"name": n, "text": t} for n, t in parts],
        "question": question,
        "labels": labels,
        "label": label,
        "spans": [
            {"part": p, "start": a, "end": b, "text": texts[p][a:b]}
            for p, a, b in located
        ],
        "needs_spans": bool(located),
    }


def _ask(family: dict[str, Any], rng: random.Random) -> str:
    return family["wordings"][rng.randrange(len(family["wordings"]))]


def _word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Widens a range to whole words (some annotations cut words short)."""

    def inside(c: str) -> bool:
        cjk = "぀" <= c <= "ヿ" or "一" <= c <= "鿿"
        return (c.isalnum() and not cjk) or c in "#_'-"

    while start > 0 and inside(text[start - 1]) and inside(text[start]):
        start -= 1
    while end < len(text) and inside(text[end]) and inside(text[end - 1]):
        end += 1
    return start, end


def ragtruth(rng: random.Random) -> list[dict[str, Any]]:
    """RAGTruth: unsupported / conflicting / baseless claims in RAG answers.

    Each response gets one question of each kind; its spans are the human
    spans of the matching types. Subset: task type (QA, Summary, Data2txt).
    """
    kinds = [
        (UNSUPPORTED, None),
        (CONFLICT, ("Evident Conflict", "Subtle Conflict")),
        (BASELESS, ("Evident Baseless Info", "Subtle Baseless Info")),
    ]
    rows = []
    for split in ("train", "test"):
        for r in hf.load_dataset("wandb/RAGTruth-processed", split=split):
            out = r["output"]
            labels = [
                x
                for x in json.loads(r["hallucination_labels"])
                if out[x["start"] : x["end"]] == x["text"]
            ]
            context = re.sub(r"\s*output:\s*$", "", r["context"].strip())
            context = f"Document:\n{context}\n\nRequest:\n{r['query']}"
            for family, types in kinds:
                spans = [
                    ("Response", x["start"], x["end"])
                    for x in labels
                    if types is None or x["label_type"] in types
                ]
                rows.append(
                    _row(
                        "ragtruth",
                        (r["id"], family["labels"]["yes"]),
                        r["task_type"],
                        context,
                        [("Prompt", context), ("Response", out)],
                        _ask(family, rng),
                        family["labels"],
                        "yes" if spans else "no",
                        spans,
                    )
                )
    return rows


def _mqm_level(annotations: list[dict]) -> str:
    """Maps the summed MQM penalty of a translation to a 1-5 level."""
    penalty = sum(_MQM_PENALTY.get(a["severity"], 0) for a in annotations)
    for limit, level in ((0.0, "5"), (1.0, "4"), (2.5, "3"), (5.0, "2")):
        if penalty <= limit:
            return level
    return "1"


def mqm(rng: random.Random) -> list[dict[str, Any]]:
    """WMT MQM: 1-5 translation quality and major-error yes/no.

    Translations with an annotation that cannot be located are skipped, so
    the label and the spans describe the same errors. Subset: language pair
    and question kind.
    """
    ds = hf.load_dataset("RicardoRei/wmt-mqm-error-spans", split="train")
    rows = []
    for r in ds:
        mt, lp = r["mt"], r["lp"].split("-")
        if lp[0] not in _LANGUAGES or lp[1] not in _LANGUAGES:
            continue
        annotations = r["annotations"]
        if isinstance(annotations, str):
            annotations = ast.literal_eval(annotations)
        located = [
            a
            for a in annotations
            if a["text"] and 0 <= a["start"] < a["end"] <= len(mt)
        ]
        if len(located) != len(annotations):
            continue
        ranges = [_word_bounds(mt, a["start"], a["end"]) for a in located]
        context = (
            f"Source text ({_LANGUAGES[lp[0]]}): {r['src']}\n"
            f"Task: translate it into {_LANGUAGES[lp[1]]}."
        )
        parts = [("Prompt", context), ("Response", mt)]
        key = (r["src"], mt)
        rows.append(
            _row(
                "mqm",
                (key, "score"),
                f"{r['lp']}/score",
                r["src"],
                parts,
                _ask(TRANSLATION, rng),
                TRANSLATION["labels"],
                _mqm_level(located),
                [("Response", a, b) for a, b in ranges],
            )
        )
        major = [
            ("Response", *rg)
            for rg, a in zip(ranges, located, strict=True)
            if a["severity"] in ("major", "critical")
        ]
        rows.append(
            _row(
                "mqm",
                (key, "major"),
                f"{r['lp']}/major_error",
                r["src"],
                parts,
                _ask(MAJOR_ERROR, rng),
                MAJOR_ERROR["labels"],
                "yes" if major else "no",
                major,
            )
        )
    return rows


def qa_feedback(rng: random.Random) -> list[dict[str, Any]]:
    """QA-Feedback (FineGrainedRLHF): human error spans on long answers.

    Error types: Irrelevant + Redundant -> irrelevant; Unverifiable +
    Wrong-Grounding -> ungrounded; all (Incoherent too) -> any problem,
    skipped when the only problem is missing information (no span).
    """
    with open(f"{_EXTERNAL}/qa_feedback/qa_train_feedback.json") as f:
        data = json.load(f)
    kinds = [
        ("irrelevant", IRRELEVANT, ("Irrelevant", "Redundant")),
        ("ungrounded", UNGROUNDED, ("Unverifiable", "Wrong-Grounding")),
        ("any_problem", ANY_ERROR, None),
    ]
    rows = []
    for i, r in enumerate(data):
        answer = r["prediction 1"]
        passages = "\n".join(
            f"[{p[0]}] " + " ".join(p[1:]) for p in r["passages"]
        )
        context = f"User: {r['question']}\n\nPassages:\n{passages}"
        errors = r["feedback"]["errors"]
        for name, family, types in kinds:
            if types is None and r["feedback"].get("missing-info"):
                continue
            spans = [
                ("Response", e["start"], e["end"])
                for e in errors
                if types is None or e["error type"] in types
            ]
            rows.append(
                _row(
                    "qa_feedback",
                    (i, name),
                    name,
                    r["question"],
                    [("Prompt", context), ("Response", answer)],
                    _ask(family, rng),
                    family["labels"],
                    "yes" if spans else "no",
                    spans,
                )
            )
    return rows


def mushroom(rng: random.Random) -> list[dict[str, Any]]:
    """Mu-SHROOM (SemEval-2025 Task 3): hallucination spans, by language."""
    rows = []
    for split in ("validation", "test"):
        for r in hf.load_dataset("Helsinki-NLP/mu-shroom", "all", split=split):
            hard = r["hard_labels"]
            if isinstance(hard, str):
                hard = None if hard == "None" else ast.literal_eval(hard)
            if hard is None:
                continue
            text = r["model_output_text"]
            if text.rstrip().endswith("<|im_end|>"):
                text = text.rstrip()[: -len("<|im_end|>")]
            spans = [
                ("Response", *_word_bounds(text, a, min(b, len(text))))
                for a, b in hard
                if a < len(text)
            ]
            rows.append(
                _row(
                    "mushroom",
                    (split, r["lang"], r["model_input"], text),
                    r["lang"],
                    r["model_input"],
                    [
                        ("Prompt", f"User: {r['model_input']}"),
                        ("Response", text),
                    ],
                    _ask(HALLUCINATION, rng),
                    HALLUCINATION["labels"],
                    "yes" if spans else "no",
                    spans,
                )
            )
    return rows


def _steps(steps: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Joins solution steps with blank lines; returns each step's range."""
    text, ranges = "", []
    for s in steps:
        if text:
            text += "\n\n"
        ranges.append((len(text), len(text) + len(s)))
        text += s
    return text, ranges


def processbench(rng: random.Random) -> list[dict[str, Any]]:
    """ProcessBench: the earliest wrong step (-1: none). Subset: benchmark."""
    rows = []
    for name in ("gsm8k", "math", "olympiadbench", "omnimath"):
        path = hf_hub_download(
            "Qwen/ProcessBench", f"{name}.json", repo_type="dataset"
        )
        with open(path) as f:
            data = json.load(f)
        for r in data:
            text, ranges = _steps(r["steps"])
            bad = [("Response", *ranges[r["label"]])] if r["label"] >= 0 else []
            rows.append(
                _row(
                    "processbench",
                    r["id"],
                    name,
                    r["problem"],
                    [
                        ("Prompt", f"Problem: {r['problem']}"),
                        ("Response", text),
                    ],
                    _ask(FIRST_MATH_ERROR, rng),
                    FIRST_MATH_ERROR["labels"],
                    "yes" if bad else "no",
                    bad,
                )
            )
    return rows


def prm800k(rng: random.Random) -> list[dict[str, Any]]:
    """PRM800K: human step ratings; steps rated -1 are the spans.

    Raters stop at the first wrong step, so a wrong solution is taken up to
    and including its first -1 step. A correct solution loses its "# Answer"
    line and is cut at a random step, so neither class has a telltale
    ending. Read from the raw JSONL files (the hub's dataset loader fails on
    their schema). Subset: phase.
    """
    rows, seen = [], set()
    for phase in ("phase1", "phase2"):
        for part in ("train", "test"):
            path = hf_hub_download(
                "tasksource/PRM800K",
                f"{phase}_{part}.jsonl",
                repo_type="dataset",
            )
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    row = _prm_row(r, phase, rng, seen)
                    if row:
                        rows.append(row)
    return rows


def _prm_row(
    r: dict[str, Any], phase: str, rng: random.Random, seen: set
) -> dict[str, Any] | None:
    steps, ratings = [], []
    for st in (r.get("label") or {}).get("steps") or []:
        if ratings and ratings[-1] == -1:
            break  # the response ends at its first wrong step
        completions = st.get("completions") or []
        k = st.get("chosen_completion")
        if k is None:  # the rater stopped: keep a wrong completion
            wrong = [c for c in completions if c.get("rating") == -1]
            if not wrong:
                return None
            steps.append(wrong[0]["text"])
            ratings.append(-1)
            break
        if k >= len(completions) or completions[k].get("rating") is None:
            return None
        steps.append(completions[k]["text"])
        ratings.append(completions[k]["rating"])
    while steps and steps[-1].lstrip().startswith("# Answer"):
        steps.pop()
        ratings.pop()
    if not steps:
        return None
    if -1 not in ratings:
        cut = rng.randint(1, len(steps))
        steps, ratings = steps[:cut], ratings[:cut]
    text, ranges = _steps(steps)
    if text in seen:
        return None
    seen.add(text)
    problem = r["question"]["problem"]
    bad = [
        ("Response", *rg)
        for rg, x in zip(ranges, ratings, strict=True)
        if x == -1
    ]
    return _row(
        "prm800k",
        text,
        phase,
        problem,
        [("Prompt", f"Problem: {problem}"), ("Response", text)],
        _ask(MATH_ERROR, rng),
        MATH_ERROR["labels"],
        "yes" if bad else "no",
        bad,
    )


_PTB = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
}
_NO_SPACE_BEFORE = {
    ".",
    ",",
    ":",
    ";",
    "?",
    "!",
    "%",
    ")",
    "]",
    "}",
    "'s",
    "n't",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    '"',
}


def _detokenize(lines: list[list[str]]) -> tuple[str, list[list[tuple]]]:
    """Readable text from ERASER's tokenized sentences, with token offsets."""
    text, offsets, quote_open = "", [], False
    for tokens in lines:
        line_offsets = []
        if text:
            text += "\n"
        for i, tok in enumerate(tokens):
            tok = _PTB.get(tok, tok)
            glue = (
                i == 0
                or tok in _NO_SPACE_BEFORE
                and not (tok == '"' and not quote_open)
                or text.endswith(("(", "[", "{"))
                or (text.endswith('"') and quote_open and tok != '"')
            )
            if tok == '"':
                quote_open = not quote_open
            if not glue:
                text += " "
            line_offsets.append((len(text), len(text) + len(tok)))
            text += tok
        offsets.append(line_offsets)
    return text, offsets


def fever(rng: random.Random) -> list[dict[str, Any]]:
    """FEVER (ERASER version): claim + Wikipedia intro -> supports/refutes.

    The evidence sentences are the spans. Subset: answer.
    """
    del rng
    base = f"{_EXTERNAL}/eraser/fever"
    labels = {
        "supports": "The document supports the claim.",
        "refutes": "The document refutes the claim.",
    }
    docs: dict[str, tuple] = {}
    rows = []
    for split in ("train", "val", "test"):
        with open(f"{base}/{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                ev = [e for group in r["evidences"] for e in group]
                docid = (r.get("docids") or [e["docid"] for e in ev])[0]
                if docid not in docs:
                    with open(f"{base}/docs/{docid}") as d:
                        lines = [ln.split() for ln in d.read().split("\n")]
                    docs[docid] = _detokenize(lines)
                text, offsets = docs[docid]
                flat = [o for line_off in offsets for o in line_off]
                spans = [
                    (
                        "Document",
                        flat[e["start_token"]][0],
                        flat[e["end_token"] - 1][1],
                    )
                    for e in ev
                    if e["docid"] == docid and e["end_token"] <= len(flat)
                ]
                label = r["classification"].lower()
                rows.append(
                    _row(
                        "fever",
                        r["annotation_id"],
                        label,
                        docid,
                        [("Document", text)],
                        f'Claim: "{r["query"]}"\nDoes the document support or'
                        " refute the claim?",
                        labels,
                        label,
                        spans,
                    )
                )
    return rows


def hotpotqa(rng: random.Random) -> list[dict[str, Any]]:
    """HotpotQA (distractor) yes/no questions; supporting sentences = spans.

    Subset: difficulty level.
    """
    del rng
    labels = {"yes": "Yes.", "no": "No."}
    rows = []
    for split in ("train", "validation"):
        ds = hf.load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        for r in ds:
            if r["answer"] not in labels:
                continue
            text, where = "", {}
            for title, sents in zip(
                r["context"]["title"], r["context"]["sentences"], strict=True
            ):
                if text:
                    text += "\n\n"
                text += f"{title}: "
                for k, s in enumerate(sents):
                    where[(title, k)] = (len(text), len(text) + len(s))
                    text += s
            spans = []
            for t, k in zip(
                r["supporting_facts"]["title"],
                r["supporting_facts"]["sent_id"],
                strict=True,
            ):
                if (t, k) in where:
                    a, b = where[(t, k)]
                    while a < b and text[a].isspace():
                        a += 1
                    spans.append(("Document", a, b))
            rows.append(
                _row(
                    "hotpotqa",
                    r["id"],
                    r["level"],
                    r["id"],
                    [("Document", text)],
                    r["question"],
                    labels,
                    r["answer"],
                    spans,
                )
            )
    return rows


def contractnli(rng: random.Random) -> list[dict[str, Any]]:
    """ContractNLI: NDA + hypothesis -> entailed / contradicted / not mentioned.

    "Not mentioned" has no spans. Subset: hypothesis.
    """
    del rng
    labels = {
        "entailed": "The contract entails the statement.",
        "contradicted": "The contract contradicts the statement.",
        "not_mentioned": "The contract does not mention it.",
    }
    names = {
        "Entailment": "entailed",
        "Contradiction": "contradicted",
        "NotMentioned": "not_mentioned",
    }
    rows = []
    for split in ("train", "dev", "test"):
        with open(f"{_EXTERNAL}/contractnli/contract-nli/{split}.json") as f:
            data = json.load(f)
        for doc in data["documents"]:
            text = doc["text"]
            for key, a in doc["annotation_sets"][0]["annotations"].items():
                label = names[a["choice"]]
                spans = (
                    []
                    if label == "not_mentioned"
                    else [("Contract", *doc["spans"][i]) for i in a["spans"]]
                )
                rows.append(
                    _row(
                        "contractnli",
                        (doc["id"], key),
                        key,
                        doc["id"],
                        [("Contract", text)],
                        f'Statement: "{data["labels"][key]["hypothesis"]}"\n'
                        "Does the contract entail the statement, contradict"
                        " it, or not mention it?",
                        labels,
                        label,
                        spans,
                    )
                )
    return rows


def hatexplain(rng: random.Random) -> list[dict[str, Any]]:
    """HateXplain: post -> hate speech / offensive / normal.

    Majority label of the three annotators (posts without one are skipped);
    spans are the tokens that at least two annotators marked as the reason.
    "Normal" has no spans. Subset: label.
    """
    del rng
    labels = {
        "hatespeech": "Hate speech: it attacks or demeans a group of people.",
        "offensive": "Offensive, but not hate speech.",
        "normal": "Neither hateful nor offensive.",
    }
    with open(f"{_EXTERNAL}/hatexplain/dataset.json") as f:
        data = json.load(f)
    rows = []
    for pid, p in data.items():
        votes = collections.Counter(a["label"] for a in p["annotators"])
        label, count = votes.most_common(1)[0]
        if count < 2:
            continue
        tokens, text, offsets = p["post_tokens"], "", []
        for tok in tokens:
            if text:
                text += " "
            offsets.append((len(text), len(text) + len(tok)))
            text += tok
        spans = []
        if label != "normal" and p["rationales"]:
            marked = [
                sum(r[i] for r in p["rationales"]) >= 2
                for i in range(len(tokens))
            ]
            i = 0
            while i < len(tokens):
                if marked[i]:
                    j = i
                    while j + 1 < len(tokens) and marked[j + 1]:
                        j += 1
                    spans.append(("Post", offsets[i][0], offsets[j][1]))
                    i = j + 1
                else:
                    i += 1
        rows.append(
            _row(
                "hatexplain",
                pid,
                label,
                pid,
                [("Post", text)],
                "Is this post hate speech, offensive, or normal?",
                labels,
                label,
                spans,
            )
        )
    return rows


LOADERS = {
    "ragtruth": ragtruth,
    "mqm": mqm,
    "qa_feedback": qa_feedback,
    "mushroom": mushroom,
    "processbench": processbench,
    "prm800k": prm800k,
    "fever": fever,
    "hotpotqa": hotpotqa,
    "contractnli": contractnli,
    "hatexplain": hatexplain,
}


def main() -> None:
    """Prints counts per dataset, subset and answer."""
    for name, load in LOADERS.items():
        rows = load(random.Random(0))
        subsets = collections.Counter(r["subset"] for r in rows)
        answers = collections.Counter(r["label"] for r in rows)
        spans = sum(r["needs_spans"] for r in rows)
        print(
            f"{name:13} rows {len(rows):7} with spans {spans:7} groups "
            f"{len({r['group'] for r in rows}):6} | subsets {len(subsets)} | "
            f"answers {dict(answers.most_common(6))}"
        )


if __name__ == "__main__":
    main()
