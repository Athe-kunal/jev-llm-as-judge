"""Reason+Long-12k: the training mixture of Li & Wagle (arXiv 2610.02076v3).

Usage:
  # fetch the pinned sources into data/external/<dataset>/ (once)
  PYTHONPATH=src .venv/bin/python -m spanjudge.reason_long --download
  # build data/splits/reason_long_12k/{train,eval}.jsonl and manifest.json
  PYTHONPATH=src .venv/bin/python -m spanjudge.reason_long

Twelve public human-annotated datasets, 1,000 training rows each (Table 3 of
the paper): CLINC150, MASSIVE (en-US), Bitext customer support, ReClor,
LogiQA 2.0, CosmosQA ("none of the above" gold upsampled to 16%), PIQA,
alpha-NLI, ProofWriter, ContractNLI, MuSiQue (answerability over 20
paragraphs) and QuALITY. Rows use human.py's format (human._row); labels are
the datasets' own gold answers. Native evidence spans are kept where the
release has them: ContractNLI (human spans, human.contractnli), MuSiQue
(supporting paragraphs of answerable questions) and ProofWriter (the facts
and rules of the gold proof).

Cleaning, applied to each training pool before sampling (paper section 3.5,
Appendix D): rows sharing a 13-word n-gram, or an identical sentence of six
or more words, with any evaluation benchmark are dropped (decontaminate());
rows whose input also appears in the dataset's official validation split
are dropped; exact duplicate inputs within the pool are collapsed. Training
rows come from each dataset's official train split, eval rows (up to
EVAL_ROWS per dataset, whole groups) from its official validation split;
Bitext has no split, so a seeded group hold-out stands in for one. Rows whose
prompt (train_joint.build_prompt, question_first) exceeds MAX_TOKENS are
left out, never truncated. See docs/data.md.
"""

import argparse
import collections
import contextlib
import csv
import gzip
import hashlib
import io
import json
import pathlib
import random
import re
import statistics
import string
import tarfile
import unicodedata
import urllib.request
import zipfile
from typing import Any, Callable, Iterable

from huggingface_hub import hf_hub_download

from spanjudge import human
from spanjudge.human import _EXTERNAL, _row

SEED = 20260922  # the paper's training seed (Table 2)
PER_DATASET = 1000
EVAL_ROWS = 500  # default; change with --eval-rows
MAX_TOKENS = 10240  # Table 2: 10,240 for Reason+Long-12k, never truncated
COSMOS_NONE_SHARE = 0.16  # Appendix G: "we upsample these to 16%"
NGRAM = 13
MIN_SENTENCE_WORDS = 6
OUT = "data/splits/reason_long_12k"

# ---------------------------------------------------------------- sources
# (local dataset dir, kind, where, revision, files). kind "hf": Hugging Face
# dataset repo (files fetched with hf_hub_download at that revision); "url":
# plain download (GitHub raw URLs pinned by commit; the sha256 of every
# fetched file is written to <dir>/SOURCES.json).
TRAIN_SOURCES: dict[str, dict[str, Any]] = {
    "clinc150": {
        "kind": "hf", "repo": "clinc/clinc_oos",
        "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "files": ["plus/train-00000-of-00001.parquet",
                  "plus/validation-00000-of-00001.parquet", "README.md"],
    },
    "massive": {
        "kind": "url", "revision": "MASSIVE 1.1 release tarball",
        "files": {"amazon-massive-dataset-1.1.tar.gz":
                  "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
                  "amazon-massive-dataset-1.1.tar.gz"},
    },
    "bitext": {
        "kind": "hf",
        "repo": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "revision": "430d1a89bd93bd1fa23c16f29dd53e73f0087443",
        "files": ["Bitext_Sample_Customer_Support_Training_Dataset_27K_"
                  "responses-v11.csv", "README.md"],
    },
    "reclor": {  # the official zip is behind a form; a mirror of it
        "kind": "hf", "repo": "metaeval/reclor",
        "revision": "0414b39bfc39b2d555b48fa245fc198744c46215",
        "files": ["train.json", "val.json", "README.md"],
    },
    "logiqa2": {
        "kind": "url", "revision": "csitfun/LogiQA2.0@955e1d3d",
        "files": {
            f"{s}.txt": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/"
            f"955e1d3df6c59d9bfb44d9913da1e1a27ec14e18/logiqa/DATA/LOGIQA/{s}.txt"
            for s in ("train", "dev")
        },
    },
    "cosmosqa": {
        "kind": "url", "revision": "wilburOne/cosmosqa@b6eb99cc",
        "files": {
            f"{s}.csv": "https://raw.githubusercontent.com/wilburOne/cosmosqa/"
            f"b6eb99cca4e2a51dd28a9a6f562534872d851639/data/{s}.csv"
            for s in ("train", "valid")
        },
    },
    "piqa": {
        "kind": "url", "revision": "yonatanbisk.com/piqa/data (unversioned)",
        "files": {
            f: f"https://yonatanbisk.com/piqa/data/{f}"
            for f in ("train.jsonl", "train-labels.lst", "valid.jsonl",
                      "valid-labels.lst")
        },
    },
    "anli": {
        "kind": "url", "revision": "AI2 alphanli-train-dev.zip (unversioned)",
        "files": {"alphanli-train-dev.zip":
                  "https://storage.googleapis.com/ai2-mosaic/public/alphanli/"
                  "alphanli-train-dev.zip"},
    },
    "proofwriter": {
        "kind": "url", "revision": "proofwriter-dataset-V2020.12.3",
        "files": {"proofwriter-dataset-V2020.12.3.zip":
                  "https://aristo-data-public.s3.amazonaws.com/proofwriter/"
                  "proofwriter-dataset-V2020.12.3.zip"},
    },
    "musique": {  # the official zip is on Google Drive; a mirror of it
        "kind": "hf", "repo": "bdsaglam/musique",
        "revision": "22873a405dd809893b22ada0b499299fb612d2df",
        "files": ["musique_full_v1.0_train.jsonl",
                  "musique_full_v1.0_dev.jsonl", "README.md"],
    },
    "quality": {
        "kind": "url", "revision": "nyu-mll/quality@f84977c4 v1.0.1",
        "files": {"QuALITY.v1.0.1.zip":
                  "https://github.com/nyu-mll/quality/raw/"
                  "f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/"
                  "QuALITY.v1.0.1.zip"},
    },
    # contractnli: already in data/external/contractnli (human.contractnli)
}

BENCH_SOURCES: dict[str, dict[str, Any]] = {
    "boolq": {"kind": "hf", "repo": "google/boolq",
              "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              "files": ["data/validation-00000-of-00001.parquet"]},
    "mmlu": {"kind": "hf", "repo": "cais/mmlu",
             "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
             "files": ["all/test-00000-of-00001.parquet",
                       "all/validation-00000-of-00001.parquet",
                       "all/dev-00000-of-00001.parquet"]},
    "mmlu_pro": {"kind": "hf", "repo": "TIGER-Lab/MMLU-Pro",
                 "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                 "files": ["data/test-00000-of-00001.parquet",
                           "data/validation-00000-of-00001.parquet"]},
    "arc_challenge": {"kind": "hf", "repo": "allenai/ai2_arc",
                      "revision": "210d026faf9955653af8916fad021475a3f00453",
                      "files": ["ARC-Challenge/test-00000-of-00001.parquet"]},
    "winogrande": {"kind": "hf", "repo": "allenai/winogrande",
                   "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                   "files": ["winogrande_xl/validation-00000-of-00001.parquet"]},
    "sciq": {"kind": "hf", "repo": "allenai/sciq",
             "revision": "2c94ad3e1aafab77146f384e23536f97a4849815",
             "files": ["data/test-00000-of-00001.parquet"]},
    "banking77": {
        "kind": "url", "revision": "PolyAI-LDN/task-specific-datasets@57ec275d",
        "files": {"test.csv":
                  "https://raw.githubusercontent.com/PolyAI-LDN/"
                  "task-specific-datasets/"
                  "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/"
                  "test.csv"}},
    "jevbench": {
        "kind": "url", "revision": "fstandhartinger/jevbench v1.4.2.2 "
        "(commit fcc69279)",
        "files": {
            f"{s}.jsonl": "https://raw.githubusercontent.com/fstandhartinger/"
            "jevbench/fcc69279a933f882ccc4489eb14bf467647dbdb7/datasets/"
            f"public/{s}.jsonl"
            for s in ("easy", "original", "hard")
        }},
    "gsm8k": {"kind": "hf", "repo": "openai/gsm8k",
              "revision": "740312add88f781978c0658806c59bc2815b9866",
              "files": ["main/test-00000-of-00001.parquet"]},
    "ifeval": {"kind": "hf", "repo": "google/IFEval",
               "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
               "files": ["ifeval_input_data.jsonl"]},
    "triviaqa": {"kind": "hf", "repo": "mandarjoshi/trivia_qa",
                 "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                 "files": ["rc.nocontext/validation-00000-of-00001.parquet"]},
    "lambada": {"kind": "hf", "repo": "EleutherAI/lambada_openai",
                "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
                "files": ["default/test/default.parquet"]},
    "wikitext2": {"kind": "hf", "repo": "Salesforce/wikitext",
                  "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                  "files": ["wikitext-2-raw-v1/test-00000-of-00001.parquet"]},
    "mmbench": {"kind": "hf", "repo": "lmms-lab/MMBench",
                "revision": "56ba1af8954932c4804bd3f522e05ed96e63b654",
                "files": ["en/dev-00000-of-00001.parquet"]},
    "mmstar": {"kind": "hf", "repo": "Lin-Chen/MMStar",
               "revision": "bc98d668301da7b14f648724866e57302778ab27",
               "files": ["mmstar.parquet"]},
    "dolly": {"kind": "hf", "repo": "databricks/databricks-dolly-15k",
              "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
              "files": ["databricks-dolly-15k.jsonl"]},
    "mt_bench": {"kind": "hf", "repo": "HuggingFaceH4/mt_bench_prompts",
                 "revision": "e3a795c5e9a82ee40611c416b8a7786c73198991",
                 "files": ["raw/question.jsonl"]},
}
SUITES = {
    "external": ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande",
                 "sciq", "banking77"),
    "jevbench": ("jevbench",),
    "general": ("gsm8k", "ifeval", "triviaqa", "lambada", "wikitext2"),
    "image": ("mmbench", "mmstar"),
    "behavior": ("dolly", "mt_bench"),
}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(names: Iterable[str] | None = None) -> None:
    """Fetches the pinned training and benchmark files (skips existing)."""
    sources = {**TRAIN_SOURCES,
               **{f"benchmarks/{k}": v for k, v in BENCH_SOURCES.items()}}
    for name, src in sources.items():
        if names and name.split("/")[-1] not in names:
            continue
        root = pathlib.Path(_EXTERNAL) / name
        root.mkdir(parents=True, exist_ok=True)
        record = {"revision": src["revision"], "files": {}}
        if src["kind"] == "hf":
            record["repo"] = src["repo"]
            for f in src["files"]:
                path = pathlib.Path(hf_hub_download(
                    src["repo"], f, repo_type="dataset",
                    revision=src["revision"], local_dir=root))
                record["files"][f] = _sha256(path)
        else:
            for f, url in src["files"].items():
                path = root / f
                if not path.exists():
                    tmp = path.with_suffix(path.suffix + ".part")
                    with urllib.request.urlopen(url, timeout=600) as r, \
                            open(tmp, "wb") as out:
                        while block := r.read(1 << 20):
                            out.write(block)
                    tmp.rename(path)
                record["files"][f] = {"url": url, "sha256": _sha256(path)}
        (root / "SOURCES.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"{name}: {len(record['files'])} files")


# ---------------------------------------------------------------- loaders
# Each loader returns {"train": rows, "validation": rows}: the official
# train split and the official validation / dev split (labelled test splits
# are not used). Option ids are the labels' own names (intent, true / false /
# unknown, yes / no) or the 1-based position of the option in the release
# ("1".."k"); train_joint shuffles the options and assigns its own codes.
# subset = gold label (used by the eval draw to round-robin over labels).
# Every row carries "_units": the source texts the overlap filter checks
# (the release's own fields: utterance, passage, question, options, theory,
# statement, hypothesis, each MuSiQue paragraph on its own). Our fixed
# question wordings and label descriptions are not source data and are not
# checked (one of them, the intent question, is a sentence of a JevBench
# item). Removed before writing.

Split = dict[str, list[dict[str, Any]]]


def _root(name: str) -> pathlib.Path:
    return pathlib.Path(_EXTERNAL) / name


def _intent_rows(source: str, split: str, items: list[tuple[str, str]],
                 names: list[str]) -> list[dict[str, Any]]:
    """Utterance -> one of the dataset's intent names (verbatim)."""
    labels = {n: n for n in names}
    return [
        {**_row(source, (split, i, text), intent, text, [("Message", text)],
                "Which intent does the user's message express?", labels,
                intent, []), "_units": [text]}
        for i, (text, intent) in enumerate(items)
    ]


def _clinc_names() -> list[str]:
    """Intent names of the clinc_oos "plus" config, by class index."""
    lines = (_root("clinc150") / "README.md").read_text().splitlines()
    start = lines.index("- config_name: plus")
    names: dict[int, str] = {}
    for line in lines[start + 1:]:
        m = re.fullmatch(r"\s+'(\d+)': (.+)", line)
        if m:
            name = m[2].strip()
            if len(name) > 1 and name[0] == name[-1] == "'":  # YAML-quoted
                name = name[1:-1].replace("''", "'")
            names[int(m[1])] = name
        elif names:
            break
    return [names[i] for i in range(len(names))]


def clinc150() -> Split:
    """CLINC150 "plus" (150 intents + oos): train / validation."""
    import pandas as pd

    names = _clinc_names()
    out = {}
    for split in ("train", "validation"):
        d = pd.read_parquet(
            _root("clinc150") / f"plus/{split}-00000-of-00001.parquet")
        out[split] = _intent_rows(
            "clinc150", split,
            [(t, names[i]) for t, i in zip(d.text, d.intent, strict=True)],
            names)
    return out


def massive() -> Split:
    """MASSIVE 1.1 en-US (60 intents): train / dev partitions."""
    with tarfile.open(_root("massive") / "amazon-massive-dataset-1.1.tar.gz") \
            as t:
        data = [json.loads(line)
                for line in t.extractfile("1.1/data/en-US.jsonl")]
    names = sorted({x["intent"] for x in data})
    return {
        split: _intent_rows(
            "massive", split,
            [(x["utt"], x["intent"]) for x in data if x["partition"] == part],
            names)
        for split, part in (("train", "train"), ("validation", "dev"))
    }


BITEXT_HELDOUT = 0.1  # share of distinct utterances held out (no split)


def bitext() -> Split:
    """Bitext customer support (27 intents); one file, so a seeded split.

    BITEXT_HELDOUT of the distinct utterances (after normalization, so
    copies stay together) form the "validation" side.
    """
    path = (_root("bitext") /
            "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses"
            "-v11.csv")
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    names = sorted({x["intent"] for x in data})
    rows = _intent_rows("bitext", "all",
                        [(x["instruction"], x["intent"]) for x in data], names)
    keys = sorted({input_key(r) for r in rows})
    random.Random(f"{SEED}:bitext-split").shuffle(keys)
    held = set(keys[: int(BITEXT_HELDOUT * len(keys))])
    return {"train": [r for r in rows if input_key(r) not in held],
            "validation": [r for r in rows if input_key(r) in held]}


def _mc_row(source: str, key: Any, group: Any, parts: list[tuple[str, str]],
            question: str, options: list[str], gold: int,
            fixed_question: bool = False) -> dict[str, Any]:
    """A multiple-choice row; gold is the 0-based index of the answer.

    fixed_question: the question is our wording (PIQA, alpha-NLI), so the
    overlap filter skips it.
    """
    labels = {str(i + 1): o for i, o in enumerate(options)}
    row = _row(source, key, str(gold + 1), group, parts, question, labels,
               str(gold + 1), [])
    row["_units"] = [t for _, t in parts] + (
        [] if fixed_question else [question]) + list(options)
    return row


def reclor() -> Split:
    """ReClor: passage + question -> one of 4 options (train / val)."""
    out = {}
    for split, f in (("train", "train.json"), ("validation", "val.json")):
        data = json.loads((_root("reclor") / f).read_text())
        out[split] = [
            _mc_row("reclor", x["id_string"], x["context"],
                    [("Passage", x["context"])], x["question"], x["answers"],
                    x["label"])
            for x in data
        ]
    return out


def logiqa2() -> Split:
    """LogiQA 2.0 (English MRC): passage + question -> 4 options."""
    out = {}
    for split, f in (("train", "train.txt"), ("validation", "dev.txt")):
        with open(_root("logiqa2") / f) as fh:
            data = [json.loads(line) for line in fh if line.strip()]
        out[split] = [
            _mc_row("logiqa2", (split, x["id"]), x["text"],
                    [("Passage", x["text"])], x["question"], x["options"],
                    x["answer"])
            for x in data
        ]
    return out


def is_none_answer(text: str) -> bool:
    """CosmosQA's catch-all option ("None of the above choices .")."""
    return " ".join(_words(text)).startswith("none of the above")


def cosmosqa() -> Split:
    """CosmosQA: context + question -> 4 options (train / valid)."""
    out = {}
    for split, f in (("train", "train.csv"), ("validation", "valid.csv")):
        with open(_root("cosmosqa") / f, newline="") as fh:
            data = list(csv.DictReader(fh))
        out[split] = [
            _mc_row("cosmosqa", x["id"], x["context"],
                    [("Context", x["context"])], x["question"],
                    [x[f"answer{i}"] for i in range(4)], int(x["label"]))
            for x in data
        ]
    return out


def piqa() -> Split:
    """PIQA: goal -> the better of two solutions (train / valid)."""
    out = {}
    for split, f in (("train", "train"), ("validation", "valid")):
        with open(_root("piqa") / f"{f}.jsonl") as fh:
            data = [json.loads(line) for line in fh if line.strip()]
        gold = (_root("piqa") / f"{f}-labels.lst").read_text().split()
        out[split] = [
            _mc_row("piqa", (split, i), x["goal"], [("Goal", x["goal"])],
                    "Which solution better achieves the goal?",
                    [x["sol1"], x["sol2"]], int(g), fixed_question=True)
            for i, (x, g) in enumerate(zip(data, gold, strict=True))
        ]
    return out


def anli() -> Split:
    """alpha-NLI (ART): two observations -> the more plausible hypothesis."""
    out = {}
    with zipfile.ZipFile(_root("anli") / "alphanli-train-dev.zip") as z:
        for split, f in (("train", "train"), ("validation", "dev")):
            data = [json.loads(line)
                    for line in z.read(f"{f}.jsonl").decode().splitlines()
                    if line.strip()]
            gold = z.read(f"{f}-labels.lst").decode().split()
            out[split] = [
                _mc_row("anli", (split, i, x["story_id"]), x["story_id"],
                        [("Observation 1", x["obs1"]),
                         ("Observation 2", x["obs2"])],
                        "Which hypothesis best explains what happened between"
                        " the two observations?",
                        [x["hyp1"], x["hyp2"]], int(g) - 1,
                        fixed_question=True)
                for i, (x, g) in enumerate(zip(data, gold, strict=True))
            ]
    return out


PROOFWRITER_CONFIG = "OWA/depth-5"
PROOFWRITER_LABELS = {
    "true": "True: the theory implies the statement.",
    "false": "False: the theory implies that the statement is false.",
    "unknown": "Unknown: the theory implies neither the statement nor its"
    " negation.",
}


def _sentence_offsets(theory: str, texts: list[tuple[str, str]]) -> dict:
    """(start, end) in the theory of each fact / rule, in theory order."""
    where, pos = {}, 0
    for key, text in texts:
        i = theory.find(text, pos)
        if i < 0:
            i = theory.find(text)
        if i >= 0:
            where[key] = (i, i + len(text))
            pos = i + len(text)
    return where


def proofwriter() -> Split:
    """ProofWriter (OWA, depth <= 5): theory + statement -> T / F / Unknown.

    Spans: the facts and rules used by the first proof listed for the
    question (the proof of the statement, or of its negation for False).
    Unknown has no proof, so no spans.
    """
    path = _root("proofwriter") / "proofwriter-dataset-V2020.12.3.zip"
    out = {}
    with zipfile.ZipFile(path) as z:
        for split, f in (("train", "train"), ("validation", "dev")):
            rows = []
            name = (f"proofwriter-dataset-V2020.12.3/{PROOFWRITER_CONFIG}/"
                    f"meta-{f}.jsonl")
            with z.open(name) as fh:
                for line in fh:
                    x = json.loads(line)
                    theory = x["theory"]
                    where = _sentence_offsets(theory, [
                        *((k, v["text"]) for k, v in x["triples"].items()),
                        *((k, v["text"]) for k, v in x["rules"].items())])
                    for qid, q in x["questions"].items():
                        label = str(q["answer"]).lower()
                        spans = []
                        proofs = q.get("proofsWithIntermediates") or []
                        if label != "unknown" and proofs:
                            used = re.findall(r"(?:triple|rule)\d+",
                                              proofs[0]["representation"])
                            spans = [("Theory", *where[k]) for k in used
                                     if k in where]
                        rows.append({**_row(
                            "proofwriter", (x["id"], qid), label, x["id"],
                            [("Theory", theory)],
                            f'Statement: "{q["question"]}"\nBased only on'
                            " the theory, is the statement true, false, or"
                            " unknown?",
                            PROOFWRITER_LABELS, label, spans),
                            "_units": [theory, q["question"]]})
            out[split] = rows
    return out


def contractnli() -> Split:
    """ContractNLI via human.contractnli (spans kept), split by document."""
    rows = human.contractnli(random.Random(0))  # the loader ignores the rng
    out = {}
    for split, f in (("train", "train"), ("validation", "dev")):
        data = json.loads(
            (_root("contractnli") / f"contract-nli/{f}.json").read_text())
        groups = {f"contractnli:{human._hash(d['id'])}"
                  for d in data["documents"]}
        out[split] = [
            {**r, "subset": r["label"],
             "_units": [r["parts"][0]["text"],
                        re.match(r'Statement: "(.*)"\n', r["question"],
                                 re.S)[1]]}
            for r in rows if r["group"] in groups]
    return out


MUSIQUE_LABELS = {
    "yes": "Yes: the paragraphs contain enough information to answer the"
    " question.",
    "no": "No: the paragraphs do not contain enough information to answer"
    " the question.",
}


def musique() -> Split:
    """MuSiQue-Full: question + its paragraphs -> answerable yes / no.

    MuSiQue-Full pairs each answerable question with an unanswerable
    contrast copy (same id, a supporting paragraph removed); both copies
    share the id, which is the group. Spans (answerable rows only): each
    supporting paragraph (title line and text).
    """
    out = {}
    for split, f in (("train", "train"), ("validation", "dev")):
        rows = []
        with open(_root("musique") / f"musique_full_v1.0_{f}.jsonl") as fh:
            for line in fh:
                x = json.loads(line)
                text, spans, units = "", [], []
                for p in x["paragraphs"]:
                    block = f"{p['title']}\n{p['paragraph_text']}"
                    if text:
                        text += "\n\n"
                    if p["is_supporting"] and x["answerable"]:
                        spans.append(("Paragraphs", len(text),
                                      len(text) + len(block)))
                    text += block
                    units.append(block)
                label = "yes" if x["answerable"] else "no"
                question = (f"Question: {x['question']}\nDo the paragraphs"
                            " contain enough information to answer the"
                            " question?")
                r = _row("musique", (x["id"], label), label, x["id"],
                         [("Paragraphs", text)], question, MUSIQUE_LABELS,
                         label, spans)
                r["_units"] = [*units, x["question"]]
                rows.append(r)
        out[split] = rows
    return out


def quality() -> Split:
    """QuALITY v1.0.1 (HTML-stripped): article + question -> 4 options.

    Gold: the release's gold_label (annotators' majority), 1-based.
    """
    out = {}
    with zipfile.ZipFile(_root("quality") / "QuALITY.v1.0.1.zip") as z:
        for split, f in (("train", "train"), ("validation", "dev")):
            rows = []
            for line in z.read(f"QuALITY.v1.0.1.htmlstripped.{f}").decode() \
                    .split("\n"):
                if not line.strip():
                    continue
                x = json.loads(line)
                article = x["article"].strip()
                for q in x["questions"]:
                    if not q.get("gold_label"):
                        continue
                    rows.append(_mc_row(
                        "quality", q["question_unique_id"], x["article_id"],
                        [("Article", article)], q["question"].strip(),
                        q["options"], int(q["gold_label"]) - 1))
            out[split] = rows
    return out


DATASETS: dict[str, Callable[[], Split]] = {
    "clinc150": clinc150,
    "massive": massive,
    "bitext": bitext,
    "reclor": reclor,
    "logiqa2": logiqa2,
    "cosmosqa": cosmosqa,
    "piqa": piqa,
    "anli": anli,
    "proofwriter": proofwriter,
    "contractnli": contractnli,
    "musique": musique,
    "quality": quality,
}
SOURCE_SPLITS = {name: {"train": "train", "validation": "validation"}
                 for name in DATASETS}
SOURCE_SPLITS.update({
    "massive": {"train": "train", "validation": "dev"},
    "bitext": {"train": "all (90% of distinct utterances)",
               "validation": "all (seeded 10% of distinct utterances)"},
    "reclor": {"train": "train", "validation": "val"},
    "logiqa2": {"train": "train", "validation": "dev"},
    "cosmosqa": {"train": "train", "validation": "valid"},
    "piqa": {"train": "train", "validation": "valid"},
    "anli": {"train": "train", "validation": "dev"},
    "proofwriter": {"train": f"{PROOFWRITER_CONFIG} train",
                    "validation": f"{PROOFWRITER_CONFIG} dev"},
    "contractnli": {"train": "train", "validation": "dev"},
    "musique": {"train": "full_v1.0 train", "validation": "full_v1.0 dev"},
    "quality": {"train": "v1.0.1 htmlstripped train",
                "validation": "v1.0.1 htmlstripped dev"},
})


# ---------------------------------------------------------- normalization
# Words: NFKC, case-folded, curly quotes made straight, then maximal runs of
# Unicode letters / digits / underscore (punctuation and whitespace are
# separators; "don't" -> "don", "t" on both sides alike). Sentences: the
# text split after . ! ? followed by whitespace, and at line breaks.

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()
    return _WORD.findall(text)


def _sentences(text: str) -> Iterable[str]:
    """Normalized sentences of at least MIN_SENTENCE_WORDS words."""
    for s in _SENTENCE_END.split(text):
        w = _words(s)
        if len(w) >= MIN_SENTENCE_WORDS:
            yield " ".join(w)


def _ngrams(text: str) -> Iterable[str]:
    w = _words(text)
    for i in range(len(w) - NGRAM + 1):
        yield " ".join(w[i:i + NGRAM])


def input_key(row: dict[str, Any]) -> str:
    """The row's input (parts, question, options) for exact-duplicate checks.

    Normalized words, so copies that differ only in case, spacing or
    punctuation count as the same input; options are order-free.
    """
    texts = [p["text"] for p in row["parts"]] + [row["question"]]
    texts += sorted(" ".join(_words(v)) for v in row["labels"].values())
    return hashlib.sha256(
        "\x1f".join(" ".join(_words(t)) for t in texts).encode()).hexdigest()


# ------------------------------------------------------------- benchmarks
def _parquet(path: pathlib.Path) -> list[dict[str, Any]]:
    import pandas as pd

    d = pd.read_parquet(path)
    return d.drop(columns=[c for c in ("image",) if c in d]).to_dict("records")


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _strings(x: Any) -> Iterable[str]:
    """Every string inside a (nested) value."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        for v in (x.tolist() if hasattr(x, "tolist") else x):
            yield from _strings(v)


def benchmark_items() -> dict[str, list[list[str]]]:
    """The text fields of every evaluation item, per benchmark.

    Whole evaluation splits (the paper's fixed samples are not released, so
    the check covers every item they could have been drawn from).
    """
    b = pathlib.Path(_EXTERNAL) / "benchmarks"
    out: dict[str, list[list[str]]] = {}

    def fields(name: str, records: Iterable[dict], keys: tuple[str, ...]):
        out.setdefault(name, []).extend(
            [s for k in keys if k in r for s in _strings(r[k])]
            for r in records)

    fields("boolq", _parquet(b / "boolq/data/validation-00000-of-00001.parquet"),
           ("question", "passage"))
    for s in ("test", "validation", "dev"):
        fields("mmlu", _parquet(b / f"mmlu/all/{s}-00000-of-00001.parquet"),
               ("question", "choices"))
    for s in ("test", "validation"):
        fields("mmlu_pro",
               _parquet(b / f"mmlu_pro/data/{s}-00000-of-00001.parquet"),
               ("question", "options", "cot_content"))
    fields("arc_challenge", _parquet(
        b / "arc_challenge/ARC-Challenge/test-00000-of-00001.parquet"),
        ("question", "choices"))
    fields("winogrande", _parquet(
        b / "winogrande/winogrande_xl/validation-00000-of-00001.parquet"),
        ("sentence", "option1", "option2"))
    fields("sciq", _parquet(b / "sciq/data/test-00000-of-00001.parquet"),
           ("question", "support", "correct_answer", "distractor1",
            "distractor2", "distractor3"))
    with open(b / "banking77/test.csv", newline="") as f:
        fields("banking77", csv.DictReader(f), ("text",))
    for s in ("easy", "original", "hard"):
        fields("jevbench", _jsonl(b / f"jevbench/{s}.jsonl"),
               ("state", "question", "labels"))
    fields("gsm8k", _parquet(b / "gsm8k/main/test-00000-of-00001.parquet"),
           ("question", "answer"))
    fields("ifeval", _jsonl(b / "ifeval/ifeval_input_data.jsonl"),
           ("prompt",))
    tq = _parquet(b / "triviaqa/rc.nocontext/validation-00000-of-00001.parquet")
    fields("triviaqa", ({"q": r["question"], "a": r["answer"]["value"]}
                        for r in tq), ("q", "a"))
    fields("lambada", _parquet(b / "lambada/default/test/default.parquet"),
           ("text",))
    # WikiText-2 test: one item per article (a line " = Title = " opens one)
    articles, cur = [], []
    for r in _parquet(
            b / "wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet"):
        t = r["text"]
        if re.fullmatch(r" = [^=].* = \n", t) and cur:
            articles.append({"text": cur})
            cur = []
        if t.strip():
            cur.append(t)
    articles.append({"text": cur})
    fields("wikitext2", articles, ("text",))
    fields("mmbench", _parquet(b / "mmbench/en/dev-00000-of-00001.parquet"),
           ("question", "hint", "A", "B", "C", "D"))
    fields("mmstar", _parquet(b / "mmstar/mmstar.parquet"), ("question",))
    # Behavior suite: the items are prompts (App. B: "Two Dolly prompts share
    # a 13-word window"); reference responses are never model inputs or
    # scoring targets there, so they are not checked.
    fields("dolly", _jsonl(b / "dolly/databricks-dolly-15k.jsonl"),
           ("instruction", "context"))
    fields("mt_bench", _jsonl(b / "mt_bench/raw/question.jsonl"),
           ("prompt",))
    return out


class OverlapIndex:
    """13-word n-grams and >= 6-word sentences of the evaluation benchmarks.

    Each key maps to a bit mask over BENCHMARKS (which benchmarks have it).
    """

    def __init__(self, items: dict[str, list[list[str]]]):
        self.names = sorted(items)
        self.ngrams: dict[int, int] = {}
        self.sentences: dict[int, int] = {}
        self.sizes = {}
        for b, name in enumerate(self.names):
            bit = 1 << b
            self.sizes[name] = len(items[name])
            for texts in items[name]:
                for t in texts:
                    for g in _ngrams(t):
                        h = hash(g)
                        self.ngrams[h] = self.ngrams.get(h, 0) | bit
                    for s in _sentences(t):
                        h = hash(s)
                        self.sentences[h] = self.sentences.get(h, 0) | bit
        self._cache: dict[str, tuple[int, int]] = {}

    def text_hits(self, text: str) -> tuple[int, int]:
        """(n-gram mask, sentence mask) of one text (cached)."""
        hit = self._cache.get(text)
        if hit is None:
            ng = st = 0
            for g in _ngrams(text):
                ng |= self.ngrams.get(hash(g), 0)
            for s in _sentences(text):
                st |= self.sentences.get(hash(s), 0)
            hit = self._cache[text] = (ng, st)
        return hit

    def row_hits(self, row: dict[str, Any]) -> tuple[int, int]:
        """Masks for a row: its parts (or _units), question and options."""
        units = row.get("_units") or [
            *(p["text"] for p in row["parts"]), row["question"],
            *row["labels"].values()]
        ng = st = 0
        for t in units:
            a, b = self.text_hits(t)
            ng, st = ng | a, st | b
        return ng, st

    def decode(self, mask: int) -> list[str]:
        return [n for b, n in enumerate(self.names) if mask >> b & 1]


def decontaminate(rows: list[dict], index: OverlapIndex,
                  stats: dict[str, Any]) -> list[dict]:
    """Drops rows sharing a 13-gram or a >= 6-word sentence with a benchmark.

    stats gets the drop counts: by kind (a row with both counts under
    "ngram") and by benchmark (a row can count for several).
    """
    keep, by_bench = [], collections.Counter()
    by_kind = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        ng, st = index.row_hits(r)
        if not (ng or st):
            keep.append(r)
            continue
        by_kind["ngram" if ng else "sentence"] += 1
        for name in index.decode(ng | st):
            by_bench[name] += 1
            if len(examples[name]) < 3:
                examples[name].append(r["id"])
    stats.update({"contaminated": sum(by_kind.values()),
                  "contaminated_by_kind": dict(by_kind),
                  "contaminated_by_benchmark": dict(sorted(by_bench.items())),
                  "contaminated_examples": dict(examples)})
    return keep


# ---------------------------------------------------------------- lengths
@contextlib.contextmanager
def _wide_codes():
    """Lets train_joint.build_prompt render > 26 options (lengths only).

    Codes past Z become AA, AB, ... (one or two letters, so token counts are
    within a few tokens per option of any one-token code scheme).
    """
    from spanjudge import train_joint

    original = train_joint.choice_letters
    letters = list(string.ascii_uppercase)
    codes = letters + [a + b for a in letters for b in letters]
    train_joint.choice_letters = lambda row, order: codes[: len(order)]
    try:
        yield
    finally:
        train_joint.choice_letters = original


def prompt_lengths(rows: list[dict], tok: Any, order: str) -> list[int]:
    """Prompt tokens of each row as train_joint renders it (given ORDER)."""
    from spanjudge import train_joint

    saved = train_joint.ORDER
    train_joint.ORDER = order
    out = []
    try:
        with _wide_codes():
            texts = [train_joint.build_prompt(r, list(r["labels"]))[0]
                     for r in rows]
        for i in range(0, len(texts), 256):
            enc = tok(texts[i:i + 256], add_special_tokens=False)
            out += [len(x) for x in enc["input_ids"]]
    finally:
        train_joint.ORDER = saved
    return out


def _length_stats(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {}
    s = sorted(lengths)
    return {"median": int(statistics.median(s)),
            "p95": s[min(len(s) - 1, int(0.95 * len(s)))], "max": s[-1],
            "min": s[0]}


# ---------------------------------------------------------------- builder
def _dedupe(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Collapses exact duplicate inputs: one copy kept when the copies agree
    on the label, none when they disagree. Returns (rows, dropped copies,
    rows dropped for conflicting labels)."""
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[input_key(r)].append(r)
    out, copies, conflict = [], 0, 0
    for r in rows:
        group = by[input_key(r)]
        if group[0] is not r:
            continue
        if len({g["label"] for g in group}) > 1:
            conflict += len(group)
            continue
        copies += len(group) - 1
        out.append(r)
    return out, copies, conflict


def take_eval(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """Up to cap rows in whole groups, round-robin over subsets (labels).

    splits.take_groups' order, but a group is only taken while it fits.
    """
    from spanjudge.splits import _by_subset_groups

    by = _by_subset_groups(rows, rng)
    taken, count, order = [], 0, sorted(by)
    while any(by[s] for s in order):
        for s in order:
            while by[s] and count + len(by[s][-1]) > cap:
                by[s].pop()  # too big to fit: skip
            if by[s]:
                g = by[s].pop()
                taken += g
                count += len(g)
    return taken


def sample_train(name: str, rows: list[dict], n: int,
                 rng: random.Random) -> list[dict]:
    """n rows drawn uniformly, except CosmosQA and ProofWriter.

    CosmosQA: COSMOS_NONE_SHARE of the rows have a "none of the above" gold
    (drawn without replacement from those rows), the rest from the others.
    ProofWriter: label-balanced (true / false / unknown in equal shares).
    """
    if len(rows) < n:
        raise ValueError(f"{name}: only {len(rows)} clean rows for {n}")
    if name == "cosmosqa":
        none = [r for r in rows if is_none_answer(r["labels"][r["label"]])]
        rest = [r for r in rows if not is_none_answer(r["labels"][r["label"]])]
        k = round(COSMOS_NONE_SHARE * n)
        out = rng.sample(none, k) + rng.sample(rest, n - k)
    elif name == "proofwriter":
        by: dict[str, list] = collections.defaultdict(list)
        for r in rows:
            by[r["label"]].append(r)
        labels = sorted(by)
        out = []
        for i, lab in enumerate(labels):
            k = n // len(labels) + (i < n % len(labels))
            out += rng.sample(by[lab], k)
    else:
        out = rng.sample(rows, n)
    rng.shuffle(out)
    return out


def _clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def build(out: pathlib.Path, eval_rows: int = EVAL_ROWS,
          max_tokens: int = MAX_TOKENS,
          names: Iterable[str] | None = None) -> dict[str, Any]:
    """Writes train.jsonl, eval.jsonl and manifest.json under out."""
    from transformers import AutoTokenizer

    from spanjudge import train_joint

    tok = AutoTokenizer.from_pretrained(train_joint.MODEL_ID,
                                        revision=train_joint.REVISION)
    print("indexing benchmarks ...", flush=True)
    items = benchmark_items()
    index = OverlapIndex(items)
    manifest: dict[str, Any] = {
        "paper": "Li & Wagle, LLM-as-Jev, arXiv 2610.02076v3, Reason+Long-12k",
        "seed": SEED, "per_dataset": PER_DATASET, "eval_rows": eval_rows,
        "max_tokens": max_tokens, "order": "question_first",
        "tokenizer": f"{train_joint.MODEL_ID}@{train_joint.REVISION}",
        "normalization": {"ngram": NGRAM, "min_sentence_words":
                          MIN_SENTENCE_WORDS},
        "benchmarks": {
            n: {"items": index.sizes[n],
                "revision": BENCH_SOURCES[n]["revision"],
                "source": BENCH_SOURCES[n].get("repo")
                or list(BENCH_SOURCES[n]["files"].values())}
            for n in index.names},
        "benchmark_suites": {k: list(v) for k, v in SUITES.items()},
        "datasets": {},
    }
    train, evals = [], []
    for name in names or DATASETS:
        print(f"{name} ...", flush=True)
        rng = random.Random(f"{SEED}:{name}")
        data = DATASETS[name]()
        st: dict[str, Any] = {
            "source_split": SOURCE_SPLITS[name],
            "revision": TRAIN_SOURCES.get(name, {}).get(
                "revision", "local data/external/contractnli (human.py)"),
            "source": TRAIN_SOURCES.get(name, {}).get("repo") or list(
                TRAIN_SOURCES.get(name, {}).get("files", {}).values()),
            "train_split_rows": len(data["train"]),
            "validation_split_rows": len(data["validation"]),
        }
        # eval: deduplicated validation rows that fit, whole groups
        val, _, _ = _dedupe(data["validation"])
        val_len = prompt_lengths(val, tok, "question_first")
        val_fit = [r for r, n in zip(val, val_len, strict=True)
                   if n <= max_tokens]
        ev = take_eval(val_fit, eval_rows, rng)
        st["validation_too_long"] = len(val) - len(val_fit)
        # train pool, cleaned in order; each row counted at its first drop
        pool, copies, conflict = _dedupe(data["train"])
        st["drop_duplicate_copies"] = copies
        st["drop_conflicting_duplicates"] = conflict
        val_keys = {input_key(r) for r in data["validation"]}
        before = len(pool)
        pool = [r for r in pool if input_key(r) not in val_keys]
        st["drop_input_in_validation"] = before - len(pool)
        val_groups = {r["group"] for r in data["validation"]}
        before = len(pool)
        pool = [r for r in pool if r["group"] not in val_groups]
        st["drop_group_in_validation"] = before - len(pool)
        pool = decontaminate(pool, index, st)
        lengths = prompt_lengths(pool, tok, "question_first")
        over = [r for r, n in zip(pool, lengths, strict=True)
                if n > max_tokens]
        near = [r for r, n in zip(pool, lengths, strict=True)
                if n > max_tokens - 64]
        doc_first = prompt_lengths(near, tok, "doc_first")
        st["drop_too_long_question_first"] = len(over)
        st["too_long_doc_first"] = sum(n > max_tokens for n in doc_first)
        fit = [r for r, n in zip(pool, lengths, strict=True)
               if n <= max_tokens]
        st["clean_pool"] = len(fit)
        st["clean_pool_labels"] = dict(sorted(collections.Counter(
            r["label"] for r in fit).items()))
        tr = sample_train(name, fit, PER_DATASET, rng)
        tr_len = prompt_lengths(tr, tok, "question_first")
        ev_len = prompt_lengths(ev, tok, "question_first")
        st.update({
            "train": len(tr), "eval": len(ev),
            "train_labels": dict(sorted(collections.Counter(
                r["label"] for r in tr).items())),
            "eval_labels": dict(sorted(collections.Counter(
                r["label"] for r in ev).items())),
            "train_with_spans": sum(bool(r["spans"]) for r in tr),
            "eval_with_spans": sum(bool(r["spans"]) for r in ev),
            "options": len(tr[0]["labels"]),
            "train_tokens": _length_stats(tr_len),
            "eval_tokens": _length_stats(ev_len),
            "clean_pool_tokens": _length_stats(
                [n for n in lengths if n <= max_tokens]),
        })
        if name == "cosmosqa":
            st["train_none_gold"] = sum(
                is_none_answer(r["labels"][r["label"]]) for r in tr)
            st["clean_pool_none_gold"] = sum(
                is_none_answer(r["labels"][r["label"]]) for r in fit)
        manifest["datasets"][name] = st
        train += [_clean(r) for r in tr]
        evals += [_clean(r) for r in ev]
        print(json.dumps({k: v for k, v in st.items()
                          if not k.startswith("contaminated_examples")}),
              flush=True)
    random.Random(SEED).shuffle(train)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("eval", evals)):
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["totals"] = {"train": len(train), "eval": len(evals)}
    manifest["checks"] = {
        "id_overlap": len({r["id"] for r in train} & {r["id"] for r in evals}),
        "group_overlap": len({r["group"] for r in train}
                             & {r["group"] for r in evals}),
        "duplicate_train_ids": len(train) - len({r["id"] for r in train}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["totals"]), json.dumps(manifest["checks"]))
    return manifest


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true",
                        help="fetch the pinned sources and stop")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--eval-rows", type=int, default=EVAL_ROWS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="subset of datasets (default: all twelve)")
    args = parser.parse_args()
    if args.download:
        download()
        return
    build(pathlib.Path(args.out), args.eval_rows, args.max_tokens,
          args.datasets)


if __name__ == "__main__":
    main()
