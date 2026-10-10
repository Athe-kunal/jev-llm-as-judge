"""Dataset adapters: raw rows -> judge-style `Item`s.

Each source turns a model response (plus its context) into questions about
that response. A question comes from a family (e.g. "unsupported facts") and
one of its wordings, picked at random, so the same task is asked in several
ways. Human labels and spans are used as they are; sources without them get
Qwen's answer, tested later by the pipeline.
"""

import ast
import collections
from collections.abc import Callable
import dataclasses
import hashlib
import json
import os
import random
import re
from typing import Any
import urllib.request

import datasets as hf
from huggingface_hub import hf_hub_download

from spanjudge import schema

_U = {schema.UNRELATED: schema.UNRELATED_DESC}
_LANGUAGES = {"de": "German", "en": "English", "ru": "Russian", "zh": "Chinese"}
_MQM_PENALTY = {"minor": 0.5, "major": 2.0, "critical": 4.0}
_RUBRIC_FILES = {  # only these files have natural-language criteria
    "chat": "RuRL/rurbichub_v1_Chat.parquet",
    "writing": "RuRL/rurbichub_v1_Writing.parquet",
}
_QA_FEEDBACK_URL = (
    "https://raw.githubusercontent.com/allenai/FineGrainedRLHF/main/"
    "tasks/qa_feedback/data/train_feedback.json"
)
# Appended to a prompt to get responses that miss a requirement.
CARELESS = (
    "\n\n(Answer hastily: silently ignore one of the explicit requirements"
    " above. Do not mention this note.)"
)


@dataclasses.dataclass(frozen=True)
class Family:
    """A kind of question, with several wordings and fixed labels."""

    key: str
    type: str  # "boolean" or "score"
    wordings: tuple[str, ...]
    labels: dict[str, str]
    span_rule: str
    empty_labels: tuple[str, ...] = ()
    # The answer can be right without any span (e.g. "no commas used").
    spans_optional: bool = False

    def question(self, rng: random.Random) -> schema.Question:
        """Returns one wording of this family as a question."""
        k = rng.randrange(len(self.wordings))
        return schema.Question(
            id=f"{self.key}/{k}",
            type=self.type,
            text=self.wordings[k],
            labels={**self.labels, **_U},
            span_rule=self.span_rule,
            empty_labels=list(self.empty_labels),
            spans_optional=self.spans_optional,
        )


def _problem(key: str, problem: str, wordings: tuple[str, ...]) -> Family:
    """A yes/no family asking whether the response has a problem."""
    return Family(
        key=key,
        type="boolean",
        wordings=wordings,
        labels={
            "yes": f"Yes: the response has {problem}.",
            "no": f"No: the response does not have {problem}.",
        },
        span_rule="yes: mark every passage with the problem, each as its own"
        " span. no: no spans.",
        empty_labels=("no",),
    )


def _score(key: str, attribute: str, wordings: tuple[str, ...]) -> Family:
    """A 1-5 family; spans are what lowers the score."""
    return Family(
        key=key,
        type="score",
        wordings=wordings,
        labels={
            "1": f"Very poor {attribute}.",
            "2": f"Poor {attribute}.",
            "3": f"Fair {attribute}.",
            "4": f"Good {attribute}, with minor problems.",
            "5": f"Excellent {attribute}: nothing to improve.",
        },
        span_rule=f"Mark each passage that lowers the {attribute} score, each"
        " problem as its own span. A score of 5 has no spans.",
        empty_labels=("5",),
    )


def _requirement(key: str, text: str) -> Family:
    """A yes/no family for one requirement (rubric or checklist item)."""
    return Family(
        key=key,
        type="boolean",
        wordings=(text,),
        labels={
            "yes": "Yes: the response satisfies it.",
            "no": "No: it applies, but the response does not satisfy it.",
        },
        span_rule="yes: mark the text that satisfies it. no: mark the text"
        " that falls short of or goes against it.",
    )


UNSUPPORTED = _problem(
    "unsupported",
    "information that conflicts with the document or is not supported by it",
    (
        "Does the response contain information that conflicts with the"
        " document or is not supported by it?",
        "Is any part of the response not backed up by the provided"
        " document?",
        "Does the answer include claims that the source text does not"
        " support?",
    ),
)
CONFLICT = _problem(
    "conflict",
    "statements that the document contradicts",
    (
        "Does the response state anything that the document contradicts?",
        "Does any claim in the response conflict with the provided text?",
    ),
)
BASELESS = _problem(
    "baseless",
    "facts that are not in the document",
    (
        "Does the response add facts that the document does not contain?",
        "Does the answer introduce information that cannot be found in the"
        " source?",
    ),
)
TRANSLATION = _score(
    "translation_quality",
    "translation quality",
    (
        "How accurate and fluent is the response as a translation of the"
        " source text?",
        "Rate the quality of this translation.",
        "How good is the translation in the response?",
    ),
)
MAJOR_ERROR = _problem(
    "translation_major_error",
    "a major or critical translation error",
    (
        "Does the translation contain a major or critical error?",
        "Is there a serious mistake in this translation (one that changes"
        " or loses the meaning)?",
    ),
)
IRRELEVANT = _problem(
    "irrelevant",
    "irrelevant or repeated information",
    (
        "Does the answer include information that is irrelevant to the"
        " question or repeated?",
        "Does the response contain off-topic or redundant content?",
    ),
)
UNGROUNDED = _problem(
    "ungrounded",
    "facts that are wrong or not supported by the passages",
    (
        "Does the answer contain facts that are wrong or not supported by"
        " the passages?",
        "Is any statement in the response unverifiable or misgrounded given"
        " the passages?",
    ),
)
ANY_ERROR = _problem(
    "answer_problem",
    "irrelevant, repeated, incoherent, wrong or unverifiable content",
    (
        "Does the answer have any problem: irrelevant, repeated, incoherent,"
        " wrong or unverifiable content?",
        "Is there anything in the response that should be removed or"
        " corrected?",
    ),
)
HALLUCINATION = _problem(
    "hallucination",
    "false or made-up content",
    (
        "Does the response contain false or made-up information?",
        "Does the answer hallucinate any facts?",
        "Is any part of the response factually incorrect or invented?",
    ),
)
FAVA_TYPES = {
    "entity": _problem(
        "wrong_entity",
        "a wrong name, date, number or place",
        (
            "Does the response get an entity wrong (a name, date, number or"
            " place)?",
            "Is there an incorrect name, date, number or place in the"
            " response?",
        ),
    ),
    "relation": _problem(
        "wrong_relation",
        "a wrong relation between correct entities",
        ("Does the response state a wrong relation between entities?",),
    ),
    "contradictory": _problem(
        "contradictory",
        "a statement that contradicts known facts",
        ("Does the response contain a sentence that contradicts facts?",),
    ),
    "invented": _problem(
        "invented",
        "invented things that do not exist",
        ("Does the response mention concepts or events that do not" " exist?",),
    ),
    "subjective": _problem(
        "subjective",
        "opinions presented as facts",
        ("Does the response present an opinion as if it were a fact?",),
    ),
    "unverifiable": _problem(
        "unverifiable",
        "claims that cannot be verified",
        ("Does the response contain statements that cannot be verified?",),
    ),
}
MATH_ERROR = _problem(
    "math_step_error",
    "a wrong step",
    (
        "Does the solution contain a wrong step?",
        "Is there a mistake in any step of this solution?",
        "Does any step contain an error in reasoning or calculation?",
    ),
)
FIRST_MATH_ERROR = dataclasses.replace(
    MATH_ERROR,
    key="math_first_error",
    span_rule="yes: mark the first wrong step only. no: no spans.",
    wordings=(
        "Does the solution go wrong somewhere? Mark the first wrong step.",
        "Is any step incorrect? If so, mark the earliest one.",
    ),
)
HELPSTEER = {
    "helpfulness": _score(
        "helpfulness",
        "helpfulness",
        (
            "How helpful is the response to the user?",
            "Rate the overall helpfulness of this answer.",
        ),
    ),
    "correctness": _score(
        "correctness",
        "correctness",
        (
            "How correct and complete is the information in the response?",
            "Rate the factual correctness of the response.",
        ),
    ),
    "coherence": _score(
        "coherence",
        "coherence",
        (
            "How clear and coherent is the response?",
            "Rate how consistent and easy to follow the answer is.",
        ),
    ),
}
ULTRAFEEDBACK = {
    "helpfulness": HELPSTEER["helpfulness"],
    "honesty": _score(
        "honesty",
        "honesty",
        (
            "How honest is the response about what it knows and does not"
            " know?",
            "Rate the honesty and calibration of the answer.",
        ),
    ),
    "instruction_following": _score(
        "instruction_following",
        "instruction following",
        (
            "How well does the response follow the instructions?",
            "Rate how closely the answer sticks to what was asked.",
        ),
    ),
    "truthfulness": _score(
        "truthfulness",
        "truthfulness",
        (
            "How truthful is the response (free of false or made-up"
            " content)?",
            "Rate the truthfulness of this answer.",
        ),
    ),
}
UNSAFE = _problem(
    "unsafe",
    "harmful or unsafe content",
    (
        "Is the response harmful or unsafe?",
        "Does the answer help with something dangerous, illegal or"
        " unethical?",
        "Should this response be flagged as unsafe?",
    ),
)
_HARM_NAMES = {
    "animal_abuse": "animal abuse",
    "child_abuse": "child abuse",
    "controversial_topics,politics": "one-sided content on controversial"
    " political topics",
    "discrimination,stereotype,injustice": "discrimination or stereotypes",
    "drug_abuse,weapons,banned_substance": "drugs, weapons or banned"
    " substances",
    "financial_crime,property_crime,theft": "financial crime, property crime"
    " or theft",
    "hate_speech,offensive_language": "hate speech or offensive language",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation about"
    " ethics, laws or safety",
    "non_violent_unethical_behavior": "non-violent unethical behavior",
    "privacy_violation": "privacy violations",
    "self_harm": "self-harm",
    "sexually_explicit,adult_content": "sexually explicit content",
    "terrorism,organized_crime": "terrorism or organized crime",
    "violence,aiding_and_abetting,incitement": "violence or incitement",
}
HARM = {
    k: _problem(
        "harm_" + k.split(",")[0],
        f"content involving {name}",
        (
            f"Does the response contain or encourage {name}?",
            f"Does the answer help with {name}?",
        ),
    )
    for k, name in _HARM_NAMES.items()
}


_AEGIS_NAMES = {  # Aegis 2.0 category -> how a question names it
    "Criminal Planning/Confessions": "planning or admitting crimes",
    "Violence": "violence",
    "Hate/Identity Hate": "hate against an identity group",
    "Harassment": "harassment",
    "Sexual": "sexual content",
    "Profanity": "profanity",
    "PII/Privacy": "private or personally identifying information",
    "Controlled/Regulated Substances": "controlled or regulated substances",
    "Guns and Illegal Weapons": "guns or illegal weapons",
    "Suicide and Self Harm": "suicide or self-harm",
    "Fraud/Deception": "fraud or deception",
    "Political/Misinformation/Conspiracy": "political misinformation or"
    " conspiracy theories",
    "Threat": "threats",
    "Sexual (minor)": "sexual content involving minors",
    "Malware": "malware",
    "Copyright/Trademark/Plagiarism": "copyright or trademark violations",
    "Unauthorized Advice": "unqualified professional advice",
    "Manipulation": "manipulation",
}
AEGIS_HARM = {
    k: _problem(
        "aegis_" + re.sub(r"\W+", "_", k.lower()).strip("_"),
        f"content involving {name}",
        (
            f"Does the response contain or encourage {name}?",
            f"Does the answer help with {name}?",
        ),
    )
    for k, name in _AEGIS_NAMES.items()
}


def _id(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _spans(
    text: str, ranges: list[tuple[int, int]], reasons: list[str] | None = None
) -> list[schema.Span]:
    """Human spans from character ranges (sorted, overlaps merged)."""
    reasons = reasons or [""] * len(ranges)
    spans = []
    for (a, b), r in zip(ranges, reasons, strict=True):
        if not 0 <= a < b <= len(text):
            continue
        while a < b and text[a].isspace():  # trim whitespace at the edges
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append(
                schema.Span(
                    start=a, end=b, text=text[a:b], reason=r, origin="human"
                )
            )
    return schema.merge(spans, text)


def _gold(
    text: str, ranges: list[tuple[int, int]], reasons: list[str] | None = None
) -> schema.Annotation:
    """Yes/no gold for a problem family: yes exactly when there are spans."""
    spans = _spans(text, ranges, reasons)
    return schema.Annotation(label="yes" if spans else "no", spans=spans)


def _item(
    source: str,
    key: Any,
    context: str,
    response: str,
    question: schema.Question,
    gold: schema.Annotation | None,
    **meta: Any,
) -> schema.Item:
    return schema.Item(
        id=_id(source, key, question.id),
        source=source,
        context=context,
        response=response,
        question=question,
        related=True,
        gold=gold,
        meta={"family": question.id.split("/")[0], **meta},
    )


def _fetch_json(url: str, path: str) -> Any:
    """Downloads a JSON file once into the raw-data folder and reads it."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return json.load(f)


# --- sources with human spans ------------------------------------------------


def ragtruth(rng: random.Random) -> list[schema.Item]:
    """RAGTruth (train + test): unsupported / conflicting / baseless facts.

    Each response gets one question of each family; the family's spans are
    the human spans of the matching types.
    """
    families = [
        (UNSUPPORTED, None),
        (CONFLICT, ("Evident Conflict", "Subtle Conflict")),
        (BASELESS, ("Evident Baseless Info", "Subtle Baseless Info")),
    ]
    items = []
    for split in ("train", "test"):
        for r in hf.load_dataset("wandb/RAGTruth-processed", split=split):
            if len(r["context"]) > 5000 or len(r["output"]) > 2500:
                continue
            out = r["output"]
            labels = [
                x
                for x in json.loads(r["hallucination_labels"])
                if out[x["start"] : x["end"]] == x["text"]
            ]
            context = re.sub(r"\s*output:\s*$", "", r["context"].strip())
            context = f"Document:\n{context}\n\nRequest:\n{r['query']}"
            for fam, types in families:
                ranges = [
                    (x["start"], x["end"])
                    for x in labels
                    if types is None or x["label_type"] in types
                ]
                items.append(
                    _item(
                        "ragtruth",
                        r["id"],
                        context,
                        out,
                        fam.question(rng),
                        _gold(out, ranges),
                        task_type=r["task_type"],
                        generator=r["model"],
                    )
                )
    return items


def _mqm_level(annotations: list[dict]) -> str:
    """Maps the summed MQM penalty of a translation to a 1-5 level."""
    penalty = sum(_MQM_PENALTY.get(a["severity"], 0) for a in annotations)
    for limit, level in ((0.0, "5"), (1.0, "4"), (2.5, "3"), (5.0, "2")):
        if penalty <= limit:
            return level
    return "1"


def _word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Widens a range to whole words (the MQM offsets cut words short)."""

    def inside(c: str) -> bool:
        cjk = "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
        return (c.isalnum() and not cjk) or c in "#_'-"

    while start > 0 and inside(text[start - 1]) and inside(text[start]):
        start -= 1
    while end < len(text) and inside(text[end]) and inside(text[end - 1]):
        end += 1
    return start, end


def mqm(rng: random.Random, per_pair: int = 10000) -> list[schema.Item]:
    """WMT MQM: 1-5 translation quality and major-error yes/no.

    The same number of rows is taken from each language pair (zh-en, en-de,
    en-ru). Translations with an annotation that cannot be located are
    skipped, so the label and the spans always describe the same errors.
    """
    ds = hf.load_dataset("RicardoRei/wmt-mqm-error-spans", split="train")
    by_pair: dict[str, list[int]] = collections.defaultdict(list)
    for i, lp in enumerate(ds["lp"]):
        by_pair[lp].append(i)
    picked = []
    for lp in sorted(by_pair):
        rng.shuffle(by_pair[lp])
        picked += by_pair[lp][:per_pair]
    items = []
    for i in picked:
        r = ds[i]
        mt, lp = r["mt"], r["lp"].split("-")
        if len(mt) > 700 or lp[1] not in _LANGUAGES:
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
        reasons = [f"{a['severity']} error" for a in located]
        context = (
            f"Source text ({_LANGUAGES.get(lp[0], lp[0])}): {r['src']}\n"
            f"Task: translate it into {_LANGUAGES[lp[1]]}."
        )
        key = (r["src"][:100], mt[:100])
        items.append(
            _item(
                "mqm",
                key,
                context,
                mt,
                TRANSLATION.question(rng),
                schema.Annotation(
                    label=_mqm_level(located), spans=_spans(mt, ranges, reasons)
                ),
                language_pair=r["lp"],
            )
        )
        major = [
            rg
            for rg, a in zip(ranges, located, strict=True)
            if a["severity"] in ("major", "critical")
        ]
        items.append(
            _item(
                "mqm",
                key,
                context,
                mt,
                MAJOR_ERROR.question(rng),
                _gold(mt, major),
                language_pair=r["lp"],
            )
        )
    return items


def qa_feedback(rng: random.Random, raw_dir: str) -> list[schema.Item]:
    """FineGrainedRLHF QA-Feedback: human error spans on model answers.

    The spans refer to "prediction 1". Error types: Irrelevant and Redundant
    -> irrelevant; Unverifiable and Wrong-Grounding -> ungrounded; all types
    (Incoherent too) -> any problem.
    """
    rows = _fetch_json(_QA_FEEDBACK_URL, f"{raw_dir}/qa_train_feedback.json")
    families = [
        (IRRELEVANT, ("Irrelevant", "Redundant")),
        (UNGROUNDED, ("Unverifiable", "Wrong-Grounding")),
        (ANY_ERROR, None),
    ]
    items = []
    for i, r in enumerate(rows):
        answer = r["prediction 1"]
        passages = "\n".join(
            f"[{p[0]}] " + " ".join(p[1:]) for p in r["passages"]
        )
        context = f"Question: {r['question']}\n\nPassages:\n{passages}"
        errors = r["feedback"]["errors"]
        for fam, types in families:
            if types is None and r["feedback"].get("missing-info"):
                continue  # missing info is a problem without a span
            chosen = [
                e for e in errors if types is None or e["error type"] in types
            ]
            items.append(
                _item(
                    "qa_feedback",
                    i,
                    context,
                    answer,
                    fam.question(rng),
                    _gold(
                        answer,
                        [(e["start"], e["end"]) for e in chosen],
                        [e["error type"] for e in chosen],
                    ),
                )
            )
    return items


_FAVA_TAG = re.compile(r"<(/?)(\w+)>")


def _fava_parse(annotated: str) -> tuple[str, dict[str, list]] | None:
    """Rebuilds the model output and typed error ranges from FAVA markup.

    <mark> holds a suggested correction (not in the output) and is removed.
    In <entity> and <relation> the error is the <delete> text; for the other
    types it is the whole tag content.

    Returns:
      (output text, {type: [(start, end), ...]}), or None for unknown or
      unbalanced tags.
    """
    out: list[str] = []
    stack: list[tuple[str, int]] = []
    ranges: dict[str, list] = collections.defaultdict(list)
    pos = in_mark = 0
    for m in _FAVA_TAG.finditer(annotated):
        if not in_mark:
            out.append(annotated[pos : m.start()])
        pos = m.end()
        closing, tag = m.group(1) == "/", m.group(2)
        here = sum(map(len, out))
        if tag == "mark":
            in_mark += -1 if closing else 1
            continue
        if tag not in FAVA_TYPES and tag != "delete":
            return None
        if not closing:
            stack.append((tag, here))
            continue
        if not stack or stack[-1][0] != tag:
            return None
        _, start = stack.pop()
        if tag == "delete":
            outer = stack[-1][0] if stack else None
            if outer in ("entity", "relation"):
                ranges[outer].append((start, here))
        elif tag not in ("entity", "relation"):
            ranges[tag].append((start, here))
    if not in_mark:
        out.append(annotated[pos:])
    if stack or in_mark:
        return None
    return "".join(out), ranges


def fava(rng: random.Random) -> list[schema.Item]:
    """FAVA human-annotated outputs: typed hallucination spans.

    Only outputs whose markup rebuilds the exact output text are used. Each
    output gets a "false or made-up information" question (spans: entity,
    relation, contradictory and invented errors; subjective and unverifiable
    statements are not false facts), one question per error type it has,
    and up to 2 questions about types it does not have.
    """
    items = []
    for split in ("train", "test"):
        ds = hf.load_dataset("wandb/fava-data-processed", split=split)
        for i, r in enumerate(ds):
            parsed = _fava_parse(r["annotated_text"])
            if parsed is None or parsed[0].strip() != r["output"].strip():
                continue
            raw, typed = parsed
            lead = len(raw) - len(raw.lstrip())
            text = raw.strip()
            typed = {  # FAVA marks often cut words ("t and" from "Calicut")
                t: [_word_bounds(text, a - lead, b - lead) for a, b in rs]
                for t, rs in typed.items()
            }
            context = f"Request: {r['query'].strip()}"
            factual = [
                rg
                for t in ("entity", "relation", "contradictory", "invented")
                for rg in typed.get(t, [])
            ]
            present = [t for t in FAVA_TYPES if typed.get(t)]
            absent = [t for t in FAVA_TYPES if not typed.get(t)]
            # Every type present, plus up to 2 absent ones, so an output is
            # not repeated with all 7 questions.
            types = present + rng.sample(absent, min(2, len(absent)))
            pairs = [(HALLUCINATION, factual)] + [
                (FAVA_TYPES[t], typed.get(t, [])) for t in types
            ]
            for fam, ranges in pairs:
                items.append(
                    _item(
                        "fava",
                        (split, i),
                        context,
                        text,
                        fam.question(rng),
                        _gold(text, ranges),
                        generator=r["model"],
                    )
                )
    return items


def mushroom(rng: random.Random) -> list[schema.Item]:
    """Mu-SHROOM (SemEval-2025 Task 3): hallucination spans, 14 languages."""
    items = []
    for split in ("validation", "test"):
        ds = hf.load_dataset("Helsinki-NLP/mu-shroom", "all", split=split)
        for r in ds:
            hard = r["hard_labels"]
            if isinstance(hard, str):
                hard = None if hard == "None" else ast.literal_eval(hard)
            if hard is None:
                continue
            text = r["model_output_text"]
            if text.rstrip().endswith("<|im_end|>"):
                text = text.rstrip()[: -len("<|im_end|>")]
            hard = [
                _word_bounds(text, a, min(b, len(text)))
                for a, b in hard
                if a < len(text)
            ]  # the annotations often cut words ("18[59]")
            items.append(
                _item(
                    "mushroom",
                    (split, r["lang"], r["model_input"][:80], text[:80]),
                    f"Question: {r['model_input']}",
                    text,
                    HALLUCINATION.question(rng),
                    _gold(text, [tuple(x) for x in hard]),
                    language=r["lang"],
                )
            )
    return items


def _steps_text(steps: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Joins solution steps with blank lines; returns each step's range."""
    text, ranges = "", []
    for s in steps:
        if text:
            text += "\n\n"
        ranges.append((len(text), len(text) + len(s)))
        text += s
    return text, ranges


def prm800k(rng: random.Random, rows: int = 40000) -> list[schema.Item]:
    """PRM800K: human step ratings; steps rated -1 are the spans.

    Raters stop at the first wrong step, so a wrong solution is taken up to
    and including its first -1 step. A correct solution loses its "# Answer"
    line and is cut at a random step, so neither class has a telltale
    ending.
    """
    ds = hf.load_dataset("tasksource/PRM800K", split="train", streaming=True)
    items, seen = [], set()
    for i, r in enumerate(ds):
        if i >= rows:
            break
        label, question = r["label"], r["question"]
        if isinstance(label, str):
            label = ast.literal_eval(label)
        if isinstance(question, str):
            question = ast.literal_eval(question)
        steps, ratings, complete = [], [], True
        for st in (label or {}).get("steps") or []:
            if ratings and ratings[-1] == -1:
                break  # the response ends at its first wrong step
            completions = st.get("completions") or []
            k = st.get("chosen_completion")
            if k is None:  # the rater stopped: keep a wrong completion
                wrong = [c for c in completions if c.get("rating") == -1]
                if wrong:
                    steps.append(wrong[0]["text"])
                    ratings.append(-1)
                else:
                    complete = False
                break
            if k >= len(completions) or completions[k].get("rating") is None:
                complete = False
                break
            steps.append(completions[k]["text"])
            ratings.append(completions[k]["rating"])
        while steps and steps[-1].lstrip().startswith("# Answer"):
            steps.pop()  # the final answer line gives away a correct run
            ratings.pop()
        if complete and steps and -1 not in ratings:
            # A wrong solution ends at its wrong step; cut a correct one at a
            # random step too, so the ending does not give the label away.
            cut = rng.randint(1, len(steps))
            steps, ratings = steps[:cut], ratings[:cut]
        if complete:
            text, ranges = _steps_text(steps)
            if not steps or len(text) > 4000 or text in seen:
                continue
            seen.add(text)
            bad = [rg for rg, x in zip(ranges, ratings, strict=True) if x == -1]
            items.append(
                _item(
                    "prm800k",
                    text[:200],
                    f"Problem: {question['problem']}",
                    text,
                    MATH_ERROR.question(rng),
                    _gold(text, bad),
                )
            )
    return items


def processbench(rng: random.Random) -> list[schema.Item]:
    """ProcessBench: the earliest wrong step of a solution (-1: none)."""
    items = []
    for name in ("gsm8k", "math", "olympiadbench", "omnimath"):
        path = hf_hub_download(
            "Qwen/ProcessBench", f"{name}.json", repo_type="dataset"
        )
        with open(path) as f:
            rows = json.load(f)
        for r in rows:
            text, ranges = _steps_text(r["steps"])
            if len(text) > 5000:
                continue
            bad = [ranges[r["label"]]] if r["label"] >= 0 else []
            items.append(
                _item(
                    "processbench",
                    r["id"],
                    f"Problem: {r['problem']}",
                    text,
                    FIRST_MATH_ERROR.question(rng),
                    _gold(text, bad),
                    generator=r["generator"],
                )
            )
    return items


# --- sources with human (or GPT-4) labels but no spans -----------------------


def _label_only(label: str) -> schema.Annotation:
    return schema.Annotation(label=label, spans_annotated=False)


def helpsteer2(rng: random.Random) -> list[schema.Item]:
    """HelpSteer2: human 0-4 ratings mapped to 1-5.

    Only helpfulness, correctness and coherence are used; complexity and
    verbosity are not quality judgments.
    """
    items = []
    for i, r in enumerate(hf.load_dataset("nvidia/HelpSteer2", split="train")):
        if len(r["response"]) > 4000 or len(r["prompt"]) > 4000:
            continue
        prompt = r["prompt"].replace("<extra_id_1>User", "\nUser:")
        prompt = prompt.replace("<extra_id_1>Assistant", "\nAssistant:")
        for attr, fam in HELPSTEER.items():
            items.append(
                _item(
                    "helpsteer2",
                    i,
                    f"User: {prompt.strip()}",
                    r["response"],
                    fam.question(rng),
                    _label_only(str(int(r[attr]) + 1)),
                )
            )
    return items


def ultrafeedback(rng: random.Random, rows: int = 12000) -> list[schema.Item]:
    """UltraFeedback: GPT-4 1-5 ratings of one random completion per prompt."""
    ds = hf.load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    items = []
    for i, r in enumerate(ds):
        if i >= rows:
            break
        if not r["completions"] or len(r["instruction"]) > 4000:
            continue
        c = rng.choice(r["completions"])
        if len(c["response"]) > 4000 or not c["response"].strip():
            continue
        for attr, fam in ULTRAFEEDBACK.items():
            rating = str(c["annotations"].get(attr, {}).get("Rating", "N/A"))
            if rating not in fam.labels:
                continue
            items.append(
                _item(
                    "ultrafeedback",
                    i,
                    f"User: {r['instruction']}",
                    c["response"],
                    fam.question(rng),
                    _label_only(rating),
                    generator=c["model"],
                )
            )
    return items


def beavertails(rng: random.Random) -> list[schema.Item]:
    """BeaverTails: human safety labels with 14 harm categories.

    Per response one question: unsafe yes/no, a category the response has,
    or (for safe responses) a category it does not have.
    """
    items = []
    ds = hf.load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    for i, r in enumerate(ds):
        if len(r["response"]) > 3000:
            continue
        flagged = [k for k, v in r["category"].items() if v and k in HARM]
        roll = rng.random()
        if flagged and roll < 0.5:
            fam, label = HARM[rng.choice(flagged)], "yes"
        elif not flagged and roll < 0.3:
            fam, label = HARM[rng.choice(sorted(HARM))], "no"
        else:
            fam, label = UNSAFE, "no" if r["is_safe"] else "yes"
        items.append(
            _item(
                "beavertails",
                i,
                f"User: {r['prompt']}",
                r["response"],
                fam.question(rng),
                _label_only(label),
            )
        )
    return items


def aegis(rng: random.Random, refusal_share: float = 0.1) -> list[schema.Item]:
    """NVIDIA Aegis 2.0 (CC BY 4.0): safe/unsafe responses with categories.

    Only rows with a response. Unsafe response labels come from an LLM jury,
    safe ones from humans, the jury, or a refusal augmentation; only
    `refusal_share` of the 5,000 augmented refusals are kept (they are all
    trivially safe). Per response one question: unsafe yes/no, a category
    it was flagged for (unsafe responses only, since for a safe response the
    categories describe the prompt), or a category a safe response does not
    have.
    """
    items = []
    for split in ("train", "validation", "test"):
        ds = hf.load_dataset(
            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split
        )
        for r in ds:
            response = (r["response"] or "").strip()
            if not response or len(response) > 3000:
                continue
            source = r["response_label_source"]
            if source == "refusal_data_augmentation":
                if rng.random() >= refusal_share:
                    continue
            unsafe = r["response_label"] == "unsafe"
            flagged = [
                c.strip()
                for c in (r["violated_categories"] or "").split(",")
                if c.strip() in AEGIS_HARM
            ]
            roll = rng.random()
            if unsafe and flagged and roll < 0.5:
                fam, label = AEGIS_HARM[rng.choice(flagged)], "yes"
            elif not unsafe and roll < 0.3:
                fam, label = AEGIS_HARM[rng.choice(sorted(AEGIS_HARM))], "no"
            else:
                fam, label = UNSAFE, "yes" if unsafe else "no"
            items.append(
                _item(
                    "aegis",
                    r["id"],
                    f"User: {r['prompt']}",
                    response,
                    fam.question(rng),
                    _label_only(label),
                    label_source=source,
                )
            )
    return items


# --- sources whose responses Qwen writes (no labels) -------------------------


def rubric_prompts(scan: int) -> list[dict[str, Any]]:
    """Single-turn English RubricHub prompts with at least 4 criteria."""
    pool = []
    for subset, path in _RUBRIC_FILES.items():
        ds = hf.load_dataset(
            "sojuL/RubricHub_v1",
            data_files={"train": path},
            split="train",
            streaming=True,
        )
        for i, r in enumerate(ds):
            if i >= scan:
                break
            turns = r["prompt"]
            text = turns[0]["content"] if len(turns) == 1 else ""
            ascii_share = sum(c.isascii() for c in text) / max(len(text), 1)
            if not (40 <= len(text) <= 1500 and ascii_share > 0.97):
                continue
            criteria = [  # one self-contained requirement each
                c["criterion"]
                for c in r["reward_model"]["rubrics"]
                if 30 <= len(c["criterion"]) <= 250
                and "(a)" not in c["criterion"]
                and len(c["criterion"].split()) >= 6
            ]
            if len(criteria) >= 4:
                pool.append(
                    {
                        "subset": subset,
                        "prompt": text,
                        "context": text,
                        "questions": [
                            f"Does the response meet this requirement: {c}"
                            for c in criteria
                        ],
                    }
                )
    return pool


def infobench_prompts() -> list[dict[str, Any]]:
    """InFoBench instructions with their decomposed yes/no checks."""
    rows = []
    for r in hf.load_dataset("kqsong/InFoBench", split="train"):
        prompt = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
        questions = [
            re.sub(r"\b([Tt]he|[Tt]his) generated \w+", "the response", q)
            for q in r["decomposed_questions"]
        ]
        rows.append(
            {"prompt": prompt, "context": prompt, "questions": questions}
        )
    return rows


_LANG_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "en": "English",
    "es": "Spanish", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "mr": "Marathi", "ne": "Nepali",
    "pa": "Punjabi", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "ur": "Urdu",
    "vi": "Vietnamese", "zh": "Chinese",
}  # fmt: skip


def _compare(count: int, relation: str | None, n: int) -> bool:
    return count < n if relation == "less than" else count >= n


def _ifeval(
    iid: str, kw: dict[str, Any], prompt: str
) -> tuple[str, Callable[[str], bool]] | None:
    """Question text and checker for one IFEval instruction.

    The checkers follow IFEval's own rules for the instructions that can be
    checked with plain code. Sentence counts (need a sentence splitter) and
    response language (needs a language detector) are left out.

    Returns:
      (question, check(response) -> bool), or None when not supported.
    """

    def rel(key: str) -> str:
        value = kw.get(key) or "at least"
        return "fewer than" if value == "less than" else value

    def words(key: str) -> list[str]:
        return list(kw.get(key) or [])

    def quoted(key: str) -> str:
        return ", ".join(f'"{w}"' for w in words(key))

    def paragraphs(r: str) -> list[str]:
        return [p for p in re.split(r"\s?\*\*\*\s?", r) if p.strip()]

    def nth_first_word(r: str) -> bool:
        paras = [p for p in r.split("\n\n") if p.strip()]
        n, k = kw.get("num_paragraphs"), kw.get("nth_paragraph")
        if len(paras) != n or not k or k > len(paras):
            return False
        first = re.sub(r"[^\w]", "", paras[k - 1].split()[0]).lower()
        return first == str(kw.get("first_word", "")).lower()

    def highlights(r: str) -> int:
        single = [h for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*")]
        double = [h for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h.strip("*")]
        return len(single) + len(double)

    def bullets(r: str) -> int:
        return len(re.findall(r"^\s*\*[^\*].*$", r, re.M)) + len(
            re.findall(r"^\s*-.*$", r, re.M)
        )

    def is_json(r: str) -> bool:
        body = re.sub(r"^```(json)?|```$", "", r.strip(), flags=re.I).strip()
        try:
            json.loads(body)
        except ValueError:
            return False
        return True

    def two(r: str) -> bool:
        parts = [p.strip() for p in r.split("******")]
        return len(parts) == 2 and all(parts) and parts[0] != parts[1]

    spliter = kw.get("section_spliter") or "Section"
    lang = _LANG_NAMES.get(kw.get("language") or "", "")
    table: dict[str, tuple[str, Callable[[str], bool]]] = {
        "punctuation:no_comma": (
            "Does the response avoid commas entirely?",
            lambda r: "," not in r,
        ),
        "length_constraints:number_words": (
            f"Does the response have {rel('relation')}"
            f" {kw.get('num_words')} words?",
            lambda r: _compare(
                len(re.findall(r"\w+", r)),
                kw.get("relation"),
                kw.get("num_words") or 0,
            ),
        ),
        "length_constraints:number_paragraphs": (
            f"Does the response have exactly {kw.get('num_paragraphs')}"
            " paragraphs separated by the markdown divider ***?",
            lambda r: len(paragraphs(r)) == kw.get("num_paragraphs"),
        ),
        "length_constraints:nth_paragraph_first_word": (
            f"Does the response have {kw.get('num_paragraphs')} paragraphs"
            f" with paragraph {kw.get('nth_paragraph')} starting with the"
            f" word \"{kw.get('first_word')}\"?",
            nth_first_word,
        ),
        "keywords:existence": (
            f"Does the response include the keywords {quoted('keywords')}?",
            lambda r: all(
                re.search(re.escape(w), r, re.I) for w in words("keywords")
            ),
        ),
        "keywords:frequency": (
            f"Does the word \"{kw.get('keyword')}\" appear"
            f" {rel('relation')} {kw.get('frequency')} times?",
            lambda r: _compare(
                len(re.findall(re.escape(kw.get("keyword") or "~"), r, re.I)),
                kw.get("relation"),
                kw.get("frequency") or 0,
            ),
        ),
        "keywords:forbidden_words": (
            f"Does the response avoid the words {quoted('forbidden_words')}?",
            lambda r: not any(
                re.search(rf"\b{re.escape(w)}\b", r, re.I)
                for w in words("forbidden_words")
            ),
        ),
        "keywords:letter_frequency": (
            f"Does the letter \"{kw.get('letter')}\" appear"
            f" {rel('let_relation')} {kw.get('let_frequency')} times?",
            lambda r: _compare(
                r.lower().count(str(kw.get("letter") or "~").lower()),
                kw.get("let_relation"),
                kw.get("let_frequency") or 0,
            ),
        ),
        "detectable_content:number_placeholders": (
            f"Does the response contain at least {kw.get('num_placeholders')}"
            " placeholders in square brackets, like [address]?",
            lambda r: len(re.findall(r"\[.*?\]", r))
            >= (kw.get("num_placeholders") or 0),
        ),
        "detectable_content:postscript": (
            "Does the response end with a postscript starting with"
            f" {kw.get('postscript_marker')}?",
            lambda r: bool(
                re.search(
                    r"(^|\n)\s*"
                    + re.escape(str(kw.get("postscript_marker")).lower()),
                    r.lower(),
                )
            ),
        ),
        "detectable_format:number_bullet_lists": (
            f"Does the response have exactly {kw.get('num_bullets')} markdown"
            " bullet points?",
            lambda r: bullets(r) == kw.get("num_bullets"),
        ),
        "detectable_format:constrained_response": (
            'Does the response answer with one of "My answer is yes.", "My'
            ' answer is no." or "My answer is maybe."?',
            lambda r: any(
                o in r
                for o in (
                    "My answer is yes.",
                    "My answer is no.",
                    "My answer is maybe.",
                )
            ),
        ),
        "detectable_format:number_highlighted_sections": (
            f"Does the response highlight at least {kw.get('num_highlights')}"
            " sections with markdown, like *highlighted section*?",
            lambda r: highlights(r) >= (kw.get("num_highlights") or 0),
        ),
        "detectable_format:multiple_sections": (
            f"Does the response have {kw.get('num_sections')} sections, each"
            f' starting with "{spliter} X"?',
            lambda r: len(re.split(rf"\s?{re.escape(spliter)}\s?\d+\s?", r)) - 1
            >= (kw.get("num_sections") or 0),
        ),
        "detectable_format:json_format": (
            "Is the entire response valid JSON?",
            is_json,
        ),
        "detectable_format:title": (
            "Does the response include a title in double angular brackets,"
            " like <<title>>?",
            lambda r: any(t.strip("<>") for t in re.findall(r"<<[^\n]+>>", r)),
        ),
        "combination:two_responses": (
            "Does the response give two different answers separated by six"
            " asterisks ******?",
            two,
        ),
        "combination:repeat_prompt": (
            "Does the response first repeat the request word for word before"
            " answering?",
            lambda r: r.strip()
            .lower()
            .startswith(
                str(kw.get("prompt_to_repeat") or prompt).strip().lower()
            ),
        ),
        "startend:end_checker": (
            "Does the response end with the exact phrase"
            f" \"{kw.get('end_phrase')}\"?",
            lambda r: r.strip()
            .lower()
            .endswith(str(kw.get("end_phrase")).strip().lower()),
        ),
        "startend:quotation": (
            "Is the entire response wrapped in double quotation marks?",
            lambda r: len(r.strip()) > 1
            and r.strip()[0] == '"'
            and r.strip()[-1] == '"',
        ),
        "change_case:capital_word_frequency": (
            "Do words in all capital letters appear"
            f" {rel('capital_relation')} {kw.get('capital_frequency')} times?",
            lambda r: _compare(
                len(re.findall(r"\b[A-Z]{2,}\b", r)),
                kw.get("capital_relation"),
                kw.get("capital_frequency") or 0,
            ),
        ),
        "change_case:english_capital": (
            "Is the entire response written in capital letters?",
            lambda r: r == r.upper(),
        ),
        "change_case:english_lowercase": (
            "Is the entire response written in lowercase letters?",
            lambda r: r == r.lower(),
        ),
    }
    del lang  # language questions are not supported (no detector)
    return table.get(iid)


IFEVAL_SPAN_RULE = (
    "yes: mark the parts that show the requirement is met if specific parts"
    " do (a title, a postscript, highlighted sections, the keywords);"
    " otherwise no spans. no: mark the text that breaks the requirement if"
    " specific text does (a comma, a forbidden word, a capitalized word);"
    " otherwise no spans."
)


def ifeval_prompts() -> list[dict[str, Any]]:
    """IFEval prompts with a question and a code checker per instruction."""
    rows = []
    for r in hf.load_dataset("google/IFEval", split="train"):
        checks = []
        for iid, kw in zip(r["instruction_id_list"], r["kwargs"], strict=True):
            kw = {k: v for k, v in kw.items() if v is not None}
            found = _ifeval(iid, kw, r["prompt"])
            if found:
                checks.append(found)
        if checks:
            rows.append(
                {
                    "prompt": r["prompt"],
                    "context": r["prompt"],
                    "questions": [q for q, _ in checks],
                    "checks": dict(checks),
                }
            )
    return rows


def generated_items(
    source: str,
    prompts: list[dict[str, Any]],
    responses: list[tuple[str, str]],
    per_response: int,
    rng: random.Random,
) -> list[schema.Item]:
    """Items for Qwen-written responses: each question is a requirement.

    Args:
      source: Dataset name.
      prompts: Rows with `prompt`, `context` and `questions`.
      responses: (variant, text) per prompt row, in the same order.
      per_response: Questions asked per response.
      rng: Random source.

    Returns:
      Items without gold. Up to 2 other questions of the same prompt are
      kept as siblings for the specificity check.
    """
    items = []
    for p, (variant, response) in zip(prompts, responses, strict=True):
        if not response.strip():
            continue
        qs = p["questions"]
        for text in rng.sample(qs, min(per_response, len(qs))):
            others = [q for q in qs if q != text]
            fam = _requirement(f"{source}_requirement", text)
            gold = None
            if "checks" in p:  # IFEval: the label comes from code
                fam = dataclasses.replace(
                    fam, span_rule=IFEVAL_SPAN_RULE, spans_optional=True
                )
                gold = _label_only(
                    "yes" if p["checks"][text](response) else "no"
                )
            item = _item(
                source,
                (p["prompt"][:200], variant, text),
                p["context"],
                response,
                fam.question(rng),
                gold,
                variant=variant,
            )
            item.siblings = [
                _requirement(f"{source}_requirement", q).question(rng)
                for q in rng.sample(others, min(2, len(others)))
            ]
            items.append(item)
    return items


# --- unrelated questions -----------------------------------------------------

# Sources whose context has no document, translation source or math problem;
# questions about those things are unrelated to their responses.
_NO_DOCUMENT = frozenset(
    {
        "helpsteer2",
        "ultrafeedback",
        "beavertails",
        "aegis",
        "fava",
        "mushroom",
        "prm800k",
        "processbench",
        "infobench",
        "ifeval",
        "rubrichub",
    }
)
_NOT_TRANSLATION = _NO_DOCUMENT | {"ragtruth", "qa_feedback"}
_NOT_MATH = (_NOT_TRANSLATION - {"prm800k", "processbench"}) | {"mqm"}
_EVERY_SOURCE = _NOT_TRANSLATION | {"mqm"}


def unrelated_pool(
    spare_requirements: list[str],
) -> list[tuple[str, Family, frozenset[str]]]:
    """Questions that are unrelated to the responses of some sources.

    Args:
      spare_requirements: Requirement questions of prompts that are used
        nowhere else.

    Returns:
      (pool name, family, sources it may be asked of) triples.
    """
    pool = [
        ("document", UNSUPPORTED, _NO_DOCUMENT),
        ("document", UNGROUNDED, _NO_DOCUMENT),
        ("translation", TRANSLATION, _NOT_TRANSLATION),
        ("translation", MAJOR_ERROR, _NOT_TRANSLATION),
        ("math", MATH_ERROR, _NOT_MATH),
        ("math", FIRST_MATH_ERROR, _NOT_MATH),
    ]
    pool += [
        ("requirement", _requirement("unrelated_requirement", q), _EVERY_SOURCE)
        for q in spare_requirements
    ]
    return pool


def unrelated(
    item: schema.Item,
    pool: list[tuple[str, Family, frozenset[str]]],
    rng: random.Random,
) -> schema.Item:
    """The same response with a question that does not apply to it.

    A pool (document, translation, math, requirement) is chosen uniformly
    among those allowed for the source, then a question from it.
    """
    allowed = [p for p in pool if item.source in p[2]]
    name = rng.choice(sorted({p[0] for p in allowed}))
    _, fam, _ = rng.choice([p for p in allowed if p[0] == name])
    question = fam.question(rng)
    return item.model_copy(
        update={
            "id": _id(item.id, "unrelated", question.text),
            "question": question,
            "related": False,
            "gold": None,
            "siblings": [],
            "meta": {
                **item.meta,
                "family": question.id.split("/")[0],
                "unrelated_pool": name,
            },
        }
    )
