"""Records for span-judging data, and quote location."""

import re
from typing import Literal

import pydantic

UNRELATED = "unrelated"
UNRELATED_DESC = "The question has nothing to do with this response."


class Question(pydantic.BaseModel):
    """A typed judging query. Every question also allows `unrelated`."""

    id: str
    type: Literal["boolean", "choice", "score"]
    text: str
    # Label id -> description, `unrelated` included. Score labels are "1".."5",
    # higher is better.
    labels: dict[str, str]
    # What the spans are, told to the judge.
    span_rule: str
    # Labels that never have spans besides `unrelated` (e.g. "no
    # hallucination", "perfect translation").
    empty_labels: list[str] = []
    # The answer can be right without any span (e.g. "no commas used").
    spans_optional: bool = False


class SpanTest(pydantic.BaseModel):
    """Counterfactual tests of one span; support = how much the answer holds.

    Support is p(label) for boolean and choice questions, and how far the
    expected score is below 5 (0 to 1) for score questions.
    """

    counter: str  # the span rewritten so the answer should change
    paraphrase: str  # the span reworded, same meaning (control)
    s_counter: float  # support with only this span rewritten
    s_paraphrase: float  # support with only this span paraphrased
    s_others_rewritten: float  # support with every other span rewritten
    control_ok: bool  # the paraphrase moved support by less than tau
    necessary: bool  # rewriting it drops support by more than tau
    sufficient: bool  # it alone holds support up vs all spans rewritten
    kept: bool


class Span(pydantic.BaseModel):
    """A highlighted character range of the response."""

    start: int
    end: int
    text: str
    reason: str = ""
    origin: Literal["human", "qwen", "qwen_given_label", "qwen_round2"] = "qwen"
    test: SpanTest | None = None


class Annotation(pydantic.BaseModel):
    """A label with the spans that justify it."""

    label: str
    spans: list[Span] = []
    # False when a dataset gives a label but no spans (spans come from Qwen).
    spans_annotated: bool = True


class Item(pydantic.BaseModel):
    """One (context, response, question) to judge."""

    id: str
    source: str
    context: str  # what the response answers: prompt, document, source text
    response: str  # the text whose spans are marked
    question: Question
    # False: the question comes from another dataset or prompt, so the right
    # answer is `unrelated` with no spans.
    related: bool
    gold: Annotation | None = None  # human label and spans, when they exist
    # Other questions about the same response, for the specificity check.
    siblings: list[Question] = []
    meta: dict = {}


class Sibling(pydantic.BaseModel):
    """Another question's p(yes) before and after all spans are rewritten."""

    question: str
    p_yes_before: float
    p_yes_after: float


class Record(Item):
    """An item with Qwen's answer, the final spans and their tests."""

    dist: dict[str, float]  # Qwen's label distribution
    qwen: Annotation  # Qwen's label and spans, as located
    raw: str
    unlocated: int
    ambiguous: int
    final: Annotation  # gold when it exists, else Qwen's kept spans
    support: float | None = None  # support for the final label, as is
    s_all_rewritten: float | None = None
    label_all_rewritten: str | None = None
    rounds: int = 0  # judge rounds (2 = missed spans were looked for)
    dropped: list[Span] = []  # Qwen spans that failed every test
    siblings_after: list[Sibling] = []


_MARKUP = "*_`"


def _flex(text: str) -> str:
    """Regex for text that tolerates changed whitespace and dropped markdown.

    Models often quote "**not**" as "not" or a heading without its "**", so
    markdown emphasis characters may appear around and inside words.
    """
    gap = f"[{re.escape(_MARKUP)}]*"
    words = ["".join(c for c in w if c not in _MARKUP) for w in text.split()]
    return r"\s+".join(
        gap + gap.join(re.escape(c) for c in w) + gap for w in words if w
    )


_TAGGED = re.compile(r"(.*?)<quote>(.*?)</quote>(.*)", re.DOTALL)


def locate_tagged(snippet: str, content: str) -> tuple[int, int] | None | bool:
    """Finds `left<quote>q</quote>right` in content; returns q's offsets.

    The snippet is tried with less and less context (both sides, left, right,
    quote alone); the first variant that occurs exactly once wins.

    Returns:
      (start, end) of the quote, None if no variant occurs, or False if the
      most specific variant that occurs matches more than once.
    """
    m = _TAGGED.fullmatch(snippet.strip("\n"))
    left, quote, right = m.groups() if m else ("", snippet, "")
    if not quote.strip(_MARKUP + " \n\t"):
        return None  # empty, or only markup like "******"
    for pre, post in ((left, right), (left, ""), ("", right), ("", "")):
        # Context and quote may be separated by whitespace in the text.
        pattern = rf"({_flex(pre)})\s*({_flex(quote)})\s*({_flex(post)})"
        hits = list(re.finditer(pattern, content))
        if len(hits) == 1:
            start, end = hits[0].span(2)
            while start < end and content[start] in _MARKUP:
                start += 1
            while end > start and content[end - 1] in _MARKUP:
                end -= 1
            return (start, end) if start < end else None
        if len(hits) > 1:
            return False
    return None


def merge(spans: list[Span], content: str) -> list[Span]:
    """Sorts spans and merges the overlapping ones."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if out and s.start < out[-1].end:
            out[-1].end = max(out[-1].end, s.end)
            out[-1].text = content[out[-1].start : out[-1].end]
        else:
            out.append(s.model_copy())
    return out


def locate_quotes(
    quotes: list[tuple[str, str]], content: str, origin: str = "qwen"
) -> tuple[list[Span], int, int]:
    """Turns (snippet, reason) pairs into spans with exact offsets.

    Returns:
      The spans (sorted, overlapping ones merged), the number of snippets not
      found and the number that matched more than once.
    """
    found, unlocated, ambiguous = [], 0, 0
    for snippet, reason in quotes:
        hit = locate_tagged(snippet, content)
        if hit is None:
            unlocated += 1
        elif hit is False:
            ambiguous += 1
        else:
            found.append(
                Span(
                    start=hit[0],
                    end=hit[1],
                    text=content[hit[0] : hit[1]],
                    reason=reason,
                    origin=origin,
                )
            )
    return merge(found, content), unlocated, ambiguous
