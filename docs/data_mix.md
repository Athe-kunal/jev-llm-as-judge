# Judge span data: mix and assumptions

This file lists every choice made when building the ~50K-example training
mix (`configs/mix.yaml`, output `data/mix/train.jsonl`). Code:
`src/spanjudge/{datasets,pipeline,schema,llm,prompts}.py`, prompt templates
in `src/spanjudge/prompts/*.j2`.

## What one example is

- Input: `context` (prompt, document, source text or problem), `response`
  (the model output being judged) and a `question` with typed labels
  (`boolean`: yes / no, `score`: 1-5). Every question also allows the label
  `unrelated`.
- Output: `label` and `spans` (exact character ranges of the response that
  the label rests on; empty for `unrelated` and for "no problem" labels).

## Sources

| Source | Judge question families | Label from | Spans from |
|---|---|---|---|
| rubrichub | rubric criterion met? (yes/no) | Qwen | Qwen, tested |
| infobench | decomposed checklist question (yes/no) | Qwen | Qwen, tested |
| ifeval | verifiable instruction followed? (yes/no) | code checker | Qwen, tested (optional) |
| ragtruth | unsupported / conflicting / baseless facts (yes/no) | human | human |
| qa_feedback | irrelevant-or-repeated / ungrounded / any problem (yes/no) | human | human |
| fava | any hallucination + 6 hallucination types (yes/no) | human | human |
| mushroom | hallucination (yes/no), 14 languages | human | human |
| mqm | translation quality (1-5), major error (yes/no) | human | human |
| prm800k | wrong step in a math solution (yes/no) | human | human |
| processbench | first wrong step (yes/no) | human | human |
| helpsteer2 | helpfulness / correctness / coherence (1-5) | human | Qwen, tested |
| ultrafeedback | helpfulness / honesty / instruction following / truthfulness (1-5) | GPT-4 | Qwen, tested |
| beavertails | unsafe + 14 harm categories (yes/no) | human | Qwen, tested |
| aegis | unsafe + 18 harm categories (yes/no) | LLM jury (unsafe), human or jury (safe) | Qwen, tested |

BeaverTails is weight 0 (left out of the export): it is CC BY-NC 4.0.
Aegis 2.0 replaces it at weight 1.5. Notes on Aegis:
- Only rows with a response are used, from all three splits.
- Only 10% of the 5,000 augmented refusals are kept, since they are all trivially safe.
- Category questions are asked as "yes" only for unsafe responses. For a
  safe response the flagged categories describe the prompt.
- Every unsafe label comes from Aegis's LLM jury, not humans, and some are
  noisy (e.g. a harmless answer flagged as criminal planning).

Each family has 1-3 wordings; one is picked at random per item, so the same
task is asked in several ways.

## Licenses (commercial use)

This is an engineering reading of the license tags and repos, not legal
advice. ⚠ means the license allows commercial use, but the responses come
from models whose terms limit using their outputs to train other models.

| Source | License | Commercial | Notes |
|---|---|---|---|
| beavertails | CC BY-NC 4.0 | no | left out; responses from Alpaca-7B (LLaMA-1) |
| aegis | CC BY 4.0 | yes | responses from Mistral-7B (Apache-2.0) |
| ragtruth | MIT | ⚠ | responses from GPT-3.5/4 and Llama-2-chat |
| ultrafeedback | MIT | ⚠ | responses from ~17 models incl. GPT-4 and Llama-2; ratings are GPT-4's |
| fava | CC BY 4.0 (fava-uw/fava-data) | ⚠ | responses from ChatGPT and Llama-2-chat |
| prm800k | MIT | yes | GPT-4 solutions, released by OpenAI under MIT |
| processbench | Apache-2.0 | yes | solutions from Qwen and Llama-3 models |
| mqm | Apache-2.0 | yes | the source sentences are WMT news test sets |
| mushroom | CC BY 4.0 | yes | attribution required |
| qa_feedback | Apache-2.0 | yes | the passages are Wikipedia text (CC BY-SA) |
| helpsteer2 | CC BY 4.0 | yes | |
| rubrichub, infobench, ifeval | Apache-2.0 / MIT / Apache-2.0 | yes | prompts only; responses are Qwen's |
| Qwen3.6-27B (teacher) | Apache-2.0 | yes | |

## Assumptions and decisions

### Mix and balance
1. Target 50,000 items: 80% related, 20% unrelated (`unrelated_share`).
2. Related items are split across the 13 sources in proportion to
   `source_weights` (configs/mix.yaml). A source with fewer candidates than
   its share gives all it has, and the rest is spread over the others by
   weight (water-filling, `_water_fill`). Counts per source are in
   `data/mix/build.json`.
2b. **Source weights.** Rich, clean sources get more and small, one-sided
   or noisier ones get less:

   | weight | sources | why |
   |---|---|---|
   | 2 | RAGTruth, MQM, PRM800K | human span-level labels, large pools, 0/10 errors in the audit |
   | 1.5 | BeaverTails, ProcessBench | clean human labels (ProcessBench is capped by its 3,375 items) |
   | 1 | HelpSteer2, UltraFeedback, RubricHub | distinct task types, but Qwen spans (2-3/10 errors) |
   | 0.5 | QA-Feedback, InFoBench, IFEval, Mu-SHROOM, FAVA | debatable labels (QA-Feedback 4/10), ~92-97% one label, or small with duplicates |

   With 40,000 related items, a weight-1 source gets about 2,760.
2c. **Weights were changed during the run** (after ~32K of the 50K
   equal-share items were judged). Nothing judged was thrown away:
   - `topup` appended 14,907 items to sources below their new share. These are
     RAGTruth, MQM, PRM800K and BeaverTails, plus HelpSteer2 and UltraFeedback,
     whose export rate is only ~61%. A new item never repeats an id or a
     (response, question family) already in `items.jsonl`. RubricHub was not
     topped up, because it would need new generated responses.
   - `judge` stops a source once it has enough records for its share:
     `_quotas` takes the share divided by the fraction of that source's
     records the export keeps, plus 3%. Items over the quota (~4,700, mostly
     QA-Feedback, Mu-SHROOM and InFoBench) stay in `items.jsonl` unjudged.
   - `report` samples each source down to its share, stratified over
     (family, label). Rows cut this way count as `over_share` in
     `left_out`.
3. Within a source, items are drawn round-robin over (question family,
   label) so rare families and labels are kept.
4. Unrelated items per source are in proportion to its related items.

### Labels and spans
5. Human labels and human spans are used unchanged as the final answer.
   The counterfactual tests are still run on them and stored, but human
   spans are never dropped.
6. Sources with a label but no spans (helpsteer2, ultrafeedback,
   beavertails): the dataset label is final. Spans come from Qwen. If
   Qwen's own label differs, the judge is asked again with the dataset
   label given ("The correct label is ..."), and those spans are used.
7. Sources without labels (rubrichub, infobench): Qwen's label is final.
   IFEval labels come from code checkers written after IFEval's own rules
   (`datasets._ifeval`); its sentence-count and response-language
   instructions are dropped (they need a sentence splitter and a language
   detector). IFEval spans are optional: "no commas used" has nothing to
   point at.
8. Qwen spans (from 6 and 7) are kept only if they pass at least one
   counterfactual test (see "Tests"); the others are stored as `dropped`.
9. "No problem" labels (`no` for problem questions, `5` for scores) have no
   spans by definition.
10. Unrelated items: the label is `unrelated` with no spans, true by
    construction. An unrelated *requirement* question is left out of the
    export when Qwen gives p(unrelated) < 0.5, because a requirement can be
    generic enough to apply. Document, translation and math questions are
    kept: they cannot apply, even though Qwen often answers them "no".
10b. A related record whose label needs evidence (not "no problem", not a
    question with optional spans) is left out of the export when no tested
    span survived and the spans are not human. This also removes
    requirements that fail only by omission, which are out of scope.
    Counts per source and reason: `left_out` in report.json.

10c. Also left out of the export: responses that repeat the hidden
    "careless" note (it leaked in ~2% of careless responses; future builds
    skip the careless variant for "repeat the request" prompts); unrelated
    math questions on chat prompts that ask for math (the response is then
    itself a solution); a (response, question) seen with conflicting labels
    (FAVA has some outputs twice with different annotations; for human
    datasets the question family counts as the same question), and exact
    repeats of a (response, question) pair (exported once).
10d. Export-time cleanup that keeps offsets valid: a trailing `<|im_end|>`
    is stripped from responses (Mu-SHROOM), HTML entities and `<br>` in
    contexts are unescaped, and Mu-SHROOM spans are widened to whole words
    (about 20% cut words; the adapter now does this too).

### Unrelated questions
11. The question for an unrelated item comes from a pool that cannot apply
    to the source: document questions (unsupported / ungrounded) for sources
    without a document, translation questions for non-translation sources,
    math-step questions for non-math sources, and rubric requirements from
    RubricHub prompts that are used nowhere else. A pool is chosen uniformly
    among those allowed, so unrelated questions have mixed formats and
    types (yes/no and 1-5), not only rubric questions.

### Per-dataset handling
12. RAGTruth: train and test splits; contexts over 5,000 or outputs over
    2,500 characters skipped; the stray trailing "output:" line of the
    context is removed. Families: all types; conflict types; baseless types.
13. MQM (RicardoRei/wmt-mqm-error-spans, 167K rows): up to 10,000 rows per
    language pair (zh-en, en-de, en-ru). The dataset's span offsets often
    cut the last character of a word (e.g. "Einschaltknop" for
    "Einschaltknopf"); spans are widened to whole words (not for Chinese or
    Japanese characters). Human spans are trimmed of edge whitespace. Translations with
    an annotation that cannot be located are skipped so the 1-5 level
    (from the summed MQM penalty: minor 0.5, major 2, critical 4; 0 -> 5,
    <=1 -> 4, <=2.5 -> 3, <=5 -> 2, else 1) and the spans describe the same
    errors. Target languages: de, en, ru, zh.
14. QA-Feedback (FineGrainedRLHF train_feedback.json): spans refer to
    "prediction 1". Irrelevant + Redundant -> irrelevant; Unverifiable +
    Wrong-Grounding -> ungrounded; all types (Incoherent too) -> any
    problem. "Missing info" feedback has no span, so answers with it get no
    "any problem" question.
15. FAVA (wandb/fava-data-processed, human-annotated outputs): `<mark>`
    text (suggested corrections) is removed; in `<entity>`/`<relation>` the
    error is the `<delete>` text; for contradictory / invented / subjective
    / unverifiable the whole tag content. Outputs whose markup does not
    rebuild the exact output text are skipped (391 of 460 parse). The
    "false or made-up information" question uses only entity, relation,
    contradictory and invented spans (subjective and unverifiable
    statements are not false facts). Each output gets that question, one
    per error type it has, and up to 2 types it does not have. FAVA has no
    reference document, so hallucination is judged against world knowledge.
16. Mu-SHROOM: validation and test splits of all 14 languages, hard labels.
    About 97% of these outputs contain a hallucination; the source stays
    imbalanced.
17. PRM800K: raters stop at the first wrong step, so a wrong solution is
    taken up to and including its first step rated -1. A correct solution
    loses its "# Answer" line and is cut at a random step, so the ending of
    a response does not give the label away. The wrong step is still always
    the last step of a "yes" response (a position bias the student could
    learn). Solutions with an unrated step before any -1 are skipped.
18. ProcessBench: the labeled first wrong step is the only span, even if
    later steps are wrong too (the question asks for the first one).
    ProcessBench is a test benchmark; training on it means it can no
    longer be used to evaluate the student.
19. HelpSteer2: human 0-4 ratings mapped to 1-5. Only helpfulness,
    correctness and coherence; complexity and verbosity are not quality
    judgments. Multi-turn prompts' `<extra_id_1>` markers are rewritten as
    "User:" / "Assistant:".
20. UltraFeedback: ratings are GPT-4 annotations, not human. One random
    completion per prompt; "N/A" ratings skipped.
21. BeaverTails (30k_train): per response one question: unsafe yes/no, a
    harm category it has (yes), or for safe responses a category it does not
    have (no).
22. RubricHub: single-turn English prompts (chat and writing files) with at
    least 4 self-contained criteria; 700 prompts answered by Qwen, the rest
    only supply unrelated requirements.
23. InFoBench and IFEval: each prompt answered by Qwen twice, once normally
    and once with a note to silently ignore one requirement (so "no"
    answers occur). "The/this generated <word>" in InFoBench questions is
    replaced by "the response".
24. Left out: ANAH (sentence-level annotations without the sentence text,
    so spans cannot be aligned reliably); WildGuard (gated on Hugging Face;
    BeaverTails covers safety); e-SNLI and HateXplain (not judge-style).

### Tests (counterfactual span validation)
25. Support for the final label: p(label) for yes/no questions; for 1-5
    questions how far the expected score is below 5, scaled to 0-1 (the
    spans are what lowers the score).
26. Per span, Qwen writes a counterfactual (the answer should change) and a
    paraphrase (it should not). The span's reason (Qwen's or the human
    annotation's) is given to the rewriter so it knows what to change. For
    problem questions, scores and requirements answered "no", the
    paraphrase must keep the shortcoming ("do not fix anything").
27. Rewrites that were cut off by the token limit, or grew beyond
    max(3x the span, span + 120 characters), are rejected. A counterfactual
    that keeps the span and only adds text (fixing an omission) leaves the
    span untested. Qwen spans covering over 90% of a response longer than
    200 characters are dropped (not evidence). Counterfactuals that are
    refusals, empty, or identical to the span are rejected; such a span has no test (a Qwen span is then dropped). A
    paraphrase identical to the span is accepted (a name or a single word
    often has no paraphrase; the control then holds trivially). Qwen refuses
    to rewrite some harmful text (e.g. BeaverTails child-abuse content), so
    those Qwen spans are dropped.
28. necessary: the paraphrase moves support by less than tau (0.05) and the
    counterfactual drops it by more than tau. sufficient: with every other
    span rewritten, support stays more than tau above the all-rewritten
    support. kept = necessary or sufficient.
29. If rewriting all Qwen spans leaves support within tau, the judge is run
    on the rewritten response once more and new spans outside the rewritten
    passages are added (round 2), then everything is tested again.
30. Specificity (rubrichub, infobench, ifeval): 2 other questions of the
    same prompt are scored before and after all spans are rewritten. In the
    pilot their p(yes) was near 0 or 1, so this check has little power.

### Model and decoding
31. One model for everything: Qwen3.6-27B served by vLLM as `judge`,
    greedy decoding, thinking disabled. The judge's reply is prefilled with
    `{"label": "` so the first generated token is the label; its top-20
    log-probabilities give the label distribution (`qwen_dist`).
32. Judge replies that are not valid JSON are parsed again with stray
    backslashes escaped (LaTeX). Items with human spans are never rejected
    for a bad reply: Qwen's label then falls back to the most likely label.
33. Items with a dataset label (with or without spans) are never rejected
    for a bad judge reply; for label-only items the judge is then asked
    again with the label given.
34. Quotes made only of markdown (e.g. "******") and empty matches are
    dropped. A refusal span ("I cannot help with that") may be rewritten as
    another refusal; such rewrites are not rejected. FAVA spans are widened
    to whole words like MQM's (about 20% cut words).
35. Known weak spots to watch in the full run: (a) score sources
    (helpsteer2, ultrafeedback) lose many records to `no_tested_span`,
    because fixing one of several errors barely moves the expected score;
    this skews them toward responses with a single flaw. (b) Some
    coherence spans are really correctness errors. (c) `qwen_dist` in the
    export is Qwen's own distribution; on unrelated rows and on rows where
    the dataset label differs it disagrees with `label`, so it must not be
    used as a soft target for those rows. (d) About 3% of Qwen-only items
    are rejected for broken JSON. (e) In the first 3,000 records many
    rewrites of InFoBench and RubricHub spans failed (113 and 80 spans).
    In a sample of 25, most failures were rewrites that grew far beyond the
    span (Qwen rewrote the whole surrounding section for a heading or a
    single word), then counterfactuals identical to the span (the span
    cannot be negated on its own), then additions; the guards reject them
    as intended, at the cost of some spans. (f) Qwen-written responses
    rarely miss a requirement even with the "careless" note: in the first
    3,000 records InFoBench is 150 yes / 13 no and IFEval 86 yes / 7 no.
    Mu-SHROOM is 127 yes / 4 no (the dataset itself).
36. Quotes are located with 5-6 characters of context; matching tolerates
    whitespace changes and dropped markdown (`*`, `_`, `` ` ``). Quotes that
    are not found or are ambiguous are dropped and counted.
