# Training data: Reason+Long-12k

All experiments from 2026-10-15 on train on `data/splits/reason_long_12k/`, our
reproduction of the Reason+Long-12k mixture of Li & Wagle, "LLM-as-Jev: LLMs
Are Already Jev-Style Decision Models—When and How to Fine-Tune Them"
(arXiv 2610.02076v3; `2610.02076v3.pdf` at the repository root). The paper has
no code or data release (checked 2026-10-15: arXiv abstract page, GitHub, Hugging
Face), so every source below was assembled from the original dataset releases
and every choice the paper leaves open is listed in Section 4.

Builder: `src/spanjudge/reason_long.py`. Outputs: `train.jsonl`, `eval.jsonl`,
`manifest.json` (all counts below are copied from the manifest).

## 1. The mixture

Reason+Long-12k (paper Table 3, App. G) is the Reason-9k mixture (three intent
datasets and six short reasoning datasets) plus three long-input datasets,
1,000 training examples each: 12,000 examples, 375 optimizer steps at global
batch 32, one epoch, maximum length 10,240 tokens with inputs never truncated,
options shuffled per example, seed 20260922 (Table 2). CommonsenseQA and
HellaSwag (intent set only) and VitaminC, QuAIL and Social IQa (Reason-30k
only) are not part of it.

The paper's prompt (App. A), applied with the chat template,
`add_generation_prompt=True`, `enable_thinking=False`:

```
State: {state}
Options:
[1] {option 1}
[2] {option 2}
...
Select one option. Answer only with its bracketed numeric identifier.
```

followed by the assistant prefix `Best answer: [`; each candidate is the
suffix `k]`. Our rows store the context (`parts`), `question` and options
(`labels`) separately; `train_joint.build_prompt` renders them in this repo's
format (Section 5).

| dataset | task | options | train split used | eval split used | split rows (train / val) | clean pool | train | eval | train rows with spans | license |
|---|---|---|---|---|---|---|---|---|---|---|
| CLINC150 | intent (10 domains), incl. out-of-scope | 151 | train | validation | 15,250 / 3,100 | 15,195 | 1,000 | 500 | 0 | CC BY 3.0 |
| MASSIVE (en-US) | intent (voice assistant) | 60 | train | dev | 11,514 / 2,033 | 11,443 | 1,000 | 500 | 0 | CC BY 4.0 |
| Bitext | intent (e-commerce support) | 27 | all (90% of distinct utterances) | all (seeded 10% of distinct utterances) | 24,220 / 2,652 | 21,664 | 1,000 | 500 | 0 | CDLA-Sharing-1.0 |
| ReClor | argument reasoning | 4 | train | val | 4,638 / 500 | 4,583 | 1,000 | 500 | 0 | research only |
| LogiQA 2.0 | logical reasoning | 4 | train | dev | 12,567 / 1,569 | 11,312 | 1,000 | 500 | 0 | CC BY-NC-SA 4.0 |
| CosmosQA | commonsense reading | 4 | train | valid | 25,262 / 2,985 | 25,129 | 1,000 | 500 | 0 | CC BY 4.0 |
| PIQA | physical commonsense | 2 | train | valid | 16,113 / 1,838 | 15,551 | 1,000 | 500 | 0 | AFL-3.0 |
| αNLI (ART) | abductive reasoning | 2 | train | dev | 169,654 / 1,532 | 165,371 | 1,000 | 500 | 0 | not stated |
| ProofWriter | rule verification | 3 | OWA/depth-5 train | OWA/depth-5 dev | 69,810 / 10,190 | 69,810 | 1,000 | 500 | 667 | not stated |
| ContractNLI | contract inference (long) | 3 | train | dev | 7,191 / 1,037 | 7,156 | 1,000 | 493 | 623 | CC BY 4.0 |
| MuSiQue | multi-hop answerability (long) | 2 | full_v1.0 train | full_v1.0 dev | 39,876 / 4,834 | 31,724 | 1,000 | 500 | 514 | CC BY 4.0 |
| QuALITY | long-document reading | 4 | v1.0.1 htmlstripped train | v1.0.1 htmlstripped dev | 2,523 / 2,086 | 2,489 | 1,000 | 498 | 0 | CC BY 4.0 |
| **Total** | | | | | | | **12,000** | **5,991** | 1,804 | |

Options: number of answer options per row (paper Table 3). Licenses are the
paper's (Table 3); ReClor is for non-commercial research only, and the αNLI and
ProofWriter releases state no license.

Sources and revisions (the sha256 of every fetched file is in
`data/external/<dataset>/SOURCES.json`):

| dataset | source | revision |
|---|---|---|
| CLINC150 | HF `clinc/clinc_oos`, config `plus` | `155b9c71` |
| MASSIVE | `amazon-massive-dataset-1.1.tar.gz` (Amazon S3), `1.1/data/en-US.jsonl` | release 1.1 |
| Bitext | HF `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, `..._27K_responses-v11.csv` | `430d1a89` |
| ReClor | HF `metaeval/reclor` (mirror of the official form-gated zip) | `0414b39b` |
| LogiQA 2.0 | GitHub `csitfun/LogiQA2.0`, `logiqa/DATA/LOGIQA/{train,dev}.txt` | `955e1d3d` |
| CosmosQA | GitHub `wilburOne/cosmosqa`, `data/{train,valid}.csv` | `b6eb99cc` |
| PIQA | `yonatanbisk.com/piqa/data/{train,valid}{.jsonl,-labels.lst}` | unversioned (sha256 recorded) |
| αNLI | AI2 `alphanli-train-dev.zip` (storage.googleapis.com/ai2-mosaic) | unversioned (sha256 recorded) |
| ProofWriter | AI2 `proofwriter-dataset-V2020.12.3.zip`, `OWA/depth-5/meta-{train,dev}.jsonl` | V2020.12.3 |
| ContractNLI | Stanford NLP `contract-nli` release (`data/external/contractnli`, loaded by `human.contractnli`) | pre-existing download (`cnli.zip`, unversioned) |
| MuSiQue | HF `bdsaglam/musique` (mirror of the official Google Drive zip), `musique_full_v1.0_{train,dev}.jsonl` | `22873a40` |
| QuALITY | GitHub `nyu-mll/quality`, `data/v1.0.1/QuALITY.v1.0.1.zip`, `htmlstripped` files | `f84977c4` |

## 2. Per-dataset conversion

Common format (`human._row`): `{id, source, subset, group, parts:[{name,text}],
question, labels:{answer_id: description}, label, spans:[{part,start,end,text}],
needs_spans}`. Labels are the datasets' own gold answers, never changed and
never produced by a model. For multiple-choice datasets the answer ids are the
1-based option positions in the release (`"1"`..`"k"`) and the descriptions are
the option texts verbatim; `train_joint.encode` shuffles the options per
example. `group` keeps rows that share a source context together (used to keep
train and eval disjoint). `subset` is the gold label (ContractNLI: also the
label; the eval draw round-robins over it).

| dataset | parts | question | labels (gold) | evidence spans |
|---|---|---|---|---|
| CLINC150 | `Message`: utterance | "Which intent does the user's message express?" | 151 intent names incl. `oos`, verbatim (`intent` field) | none |
| MASSIVE | `Message`: `utt` | same as CLINC150 | 60 intent names (`intent` field) | none |
| Bitext | `Message`: `instruction` (with its `{{Order Number}}`-style placeholders as released) | same as CLINC150 | 27 intent names (`intent` field) | none |
| ReClor | `Passage`: `context` | `question` | 4 `answers`; gold `label` | none |
| LogiQA 2.0 | `Passage`: `text` | `question` | 4 `options`; gold `answer` | none |
| CosmosQA | `Context`: `context` | `question` | `answer0..3`; gold `label` | none |
| PIQA | `Goal`: `goal` | "Which solution better achieves the goal?" | `sol1`, `sol2`; gold from `*-labels.lst` | none |
| αNLI | `Observation 1`: `obs1`; `Observation 2`: `obs2` | "Which hypothesis best explains what happened between the two observations?" | `hyp1`, `hyp2`; gold from `*-labels.lst` | none |
| ProofWriter | `Theory`: `theory` | `Statement: "<question>"` / "Based only on the theory, is the statement true, false, or unknown?" | `true` / `false` / `unknown` with fixed descriptions; gold `answer` | true/false: each fact and rule of the first proof in `proofsWithIntermediates` (the statement's proof, or its negation's for false), as sentence spans in the theory; unknown: none |
| ContractNLI | `Contract`: document text | `Statement: "<hypothesis>"` / "Does the contract entail the statement, contradict it, or not mention it?" | `entailed` / `contradicted` / `not_mentioned` (`human.contractnli`) | entailed/contradicted: the human evidence spans of the release; not_mentioned: none |
| MuSiQue | `Paragraphs`: the 20 paragraphs, each `title\nparagraph_text`, separated by a blank line, in release order | `Question: <question>` / "Do the paragraphs contain enough information to answer the question?" | `yes` / `no` (the release's `answerable` flag) | yes: each `is_supporting` paragraph (title line and text); no: none |
| QuALITY | `Article`: HTML-stripped `article` | `question` | 4 `options`; gold `gold_label` (annotator majority); questions without one skipped | none |

Rows with spans have `needs_spans = true`. Every span satisfies
`span.text == parts[span.part][start:end]` (checked). ContractNLI rows are the
existing `human.contractnli` rows (unchanged; the paper's formulation, contract
+ hypothesis → three classes, is the same), with `subset` set to the label.

## 3. Cleaning and decontamination

### What the paper says

§3.5: "We curate training sets exclusively from publicly accessible,
human-annotated datasets, converting each source into our unified decision
template without altering ground-truth labels or soliciting synthetic labels
from teacher models. To prevent train-test contamination, we filter all
training instances that share a 13-word n-gram (or an identical sentence of six
or more words) with any evaluation benchmark, and purge exact duplicates of
validation inputs."

App. D: "In addition to the overlap checks in §3.5, we remove training examples
whose inputs also appear in an official validation split."

App. G (intent set): "Following cleaning and deduplication, this intent set
comprises 95,461 instances".

App. E reports the outcome of the checks: "No 13-word window is shared between
the 9,000 training examples and the 7,311 External and JevBench items. Before
sampling, the full training sets contained one shared passage: an argument in
LogiQA 2.0 also appears in an MMLU item and its MMLU-Pro counterpart. Our
overlap filter removed it from the sample." "The intent datasets share some
label names with Banking77 and JevBench intent items (e.g., `cancel_order`),
but no user utterances." "General suite. The same check finds no overlap for
the intent set, Reason-9k, Reason-21k, or Reason-30k. Reason+Long-12k has two
incidental overlaps from MuSiQue's Wikipedia paragraphs: one with an IFEval
prompt, where evaluation concerns only the output format, and one with a
WikiText article that shares a paragraph with a MuSiQue example." App. B
(Behavior): "Two Dolly prompts share a 13-word window with MuSiQue training
paragraphs." App. B (Image): "No 13-word window is shared between the image
questions and any training set." App. E also reports TF-IDF cosine similarity
between the mixtures and each benchmark (Table 6); it is an analysis, not a
filter, and we do not reproduce it.

Evaluation benchmarks (§4.1, App. B, Table 4): External — BoolQ (500),
MMLU (1,200), MMLU-Pro (800), ARC-Challenge (600), WinoGrande (500), SciQ
(400), Banking77 test (3,080); JevBench — 231 public items; General — GSM8K,
IFEval, TriviaQA, LAMBADA, WikiText-2; Image — MMBench-EN v1.1 dev, MMStar;
Behavior — Dolly-15k, MT-Bench.

### What we implemented

Applied to each dataset's training pool, in this order, before sampling
(each row is counted at the first step that drops it):

1. **Exact duplicates within the train split.** Rows whose input (all parts,
   question, and the set of options) is identical after normalization are
   collapsed to one; if the copies disagree on the gold label all of them are
   dropped.
2. **Inputs that appear in the official validation split** (App. D; "purge
   exact duplicates of validation inputs"): same normalized input key.
3. **Shared context with the validation split**: training rows whose group
   (passage, document, story, theory, MuSiQue id, utterance) also occurs in
   the validation split. This keeps `eval.jsonl` disjoint from train by group;
   it is our addition.
4. **Benchmark overlap** (§3.5): a row is dropped if any of its source texts
   shares a 13-word n-gram, or an identical sentence of at least six words,
   with any item of any benchmark below.
5. **Length**: rows whose prompt exceeds 10,240 tokens (Section 6) are left
   out, never truncated.

Benchmark texts checked — the whole evaluation split each fixed sample is drawn
from (the paper's samples are not released, so we cover every item they could
contain):

| suite | benchmark | split / fields | items |
|---|---|---|---|
| External | BoolQ | validation; question, passage | 3,270 |
| External | MMLU | `all` test + validation + dev; question, choices | 15,858 |
| External | MMLU-Pro | test + validation; question, options, CoT content | 12,102 |
| External | ARC-Challenge | test; question, choices | 1,172 |
| External | WinoGrande | `winogrande_xl` validation; sentence, options | 1,267 |
| External | SciQ | test; question, support, answers | 1,000 |
| External | Banking77 | test; text | 3,080 |
| JevBench | JevBench public | easy + original + hard; state, question, labels | 231 |
| General | GSM8K | `main` test; question, answer | 1,319 |
| General | IFEval | prompts | 541 |
| General | TriviaQA | `rc.nocontext` validation; question, answer | 17,944 |
| General | LAMBADA | OpenAI test; text | 5,153 |
| General | WikiText-2 | raw test, one item per article | 62 |
| Image | MMBench | `lmms-lab/MMBench` en dev; question, hint, options | 4,329 |
| Image | MMStar | question | 1,500 |
| Behavior | Dolly-15k | instruction, context (the prompt; responses not checked) | 15,011 |
| Behavior | MT-Bench | prompts (both turns) | 80 |

Hugging Face ids and revisions are in `manifest.json` (`benchmarks`) and
`data/external/benchmarks/<name>/SOURCES.json`: `google/boolq`, `cais/mmlu`,
`TIGER-Lab/MMLU-Pro`, `allenai/ai2_arc`, `allenai/winogrande`, `allenai/sciq`,
PolyAI `task-specific-datasets` (Banking77 `test.csv`), `openai/gsm8k`,
`google/IFEval`, `mandarjoshi/trivia_qa`, `EleutherAI/lambada_openai`,
`Salesforce/wikitext`, `lmms-lab/MMBench`, `Lin-Chen/MMStar`,
`databricks/databricks-dolly-15k`, `HuggingFaceH4/mt_bench_prompts`.

Nothing is missing: all 17 benchmarks named by the paper, including the 231
public JevBench items (v1.4.2.2, `fstandhartinger/jevbench` commit
`fcc69279`, `datasets/public/{easy,original,hard}.jsonl`), were downloaded and
checked (the version string is the one recorded at download; JevBench items
are checked whole: state, question and labels). MMBench and MMStar are checked
on their text only (MMBench's question, hint and options; MMStar's question);
images are not compared. The paper uses VLMEvalKit's MMBench-EN v1.1 dev
(1,292 questions); we check `lmms-lab/MMBench` en dev (4,329 rows), the
closest text release we have, so v1.1-only questions may be missed.

Source texts checked per row (`_units` in the builder): the release's own
fields — utterance; passage/context/article, question and options; goal and
solutions (PIQA); observations and hypotheses (αNLI); theory and statement
(ProofWriter); contract and hypothesis (ContractNLI); each MuSiQue paragraph on
its own and the question. Our fixed question wordings and label descriptions
are not source data and are not checked; the intent question "Which intent
does the user's message express?" is itself an identical sentence of a
JevBench item and would otherwise remove every intent row.

Normalization: Unicode NFKC, case-folded, curly quotes straightened; words are
maximal runs of letters, digits and underscore (all punctuation and whitespace
separate words). N-grams are 13 consecutive such words within one text.
Sentences: the text split after `.`, `!` or `?` followed by whitespace, and at
every line break; a sentence counts if it has at least six words and its
normalized word sequence equals a benchmark sentence's.

Drop counts (training pools; manifest `datasets.<name>`):

| dataset | train split | duplicate copies | conflicting duplicates | input in validation | group in validation | benchmark overlap (n-gram / sentence only) | too long | clean pool |
|---|---|---|---|---|---|---|---|---|
| CLINC150 | 15,250 | 28 | 0 | 16 | 0 | 11 (0 / 11) | 0 | 15,195 |
| MASSIVE (en-US) | 11,514 | 43 | 8 | 11 | 0 | 9 (0 / 9) | 0 | 11,443 |
| Bitext | 24,220 | 2,555 | 0 | 0 | 0 | 1 (0 / 1) | 0 | 21,664 |
| ReClor | 4,638 | 4 | 6 | 17 | 28 | 0 (0 / 0) | 0 | 4,583 |
| LogiQA 2.0 | 12,567 | 888 | 27 | 212 | 89 | 39 (4 / 35) | 0 | 11,312 |
| CosmosQA | 25,262 | 15 | 86 | 0 | 21 | 11 (0 / 11) | 0 | 25,129 |
| PIQA | 16,113 | 5 | 3 | 2 | 549 | 3 (0 / 3) | 0 | 15,551 |
| αNLI (ART) | 169,654 | 1,395 | 2,883 | 0 | 0 | 5 (0 / 5) | 0 | 165,371 |
| ProofWriter | 69,810 | 0 | 0 | 0 | 0 | 0 (0 / 0) | 0 | 69,810 |
| ContractNLI | 7,191 | 16 | 2 | 0 | 0 | 0 (0 / 0) | 17 | 7,156 |
| MuSiQue | 39,876 | 0 | 0 | 0 | 0 | 8,152 (7,818 / 334) | 0 | 31,724 |
| QuALITY | 2,523 | 0 | 0 | 0 | 0 | 34 (0 / 34) | 0 | 2,489 |
| **Total** | 398,618 | 4,949 | 3,015 | 258 | 687 | 8,265 | 17 | 381,427 |

Benchmark-overlap drops by benchmark (a row can count for several):

| dataset | boolq | mmlu | mmlu_pro | arc_challenge | winogrande | sciq | banking77 | jevbench | gsm8k | ifeval | triviaqa | lambada | wikitext2 | mmbench | mmstar | dolly | mt_bench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLINC150 | · | · | · | · | · | · | 3 | 1 | · | · | 1 | 3 | · | · | · | 3 | · |
| MASSIVE (en-US) | · | · | 1 | · | · | · | · | 1 | · | · | · | · | · | 1 | · | 7 | · |
| Bitext | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · |
| ReClor | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| LogiQA 2.0 | · | 38 | 34 | 4 | · | · | · | · | · | · | · | · | · | 4 | 4 | 1 | · |
| CosmosQA | · | 5 | 4 | 2 | · | · | · | · | · | · | · | 5 | · | 3 | 2 | · | · |
| PIQA | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · |
| αNLI (ART) | · | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · | · | · |
| ProofWriter | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| ContractNLI | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| MuSiQue | 3758 | 39 | 5 | · | · | 5 | · | · | · | 155 | 221 | · | 50 | · | · | 5023 | · |
| QuALITY | · | · | · | · | · | · | · | · | · | · | · | 34 | · | · | · | · | · |
| **Total** | 3758 | 82 | 44 | 6 | 0 | 5 | 4 | 2 | 0 | 155 | 222 | 47 | 50 | 8 | 6 | 5037 | 0 |

What the overlaps are. Almost all non-MuSiQue drops come from the
identical-sentence rule matching short generic sentences, e.g. LogiQA 2.0 and
CosmosQA question stems "Which of the following is true / statements is true /
must be true" (MMLU, MMLU-Pro, ARC, MMBench, MMStar), utterances such as "what is
the capital of france" or "how do you jump start a car" (Dolly), "I would like to
make a transfer" (Banking77), "when will my order be delivered" and "turn off
the lights in the bedroom" (JevBench), and fiction sentences such as "I don't
know what to think" or "There was nothing else to do" (LAMBADA; 34 QuALITY
questions over a few articles). The 4 LogiQA 2.0 13-gram drops include the
passage the paper reports (shared with MMLU and MMLU-Pro). MuSiQue drops are
13-gram matches between its Wikipedia paragraphs and Wikipedia text in the
benchmarks, much of it templated article text (e.g. US-census sentences
"…had someone living alone who was 65 years of age or older. The average
household size was…" in Dolly contexts, and BoolQ passages); a row is dropped if
any of its 20 paragraphs matches. The paper reports only the two MuSiQue
overlaps (IFEval, WikiText-2) and two Dolly prompts for its 1,000 sampled rows;
our pools are checked against whole benchmark splits, so many more rows match
before sampling. MuSiQue still has 31,724 clean rows (16,098 no / 15,626 yes).

The eval rows are drawn from the deduplicated validation split; they are not
filtered against the benchmarks (they are never trained on).

## 4. What the paper leaves unspecified, and our defaults

| question | paper | our default |
|---|---|---|
| Source split per dataset | not stated | train rows from each official train split; eval rows from the official validation/dev split; labelled test splits unused. Bitext has a single file: a seeded 10% of distinct normalized utterances is held out as its validation side. |
| CLINC150 variant | "151 options", "including an explicit out-of-scope category" | `plus` config (150 intents × 100 + 250 `oos` in train). `small`/`imbalanced` would give the same label set. |
| MASSIVE version | "en-US", 60 intents | release 1.1, `partition` train / dev. |
| Bitext file | HF id only | the 27K-responses v11 CSV (the repository's only data file); `instruction` → `intent`; responses unused. |
| LogiQA 2.0 version | "LogiQA 2.0" | the English MRC files `logiqa/DATA/LOGIQA/{train,dev}.txt` (not the NLI version). |
| ProofWriter config / depth | "rule verification", 3 options, "balanced true, false, and unknown answers" | OWA (open-world; CWA has no unknown), `depth-5` (questions of depth 0–5); label-balanced sample (334 / 333 / 333). |
| MuSiQue answerability | "deciding whether 20 paragraphs contain enough information" | MuSiQue-Full v1.0, which pairs each answerable question with an unanswerable contrast copy (same id); `answerable` → yes/no. Both copies share a group. Uniform sample from the clean pool. |
| QuALITY gold / text | not stated | `gold_label` (majority of validators), HTML-stripped article. |
| CosmosQA 16% upsampling | "we upsample these to 16%" | 160 of the 1,000 rows are drawn (without replacement) from rows whose gold option is "None of the above choices .", 840 from the rest; the natural share in the clean pool is 5.6% (1,397 of 25,129). |
| Sampling within other datasets | not stated | uniform random 1,000 from the clean pool (natural label distribution), seed `20260922:<dataset>`. |
| Question wording | "unified decision template" with a `{state}` | the wordings in Section 2 (ours). |
| N-gram normalization, sentence segmentation | not stated | Section 3. |
| Duplicates with conflicting labels | not stated | all copies dropped. |
| Benchmark scope of the filter | §3.5 "any evaluation benchmark"; App. E says Reason+Long-12k kept two MuSiQue/General overlaps | every benchmark of all five suites, General included (stricter than the paper's released data; see Section 7). |
| Eval split | not stated (paper evaluates out of domain) | up to 500 rows per dataset (a default; `--eval-rows`) from the deduplicated validation split, whole groups, round-robin over labels, prompt ≤ 10,240 tokens. |

## 5. Differences from the paper

- **Answer codes.** The paper lists options as `[1]`, `[2]`, … and scores the
  suffix `k]` after `Best answer: [` with a tree-factorized listwise loss over
  the candidate trie. Our trainer (`train_joint.py`) renders
  `Question: …\nChoices:\nA. …` and scores one letter token A–Z after its
  answer prefix. It supports at most 26 options.
- **More than 26 options.** CLINC150 (151 options) and MASSIVE (60) exceed
  26: `train_joint.encode` fails on them (`ValueError: zip() argument 2 is longer than argument 1` in `question_block`, because `choice_letters` returns at most 26 codes). Bitext (27) also
  exceeds 26. The rows are written as released (all intent names as options);
  `train_joint` is not changed. See Section 7.
- **No KL anchors, no tree loss.** The paper's objective adds three KL anchors
  to the frozen base model (Eq. 3, λ = 1); our joint trainer has a choice loss
  and an evidence-span head instead.
- **Prompt layout.** Rows are rendered with `ORDER = "question_first"`
  (question and choices before the context) and this repo's system prompt; the
  paper puts the state first and the options after it. Token counts in
  Section 6 are for our rendering.
- **Evidence spans.** The paper trains no span supervision. We keep native
  human/structural evidence where the release has it (ContractNLI, MuSiQue,
  ProofWriter); all other rows have empty spans.
- **Benchmark filter scope.** We drop the General-suite overlaps the paper
  kept (Section 3).
- **In-domain eval.** `eval.jsonl` is ours; the paper reports only external
  benchmarks and JevBench.

## 6. Length statistics

Prompt tokens with the `Qwen/Qwen3.5-2B` tokenizer at
`train_joint.REVISION`, rendered by `train_joint.build_prompt` with
`ORDER = "question_first"` (for the >26-option datasets, codes past Z are
rendered as two letters for counting only).

| dataset | train median | train p95 | train max | eval median | eval p95 | eval max | clean-pool max | candidates over 10,240 |
|---|---|---|---|---|---|---|---|---|
| CLINC150 | 896 | 904 | 916 | 896 | 905 | 919 | 928 | 0 (val: 0) |
| MASSIVE (en-US) | 462 | 469 | 492 | 461 | 469 | 479 | 492 | 0 (val: 0) |
| Bitext | 256 | 261 | 270 | 256 | 261 | 268 | 270 | 0 (val: 0) |
| ReClor | 284 | 369 | 463 | 287 | 379 | 637 | 515 | 0 (val: 0) |
| LogiQA 2.0 | 279 | 368 | 567 | 279 | 362 | 582 | 616 | 0 (val: 0) |
| CosmosQA | 217 | 280 | 349 | 228 | 284 | 318 | 353 | 0 (val: 0) |
| PIQA | 144 | 245 | 474 | 144 | 249 | 398 | 1,183 | 0 (val: 0) |
| αNLI (ART) | 159 | 173 | 197 | 159 | 174 | 196 | 214 | 0 (val: 0) |
| ProofWriter | 287 | 371 | 408 | 272 | 375 | 389 | 411 | 0 (val: 0) |
| ContractNLI | 2,076 | 4,490 | 8,566 | 2,643 | 5,967 | 6,181 | 8,566 | 17 (val: 0) |
| MuSiQue | 2,467 | 3,357 | 5,866 | 2,581 | 3,513 | 4,952 | 7,706 | 0 (val: 0) |
| QuALITY | 6,865 | 8,133 | 8,900 | 6,741 | 8,279 | 8,603 | 8,900 | 0 (val: 0) |

The paper's medians for the long datasets — ContractNLI 2.1k, MuSiQue 2.4k,
QuALITY 6.8k tokens (Table 3) — match ours (2,076, 2,467 and 6,865), although
the paper measures its own prompt and tokenizer (Qwen3.5-4B). 17 ContractNLI
training candidates (out of 7,173 after the earlier steps) exceed 10,240 tokens
under question_first (the same 17 under doc_first) and are left out; no
validation row and no candidate of any other dataset exceeds the limit. CLINC150
prompts are long for single utterances (~900 tokens) because all 151 intent
names are listed as options.

## 7. Open decisions for the user

1. **More than 26 options (CLINC150 151, MASSIVE 60, Bitext 27).**
   `train_joint` cannot encode these rows (3,000 of 12,000 training rows, 1,500
   eval rows). Options: (a) extend the codes in `train_joint` (two-letter codes,
   or the paper's bracketed numbers `[k]` with a multi-token/tree loss);
   (b) present a subset of candidate intents per row (e.g. gold + 25 sampled
   negatives), which changes the task relative to the paper; (c) drop the intent
   datasets (no longer Reason+Long-12k). The data is written as released (all
   options), so any of these can be applied at load time.
2. **Evidence spans for the nine datasets without native evidence** (10,196 of
   12,000 training rows have empty spans, including the unknown / not
   mentioned / unanswerable rows of the three span datasets; 1,804 have native
   spans: ProofWriter 667, ContractNLI 623, MuSiQue 514). Options: train the span head only on the
   rows with spans (`needs_spans`), or annotate the rest with our synthetic span
   pipeline (`synth.py`; not run here). For intent and short-commonsense rows
   (one utterance, PIQA goal, αNLI observations) span evidence may be of little
   use.
3. **Benchmark overlap scope.** We filter against every benchmark of all five
   suites over whole evaluation splits, as §3.5 states. This removes 8,152
   MuSiQue rows (mostly templated Wikipedia text shared with Dolly contexts and
   BoolQ passages), and the paper kept two General-suite MuSiQue overlaps in its
   data. Alternatives: only the External + JevBench suites (the check App. E
   reports for the reasoning data), or an exemption for templated boilerplate.
   The sentence rule also drops rows for generic question stems ("Which of the
   following is true"); the paper does not say whether it did.
4. **Bitext validation side.** Bitext has no official split; we hold out a
   seeded 10% of distinct utterances. Bitext utterances are template variants of
   each other, so the held-out side is near-duplicate of training text.
5. **Sampling within datasets.** Uniform from the clean pool except CosmosQA
   (16% "none of the above", paper) and ProofWriter (label-balanced, paper).
   The intent datasets are therefore not intent-balanced (MASSIVE covers 57 of
   60 intents in train; CLINC150 has 1–14 rows per intent).
6. **ProofWriter config.** OWA depth-5 (questions of depth 0–5). The paper does
   not say which depth subset.
7. **MuSiQue construction.** MuSiQue-Full answerable/unanswerable pairs; the
   label is the release's `answerable` flag. Spans mark all supporting
   paragraphs of answerable rows; unanswerable rows have none.
8. **Eval size.** 500 rows per dataset by default (ContractNLI 493 and QuALITY
   498, because eval draws whole documents).

## 8. How to rebuild

```bash
# 1. fetch the pinned sources (skips files already present)
PYTHONPATH=src .venv/bin/python -m spanjudge.reason_long --download
# ContractNLI is read from data/external/contractnli/contract-nli (existing download)

# 2. build data/splits/reason_long_12k/{train,eval}.jsonl and manifest.json
PYTHONPATH=src .venv/bin/python -m spanjudge.reason_long
# options: --eval-rows 500 --max-tokens 10240 --datasets <subset> --out <dir>
```

The build is deterministic (seed 20260922; per-dataset streams
`20260922:<dataset>`) and runs on CPU (about 10 minutes on CPU, mostly tokenization; no GPU, no model inference).
