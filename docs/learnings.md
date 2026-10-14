# Span judge: findings and ablations

A joint judge (Qwen3.5-2B + LoRA) reads a document and a question, picks an
answer from lettered choices (Nimble-style one-letter codes), and marks the
evidence for it in the document with a cross-attention span head. Synthetic
evidence spans come from Qwen3.6-27B, which is given the gold answer.

Unless stated otherwise: span F1 is token-level at a 0.5 threshold; "human
gold" is a test set of 20 rows with human spans from each of 10 human
datasets (200 rows); "Nimble" is 100 Nimble train rows scored against
Nimble's verified evidence.

## 1. Training the joint model

### 1.1 Prompt order: question before document

500-step head-to-head, same data and settings.

| order | accuracy | span F1 | FEVER | WANLI | HotpotQA | Mu-SHROOM | QA-Feedback |
|---|---|---|---|---|---|---|---|
| document first (Nimble) | 0.75 | 0.24 | 0.11 | 0.18 | 0.01 | 0.51 | 0.07 |
| question first | 0.74 | 0.27 | 0.49 | 0.57 | 0.36 | 0.00 | 0.00 |

With the document first, document tokens cannot see the question, and the
span head must recover the link through one cross-attention layer. Putting
the question first gives large span gains on claim and multi-hop tasks at
the same accuracy (at the cost of Nimble's KV-cache reuse across questions).

### 1.2 Optimisation stability

At LoRA lr 1e-4 / head lr 1e-3 the first full run diverged at step ~1,550
(gradient norm ~3.5e12, then NaN). No single bad row caused it: the weights
drifted at peak learning rate. With LoRA lr 3e-5, head lr 3e-4 and skipping
updates whose gradient norm is non-finite or > 50, a 3-epoch run (7,797
steps) never skipped a step (norms 3-8).

| run | step 1,000 quick-eval span F1 | accuracy |
|---|---|---|
| lr 1e-4 / 1e-3 (diverged later) | 0.20 | 0.81 |
| lr 3e-5 / 3e-4 | 0.51 | 0.79 |

The lower learning rate learned spans 2.5x faster in this window.

### 1.3 Epochs

| full eval (18,518 rows) | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| accuracy | 0.74 | **0.78** | 0.77 |
| span F1 | 0.62 | **0.64** | 0.64 |

Two epochs are enough.

### 1.4 Results per dataset (epoch 2)

| dataset | accuracy | macro F1 | span F1 |
|---|---|---|---|
| customer_control | 1.000 | 1.000 | 0.76 |
| citation_control | 0.996 | 0.996 | 0.33 |
| fever | 0.947 | 0.944 | 0.69 |
| aegis | 0.911 | 0.908 | 0.79 |
| hotpotqa | 0.906 | 0.906 | 0.89 |
| contractnli | 0.892 | 0.866 | 0.63 |
| qa_feedback | 0.834 | 0.814 | 0.61 |
| wanli | 0.764 | 0.761 | 0.63 |
| processbench | 0.768 | 0.714 | 0.07 |
| prm800k | 0.860 | 0.736 | 0.44 |
| ragtruth | 0.685 | 0.684 | 0.45 |
| mushroom | 0.971 | 0.674 | 0.65 |
| hatexplain | 0.630 | 0.618 | 0.56 |
| mqm | 0.373 | 0.270 | 0.32 |
| **mean** | **0.824** | **0.778** | |

Accuracy hides majority-class guessing: Mu-SHROOM 0.97 accuracy but 4 of 15
clean responses recognised; HateXplain "normal" recall 0.35; MQM
major-error flags 72% of clean translations. Report macro F1.

### 1.5 Training label mix causes a "problem found" bias

To keep span supervision dense, at most 20% of each human dataset's
training rows had no spans. In problem-detection datasets the "no problem"
rows are exactly the span-free ones, so training saw 20% "no" while the
natural (eval) rate is much higher:

| dataset | "no problem" in train | in eval |
|---|---|---|
| RAGTruth | 20% | 66% |
| MQM major-error | 20% | 72% |
| QA-Feedback | 20% | 39% |
| ProcessBench | 20% | 35% |
| ContractNLI "not mentioned" | 20% | 38% |
| HateXplain "normal" | 14% | 33% |
| MQM 1-5, level 5 | 20% | 43% |

This matches the per-class errors in 1.4. Datasets whose natural mix is
already mostly "yes" (Mu-SHROOM, PRM800K) are unaffected. Lesson: capping
span-free rows changes the label prior; balance labels separately from span
density.

### 1.6 MQM 1-5 quality levels

The level is computed from human error spans (penalty 0.5 minor / 2 major /
4 critical; 5 = no error, 4 = <= 1, 3 = <= 2.5, 2 = <= 5, 1 above). The
model learned the order (mean predicted level 3.03 / 3.21 / 3.45 / 3.62 /
3.78 for gold 1-5; 82% within one level) but compresses it into 3-4: 29 of
434 gold "5" predicted 5. Two causes: too few 5s in training (1.5), and
thresholds the model never sees.

## 2. Nimble as an evaluation of evidence

Nimble (3,000 rows; base / counterfactual pairs whose answers differ) stores
verified evidence per row:

- **focus evidence**: exactly two sentences, by construction, whose change
  flips the answer;
- **policy evidence**: verbatim rule / rubric quotes the answer applies
  (0-6 per row, often a whole paragraph).

Neither is exhaustive: focus covers only the fact that was varied.

### 2.1 How Nimble verifies evidence

The case is decomposed into atomic facts and explicit rules from fact
states to the answer. One focus fact is written as two sentences that only
establish it together. Variants are built (base, one sentence negated, each
sentence removed, each alone, the pair alone) and a separate checker that
sees only the text (no rules, answers or other variants) labels the fact
supported / refuted / unknown. Required: base supported, negated refuted,
either sentence removed unknown, each alone unknown, the pair alone
supported. The answer is computed from the rules, so base and counterfactual
answers differ by construction.

### 2.2 A judge trained without Nimble

`joint_v2` on nimble-eval: accuracy 0.691 (true/false 0.78, choice 0.67,
score 0.58). Only 39.5% of base / counterfactual pairs get both answers
right; in 50% of pairs the model gives the same answer to both versions,
i.e. it ignores the flipped fact. Span recall on Nimble's long multi-section
documents is 0.02.

## 3. Synthetic evidence annotation

Pipeline: Qwen3.6-27B, given the gold answer, writes evidence quotes; 8
samples at temperature 0.7; quotes located and snapped to word boundaries;
characters marked by >= k samples kept; vote counts kept as soft targets.

### 3.1 Prompt ablation (5 of 8 votes)

| prompt | human-gold F1 | Nimble focus covered |
|---|---|---|
| original | 0.54 | 0.42 |
| general, descriptive + a "checks" field (conditions first, in the JSON) | 0.48 | 0.39 |
| **general, descriptive** | **0.54** | **0.64** (0.52 at equal highlight size) |

The descriptive prompt explains what evidence is for each kind of question
(claim checks, problem detection, categories, scales), prefers specific
facts to summaries, asks to follow references, and to mark a stated rule
with the facts it applies to. Writing the reasoning as a structured field
hurt. On the full Nimble set it raised focus coverage 0.42 -> 0.60 but also
the highlighted share 0.31 -> 0.40, at unchanged precision: coverage must
always be read with highlight size.

### 3.2 Agreement with Nimble's evidence (descriptive prompt, 5 of 8)

| reference | precision | recall | F1 |
|---|---|---|---|
| focus only | 0.30 | 0.56-0.62 | 0.40 |
| focus + policy | 0.54 | 0.47-0.49 | 0.50-0.52 |

Precision is flat (~0.54) at every vote threshold: text that all 8 samples
mark is no more often correct than text 2 mark.

Why focus sentences are missed (train / eval):

| outcome | train | eval |
|---|---|---|
| covered (>= half the sentence) | 57% | 60% |
| only the key value marked (a hash, a time, IDs) | 21% | 13% |
| marked by 1-4 of 8 samples, removed by the vote | 15% | 16% |
| marked by no sample | 5% | 8% |
| quote not locatable | 1% | 1% |

- Counting any overlap, 78% / 73% of focus sentences are hit: much of the
  gap is granularity (word or phrase vs Nimble's whole sentence).
- Vote losses are mostly linking sentences in a chain ("option C is
  itinerary TC-92", "the object in IMG-2 was MRC-417"): samples that follow
  the chain mark them, the others mark only its end.
- Never-marked sentences are mostly "everything passes" answers, where the
  annotator marks some of the passing conditions and skips others.

### 3.3 Aggregating the 8 samples

Same saved responses for every method.

| method | human F1 (P / R) | Nimble F1 vs focus+policy | focus touched | share highlighted (human / Nimble) |
|---|---|---|---|---|
| vote >= 5 of 8 | 0.53 (0.67 / 0.44) | 0.54 | 0.79 | 0.14 / 0.42 |
| vote >= 4 | 0.55 | 0.59 | 0.86 | 0.16 / 0.50 |
| **vote >= 3** | **0.58** (0.61 / 0.55) | **0.62** | **0.89** | 0.19 / 0.56 |
| vote >= 2 | 0.58 (0.55 / 0.61) | 0.67 | 0.96 | 0.23 / 0.66 |
| grouped spans, support >= 2-5 | 0.54-0.56 | 0.56-0.58 | 0.72-0.80 | |
| self-verification of every distinct span | 0.55 (0.50 / 0.62) | 0.60 | 0.77 | 0.25 / 0.47 |
| vote >= 2, then self-verification | 0.56 | 0.59 | 0.78 | 0.21 / 0.46 |

- A lower threshold (3 of 8) is the best trade-off; 2 of 8 highlights two
  thirds of every Nimble document.
- Grouping spans across samples and using the majority boundary gives
  nothing over character voting.
- Self-verification (the same model, told the gold answer, asked whether
  the answer could change if the text differed) does not raise precision and
  hurts HotpotQA (0.66 -> 0.47-0.56): it rejects bridge facts. The verifier
  agrees with its own proposals.
- Best thresholds differ by dataset (FEVER, HotpotQA, ProcessBench prefer
  2-3; MQM and RAGTruth prefer 5), within noise at 20 rows each.

Precision does not improve with stricter voting or self-verification;
improving it needs an independent signal (reasoning before marking, another
model, or counterfactual checks that do not see the gold answer).

### 3.4 Limits of sample agreement

All samples share the model's biases, so agreement filters random noise but
not systematic errors; strict votes drop chain links; character voting
splits votes between granularities; without reasoning, samples at
temperature 0.7 differ mostly in boundaries; since the annotator is told the
answer, every sample is a justification of it.

### 3.5 Reasoning before marking

4 samples with reasoning (up to 4,096 tokens each) vs the first 4 of the 8
non-reasoning samples, same 300 test rows.

| annotation | best human F1 (P / R) | precision at 3 of 4 | Nimble F1 vs focus+policy |
|---|---|---|---|
| no reasoning | 0.56 (0.51 / 0.63) | 0.68 | 0.64 |
| reasoning | **0.61** (0.58 / 0.64) | **0.76** | 0.63 |

Per dataset (2 of 4): HateXplain 0.37 -> 0.57 and ProcessBench 0.34 -> 0.53
gain; MQM 0.61 -> 0.52, PRM800K 0.65 -> 0.56, RAGTruth 0.38 -> 0.28 lose.
Median reasoning ~3,300 tokens and 37% of samples hit the budget: about
15x the cost of no reasoning. Worth it only for selected task types.

### 3.6 Counterfactual checks without the gold answer

Every distinct span the samples proposed (301 candidates, 20 Nimble + 20
human rows) was tested, with no step told the gold answer: Qwen states the
fact the span gives, rewrites the span to negate it, and a checker that sees
only the document and the fact (as in Nimble) judges it; the answer is
re-scored on the full, span-removed and span-negated documents.

| method | human F1 | Nimble F1 vs focus+policy | share highlighted (human / Nimble) |
|---|---|---|---|
| vote >= 3 of 8 | **0.64** | 0.64 | 0.20 / 0.56 |
| fact-level: removing the span makes the fact unknown | 0.50 | 0.69 | 0.17 / 0.72 |
| answer-level: negating the span lowers p(gold) by >= 0.1 | 0.11 | 0.48 | 0.09 / 0.33 |
| both (Nimble-like) | 0.02 | 0.46 | 0.03 / 0.32 |

The fact-level check is close to a tautology on real documents (the fact
was stated by that span, so removing it makes the fact unknown 69% of the
time). The answer-level check fails because the oracle is weak: without
being told, Qwen3.6-27B gives the gold answer only 0.60 probability on
average and prefers another answer in 15 of 40 rows. Nimble's checks work
because Nimble builds documents where one fact matters and computes the
answer from rules; on real data they do not beat the vote.

## 4. Span granularity

Evidence should be a word, a phrase or a sentence, as needed. Nimble's
references are whole sentences or paragraphs, so they are used for
evaluation (any-overlap hits, precision against focus + policy), not as
training targets.

## 5. Making the answer depend on the evidence

### 5.1 Setup: 10K label-balanced training set

`splits.py --small`: 9,996 rows (600 per human dataset, MQM split between its
1-5 and yes/no questions; 750 per synthetic source; 500 Nimble pairs), each
dataset label-balanced with span-bearing labels weighted 1.5x. Eval: the full
eval split plus nimble-eval (18,842 rows). 3 epochs, from the base model.

### 5.2 Consistency losses (question first)

On rows with gold spans, the tokens the span head selects (p >= 0.5) give two
more passes: sufficiency (cross-entropy of the gold answer from the prompt
with only those tokens) and comprehensiveness (a hinge wanting p(gold) with
them removed at least 0.3 below the full prompt).

| full eval | plain | + consistency |
|---|---|---|
| accuracy (mean over datasets) | 0.796 | 0.797 |
| macro F1 | 0.746 | 0.746 |
| span F1 | 0.41 | 0.41 |
| sufficiency, own highlights (lower better) | 0.164 | 0.120 |
| sufficiency, random tokens of the same size | 0.324 | 0.241 |
| comprehensiveness, own highlights (higher better) | 0.318 | 0.291 |
| comprehensiveness, random tokens | 0.095 | 0.088 |

No cost, but no gain in faithfulness: the model learned to answer from
shortened inputs in general (the random baseline improved as much), so the
gap between its highlights and random text shrank (0.160 -> 0.121).
Reasons: the hard selection gives the span head no gradient, so only the
answer path learned; the losses were absolute, so being robust to any
deletion passes them; and the comprehensiveness hinge on probabilities was
mostly inactive (saturation).

### 5.3 SelfCite-style reward (Chuang et al., ICML 2025)

SelfCite scores a citation by context ablation on the model's own answer y:
log p(y | only the cited text) - log p(y | cited text removed), and uses it
to choose among sampled citation sets (best-of-N) and for preference tuning.
It caps citation length and balances lengths in preference pairs, since
otherwise longer citations win (without balancing their F1 fell to 60.5).
Our version (`selfcite()` in `train_joint.py`): sample 3 evidence sets from
the span head (p >= 0.5 plus two Bernoulli samples), drop empty sets or sets
over half the document, reward each, and train the span head by policy
gradient on the reward minus the candidates' mean.

It fixes the three problems we found with the first consistency losses:
- The signal reaches the highlights. Earlier, the losses only trained how
  the model answers from shortened text, never what it highlights.
- It compares candidates rather than using an absolute target. A model that
  answers well from any text gains nothing, because all candidates then
  score the same.
- It uses log-probabilities, which don't saturate near 0 and 1 the way
  plain probabilities did.

ContextCite (Cohen-Wang et al., NeurIPS 2024) attributes by fitting a sparse
linear model to ~32 random sentence ablations; it also found attention an
unreliable attribution signal.

### 5.4 Error-finding datasets need more data, not shorter documents

With 10K rows the span head marks almost nothing on MQM (recall 0.03),
PRM800K (0.05), ProcessBench (0.01), QA-Feedback (0.10) and RAGTruth (0.01),
against 0.24 / 0.38 / 0.04 / 0.62 / 0.39 for `joint_v2` (83K). Length is not
the cause: MQM translations are 75 tokens long, while HotpotQA (~1,200) and
ContractNLI (~2,000) work. All five are error-finding tasks (the evidence is
a flaw in a response) and the 10K mix has ~360 rows with errors for each.

### 5.5 Question at the end: draft -> evidence -> strict final answer

Document first (cacheable, question-independent document states). The LM's
letter is a draft answer; a head (2 layers: document tokens attend to the
question, the choices and the draft as a probability-weighted mix of the
choices, then to each other) gives the spans; the final answer is strict:
its query is the question encoded without the document, it reads the raw
document states only through attention biased by log p(evidence), and the
choices are scored from document-free encodings. The gold answer is never an
input.

| quick eval, step 800 | final answer | draft (LM letter) | span F1 |
|---|---|---|---|
| question at the end, single-read head | 0.605 | **0.804** | 0.43 |
| + SelfCite | 0.600 | 0.799 | 0.41 |
| question first, plain | 0.797 | - | 0.51 |

- The draft with the question at the end matches question-first (0.80).
- The strict final answer plateaus near 0.60. Diagnostic (epoch 1): zeroing
  the whole document lowers it only 0.58 -> 0.56, and removing the evidence
  bias changes nothing: the single-vector reader learned to answer from
  question-only priors and ignore the document.
- Next: the question's tokens as queries with 2 stacked reader layers (a
  small decoder over cached document states; like ColBERT's separate
  encoding and token-level interaction, but with learned, multi-step
  attention), plus training the final answer toward the detached draft.

## 6. Candidate human-span datasets (not yet vetted)

FaithBench (LLM summaries, expert spans with reasons), REVEAL (reasoning
steps with evidence attribution), ConditionalQA (policy conditions plus
facts), WiCE and SciFact (claim checking with evidence sentences); also XSum
factuality (noisy spans), FELM, SelfCheckGPT WikiBio, HealthFC / HealthVer,
VitaminC, SpanEx, BIG-Bench Mistake, PolicyQA / PrivacyQA, WMT ESA,
LLMs-as-Span-Annotators. Human token-level data remains scarce; many sets
annotate whole sentences.

## 7. Open questions

1. Vote threshold 3 of 8 instead of 5.
2. Reasoning before marking: selectively, for task types where it helps.
3. More weight on error-finding datasets in the small mix.
4. MQM 1-5: keep or drop.
5. Does SelfCite raise faithfulness (full eval pending)?
6. Does the stronger reader close the final-vs-draft gap?
