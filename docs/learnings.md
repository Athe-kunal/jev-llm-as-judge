# Span judge: findings and ablations

A joint judge (Qwen3.5-2B + LoRA) reads a document and a question, picks an
answer from lettered choices (Nimble-style one-letter codes), and marks the
evidence for it in the document with a cross-attention span head. Synthetic
evidence spans come from Qwen3.6-27B, which is given the gold answer.

Unless stated otherwise: span F1 is token-level at a 0.5 threshold; "human
gold" is a test set of 20 rows with human spans from each of 10 human
datasets (200 rows); "Nimble" is 100 Nimble train rows scored against
Nimble's verified evidence.

## 0. How the decision is made (LM letter)

Every task is a question with lettered choices; the answer is the LM letter:
the next-token probabilities at the assistant position, read only for the
row's choice letters and renormalised over them (one forward pass, no
generation; Nimble-style one-letter codes).

| task | choices in the prompt | classified as |
|---|---|---|
| yes/no (RAGTruth, Aegis, HotpotQA, ...) | A. Yes: the response has statements the document contradicts. / B. No: ... | A or B |
| three-way NLI (WANLI, ContractNLI) | A. The premise supports the claim. / B. ... contradicts ... / C. ... leaves it undetermined. | A, B or C |
| HateXplain | A. Hate speech ... / B. Offensive ... / C. Neither ... | A, B or C |
| MQM 1-5 quality | A. Very poor ... / B. Poor ... / C. Fair ... / D. Good ... / E. Excellent ... | A-E |
| Nimble / customer-control | the row's own options (2-15 of them) | A, B, C, ... up to 26 |

The order of the choices is shuffled per row in training, so a letter has no
fixed meaning and the model must read the descriptions; at evaluation the
order is fixed.

From `--yes-no-codes` (saved in each run's args.json; all runs before
2026-10-14 used A / B): yes/no and true/false questions use the codes Y and N
instead, in a fixed order (Y = yes / true), not shuffled; every other task
keeps A, B, C, ... (shuffled in training).

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

### 5.6 A stronger reader for the strict final answer

The question's tokens (encoded without the document) are the queries; each
of 2 reader layers reads the cached document states with evidence-biased
attention, lets the question tokens attend to each other, and applies a
feed-forward step (a small decoder over cached features, like a
question-driven BLIP-2 / InstructBLIP Q-Former without learned queries; like
ColBERT it encodes question and document separately and interacts token by
token, but with learned, multi-step attention). The reader starts from
random weights at width 512 and reads final-layer document states. Training
adds KL(draft, detached || final), so the draft teaches the final answer;
at test time the final answer still sees the document only through the
reader.

| quick eval | single read: final / draft | stronger reader: final / draft | stronger reader + SelfCite: final / draft |
|---|---|---|---|
| step 400 | 0.581 / 0.800 | 0.614 / 0.795 | 0.614 / 0.794 |
| step 600 | 0.588 / 0.802 | 0.697 / 0.802 | 0.671 / 0.801 |
| step 800 | 0.605 / 0.804 | **0.737** / 0.800 | (pending) |

The gap between the strict final answer and the draft shrank from ~0.20 to
~0.06 and was still closing; span F1 is similar across runs (0.36-0.41), so
the gain comes from the reader actually using the document.

Full eval (18,842 rows) of the single-read version, for comparison with the
question-first pair (5.2):

| run | accuracy (mean) | macro F1 (mean) | sufficiency, own (random), lower better | comprehensiveness, own (random), higher better |
|---|---|---|---|---|
| question first, plain | 0.797 | 0.746 | 0.164 (0.368) | 0.318 (0.121) |
| question first, consistency | 0.798 | 0.746 | 0.120 (0.257) | 0.290 (0.104) |
| question at the end, single-read final answer | 0.605 | 0.514 | 0.104 (0.152) | 0.072 (0.045) |

Removing the single-read head's highlights barely changes its answer
(comprehensiveness 0.072, close to random 0.045), consistent with a reader
that ignores the document. Pending: full eval of the stronger reader (with
and without SelfCite) and the zeroed-document diagnostic on it.

### 5.7 SelfCite on the question-at-the-end head

| full eval (18,842 rows) | accuracy (mean) | sufficiency, own (random) | comprehensiveness, own (random) |
|---|---|---|---|
| single read | 0.605 | 0.104 (0.152) | 0.072 (0.045) |
| single read + SelfCite | 0.600 | 0.120 (0.170) | 0.076 (0.050) |

| quick eval, step 800 | final | draft | span F1 |
|---|---|---|---|
| stronger reader | 0.737 | 0.800 | 0.41 |
| stronger reader + SelfCite | 0.720 | 0.804 | 0.41 |

No effect on accuracy, spans or the final answer's faithfulness. Caveat: the
SelfCite reward trains the evidence choice through the draft (the LM letter
on shortened prompts), while these faithfulness numbers measure the final
answer. `eval_joint` now also reports the draft's sufficiency and
comprehensiveness on the same highlights (evidence head only), so SelfCite
can be judged on what it optimises; the stronger-reader pair is being
re-evaluated with it.

### 5.8 Full comparison of the six 10K runs

Full eval, 18,842 rows; accuracy and macro F1 are means over datasets; for
the evidence-head runs these are the strict final answer.

| run | accuracy | macro F1 | sufficiency, own (random), lower better | comprehensiveness, own (random), higher better |
|---|---|---|---|---|
| question first, plain | **0.797** | **0.746** | 0.164 (0.368) | **0.318** (0.121) |
| question first, consistency | 0.798 | 0.746 | 0.120 (0.257) | 0.290 (0.104) |
| question at the end, single read | 0.605 | 0.514 | 0.104 (0.152) | 0.072 (0.045) |
| single read + SelfCite | 0.600 | 0.509 | 0.120 (0.170) | 0.076 (0.050) |
| question at the end, stronger reader | 0.735 | 0.680 | 0.076 (0.146) | 0.071 (0.018) |
| stronger reader + SelfCite | 0.711 | 0.647 | 0.103 (0.195) | 0.075 (0.025) |

Draft answer (LM letter) of the evidence-head runs, same highlights:

| run | draft accuracy | sufficiency (random) | comprehensiveness (random) |
|---|---|---|---|
| stronger reader | 0.753 | 0.194 (0.278) | 0.176 (0.109) |
| stronger reader + SelfCite | 0.755 | 0.181 (0.267) | 0.189 (0.116) |

Per dataset (accuracy / span F1 / comprehensiveness, random in brackets):

| dataset | question first, plain | stronger reader | + SelfCite |
|---|---|---|---|
| aegis | 0.906 / 0.69 / 0.316 (0.177) | 0.877 / 0.68 / 0.120 (0.046) | 0.873 / 0.70 / 0.143 (0.093) |
| citation_control | 1.000 / 0.48 / 0.651 (0.276) | 0.996 / 0.59 / 0.438 (0.352) | 0.778 / 0.59 / 0.100 (0.022) |
| contractnli | 0.786 / 0.39 / 0.314 (0.008) | 0.764 / 0.00 / 0.006 (0.001) | 0.758 / 0.00 / 0.008 (-0.001) |
| customer_control | 1.000 / 0.76 / 0.584 (0.020) | 0.962 / 0.73 / 0.492 (0.093) | 0.944 / 0.76 / 0.473 (0.073) |
| fever | 0.937 / 0.57 / 0.427 (0.110) | 0.805 / 0.31 / 0.001 (0.000) | 0.804 / 0.38 / 0.007 (0.001) |
| hatexplain | 0.601 / 0.53 / 0.322 (0.092) | 0.574 / 0.49 / 0.253 (0.067) | 0.542 / 0.52 / 0.231 (0.063) |
| hotpotqa | 0.854 / 0.64 / 0.222 (0.032) | 0.756 / 0.24 / -0.001 (-0.002) | 0.752 / 0.20 / -0.002 (-0.001) |
| mqm | 0.397 / 0.06 / 0.043 (0.002) | 0.447 / 0.07 / 0.024 (0.001) | 0.416 / 0.07 / 0.012 (0.004) |
| mushroom | 0.954 / 0.64 / 0.004 (0.001) | 0.965 / 0.63 / 0.033 (-0.009) | 0.960 / 0.64 / 0.001 (-0.007) |
| nimble_eval | 0.784 / 0.49 / 0.495 (0.371) | 0.364 / 0.38 / 0.006 (-0.011) | 0.392 / 0.34 / 0.001 (-0.004) |
| prm800k | 0.785 / 0.09 / 0.019 (0.009) | 0.807 / 0.28 / 0.009 (-0.016) | 0.807 / 0.33 / 0.064 (-0.010) |
| processbench | 0.768 / 0.01 / -0.001 (-0.004) | 0.784 / 0.03 / -0.002 (-0.006) | 0.808 / 0.05 / 0.003 (0.001) |
| qa_feedback | 0.756 / 0.17 / 0.052 (0.023) | 0.768 / 0.15 / 0.010 (-0.002) | 0.662 / 0.17 / 0.045 (0.001) |
| ragtruth | 0.732 / 0.02 / 0.024 (0.003) | 0.702 / 0.09 / 0.006 (-0.002) | 0.731 / 0.14 / 0.032 (-0.001) |
| wanli | 0.696 / 0.55 / 0.477 (0.308) | 0.459 / 0.49 / -0.009 (-0.013) | 0.444 / 0.52 / -0.008 (0.001) |

Findings:
- Question first is still the best model: highest accuracy, macro F1 and
  comprehensiveness (its answers depend on its highlights the most).
- The stronger reader lifts the strict final answer from 0.605 to 0.735,
  but it stays 6 points behind, and most of the gap is in claim and
  multi-hop tasks: WANLI -0.24, Nimble -0.42, FEVER -0.13, HotpotQA -0.10.
  There, with the question at the end, the span head barely works
  (ContractNLI span F1 0.00, HotpotQA 0.24, FEVER 0.31 vs 0.39 / 0.64 / 0.57
  with the question first) and removing the highlights does not change the
  final answer (comprehensiveness ~0).
- The strict reader is only weakly evidence-dependent overall:
  comprehensiveness 0.071 vs 0.018 random. The soft log p(evidence) bias
  still lets it read unhighlighted text.
- It does slightly better than question first on several error-finding
  datasets (MQM 0.447 vs 0.397, PRM800K 0.807 vs 0.785, ProcessBench 0.784 vs
  0.768; PRM800K span F1 0.28 vs 0.09).
- SelfCite: no gain on the final answer (accuracy 0.711 vs 0.735) and at
  most a marginal one on the draft it trains (comprehensiveness 0.189 vs
  0.176, random 0.116 vs 0.109); it hurt citation-control (1.000 -> 0.778)
  and QA-Feedback (0.768 -> 0.662).

### 5.9 Full run: question at the end, label-balanced data

`splits.py --balanced`: the full training set re-sampled label-balanced (each
dataset keeps its size; labels equal, span-bearing labels weighted 1.5x;
eval groups excluded) plus all of nimble-train: 85,860 rows. Aegis (1,233
unsafe rows available), ContractNLI ("contradicted") and Mu-SHROOM (63 "no")
could not be fully balanced. Model: question at the end, evidence head with
the stronger reader and the draft-to-final teaching loss, no SelfCite; 2
epochs (5,366 steps, ~12 h on 2 GPUs).

Full eval of the final checkpoint (18,842 rows); "LM letter" is the
Nimble-style answer: the probabilities of the choice letters at the
assistant position (the draft).

| dataset | LM letter | strict reader | span F1 | question first 10K: acc / span F1 | joint_v2 83K (question first, old mix): acc / span F1 |
|---|---|---|---|---|---|
| aegis | 0.901 | 0.885 | 0.77 | 0.906 / 0.69 | 0.911 / 0.79 |
| citation_control | 1.000 | 0.746 | 0.63 | 1.000 / 0.48 | 0.996 / 0.33 |
| contractnli | 0.887 | 0.850 | 0.25 | 0.786 / 0.39 | 0.892 / 0.63 |
| customer_control | 1.000 | 0.842 | 0.75 | 1.000 / 0.76 | 1.000 / 0.76 |
| fever | 0.953 | 0.832 | 0.58 | 0.937 / 0.57 | 0.947 / 0.69 |
| hatexplain | 0.667 | 0.676 | 0.56 | 0.601 / 0.53 | 0.630 / 0.56 |
| hotpotqa | 0.882 | 0.782 | 0.49 | 0.854 / 0.64 | 0.906 / 0.89 |
| mqm | 0.477 | 0.480 | 0.31 | 0.384 / 0.06 | 0.369 / 0.32 |
| mushroom | 0.969 | 0.969 | 0.71 | 0.954 / 0.64 | 0.971 / 0.65 |
| nimble_eval | 0.765 | 0.602 | 0.56 | 0.784 / 0.49 | - |
| prm800k | 0.831 | 0.826 | 0.71 | 0.785 / 0.09 | 0.860 / 0.44 |
| processbench | 0.774 | 0.786 | 0.03 | 0.768 / 0.01 | 0.768 / 0.07 |
| qa_feedback | 0.850 | 0.834 | 0.63 | 0.756 / 0.17 | 0.834 / 0.61 |
| ragtruth | 0.809 | 0.811 | 0.45 | 0.732 / 0.02 | 0.685 / 0.45 |
| wanli | 0.773 | 0.596 | 0.68 | 0.696 / 0.55 | 0.764 / 0.63 |
| **mean** | **0.836** | 0.768 | 0.54 | 0.796 / 0.41 | 0.824 / 0.56 (no Nimble) |

| faithfulness (rows with highlights) | sufficiency, own (random) | comprehensiveness, own (random) |
|---|---|---|
| LM letter | 0.183 (0.324) | 0.232 (0.105) |
| strict reader | 0.131 (0.220) | 0.140 (0.051) |
| question first 10K (LM letter), for reference | 0.164 (0.368) | 0.318 (0.121) |

Step 4,000 vs final: LM letter mean 0.835 -> 0.836 (converged); strict reader
0.737 -> 0.768 (still improving).

Findings:
- The LM letter is the best decision model so far (mean 0.836 vs 0.824 for
  joint_v2). The gains are on the datasets the old label mix had biased:
  RAGTruth 0.685 -> 0.809, MQM 0.369 -> 0.477, HateXplain 0.630 -> 0.667,
  QA-Feedback 0.834 -> 0.850. Label balancing works.
- Spans are strong on error-finding tasks (PRM800K 0.71 vs 0.09 at 10K and
  0.44 for joint_v2; QA-Feedback 0.63; Mu-SHROOM 0.71; RAGTruth 0.45) but
  weak on long claim / multi-hop documents with the question at the end
  (ContractNLI 0.25, HotpotQA 0.49 vs 0.63 / 0.89 question first): document
  tokens never see the question inside Qwen. ProcessBench spans stay ~0 in
  every model.
- The strict reader trails the LM letter by ~0.07 and is only weakly
  evidence-dependent; the main design should be the LM letter plus the span
  head, with the reader optional. Whether training the reader helped the
  letter or the spans is untested (no same-data run without it).
- The LM letter depends on its highlights more than at 10K (comprehensiveness
  0.232 vs 0.176) but less than with the question first (0.318).

### 5.10 All runs side by side

Full eval (18,842 rows). Every column is a mean over datasets (so it can
differ from the row-weighted numbers above). "accuracy" is the answer each
model reports: the LM letter for question-first runs, the strict reader for
question-at-the-end runs, whose LM letter is in its own column. Sufficiency:
lower is better; comprehensiveness: higher is better; random baselines in
brackets. joint_v2 was evaluated before the faithfulness metrics existed.

| run | accuracy | macro F1 | LM letter acc | span P | span R | span F1 | suff (rand) | comp (rand) | LM letter comp (rand) |
|---|---|---|---|---|---|---|---|---|---|
| joint_v2: question first, 83K old mix, 3 ep | 0.824 | 0.778 | – | 0.681 | 0.508 | 0.559 | – | – | – |
| 10K question first, plain | 0.797 | 0.746 | – | 0.580 | 0.352 | 0.405 | 0.183 (0.324) | 0.263 (0.095) | – |
| 10K question first, + consistency | 0.798 | 0.746 | – | 0.577 | 0.353 | 0.407 | 0.120 (0.241) | 0.254 (0.088) | – |
| 10K question at end, single read | 0.605 | 0.514 | 0.800 | 0.607 | 0.295 | 0.359 | 0.095 (0.136) | 0.035 (0.021) | – |
| 10K question at end, single read + SelfCite | 0.600 | 0.509 | 0.798 | 0.589 | 0.300 | 0.353 | 0.137 (0.178) | 0.034 (0.024) | – |
| 10K question at end, stronger reader | 0.735 | 0.680 | 0.797 | 0.546 | 0.280 | 0.345 | 0.127 (0.214) | 0.092 (0.033) | 0.157 (0.100) |
| 10K question at end, stronger reader + SelfCite | 0.711 | 0.647 | 0.800 | 0.535 | 0.305 | 0.361 | 0.129 (0.236) | 0.074 (0.016) | 0.153 (0.085) |
| 86K balanced, question at end, stronger reader (step 4000) | 0.737 | 0.661 | 0.835 | 0.706 | 0.455 | 0.527 | 0.147 (0.208) | 0.095 (0.038) | 0.229 (0.111) |
| 86K balanced, question at end, stronger reader (final) | 0.768 | 0.696 | 0.836 | 0.715 | 0.469 | 0.541 | 0.159 (0.238) | 0.121 (0.046) | 0.230 (0.103) |
| 10K Qwen3-1.7B bidirectional, question at end, cross head (5.15) | 0.530 | 0.350 | – | 0.475 | 0.164 | 0.200 | 0.017 (0.021) | 0.007 (0.007) | – |
| 10K Qwen3-1.7B causal, question at end, cross head (5.15) | 0.757 | 0.698 | – | 0.533 | 0.238 | 0.285 | 0.252 (0.316) | 0.176 (0.121) | – |
| 10K DeFormer, top 6 layers joint, cross head (5.16) | 0.650 | 0.571 | – | 0.542 | 0.253 | 0.297 | 0.117 (0.173) | 0.094 (0.071) | – |

### 5.11 GLiNER + ColBERT span head (stopped early)

`--head match` (`MatchHead` in `train_joint.py`), question at the end, LM
letter as the answer, 10K balanced data. Document tokens (document-first,
cacheable) get one bidirectional self-attention block in the head; the
question and the answer choices are encoded without the document. Evidence
logit = learned mix of: a soft maximum over question tokens of cosine
similarity (ColBERT's MaxSim made soft), how much attention the question
tokens give the token (each question token spreads a softmax over the
document), similarity to the answer choices weighted by the LM letter's
probabilities (GLiNER label matching), and a per-token prior. Stopped at step
490 of 936 because it was not closing the gap.

| quick eval | GLiNER + ColBERT: span F1 (letter acc) | stronger reader, question at the end | question first |
|---|---|---|---|
| step 200 | 0.205 (0.751) | 0.116 (0.747) | 0.282 (0.756) |
| step 400 | 0.336 (0.793) | 0.382 (0.795) | 0.365 (0.796) |

Span F1 at step 400:

| dataset | GLiNER + ColBERT | stronger reader | question first |
|---|---|---|---|
| fever | 0.05 | 0.06 | 0.41 |
| hotpotqa | 0.16 | 0.03 | 0.43 |
| contractnli | 0.00 | 0.00 | 0.05 |
| wanli | 0.27 | 0.33 | 0.39 |
| citation_control | 0.34 | 0.53 | 0.17 |
| qa_feedback | 0.34 | 0.01 | 0.06 |
| prm800k | 0.16 | 0.03 | 0.00 |
| mqm | 0.10 | 0.00 | 0.02 |
| customer_control | 0.66 | 0.64 | 0.54 |
| aegis | 0.68 | 0.72 | 0.69 |
| hatexplain | 0.48 | 0.47 | 0.56 |
| mushroom | 0.59 | 0.59 | 0.56 |
| nimble_eval | 0.48 | 0.59 | 0.48 |

- Late interaction does not fix claim and multi-hop evidence with the
  question at the end (FEVER 0.05 and HotpotQA 0.16 vs 0.41 and 0.43 question
  first; ContractNLI 0): matching finds related text, not what decides a
  claim or the second hop.
- It learned error-finding spans earlier than the other heads (QA-Feedback
  0.34, PRM800K 0.16, MQM 0.10), probably through its per-token prior, not
  the matching; the other heads caught up later in their runs.
- Conclusion: the gap is depth of interaction between the question and the
  document, not the shape of the head. Next: bidirectional models (document
  tokens that see the question, or at least the whole document).

### 5.12 Bidirectional options (decided direction)

Qwen3.5 cannot be made bidirectional (18 of 24 layers are left-to-right
linear attention); Qwen3 (all full attention) can by changing the mask, and
gte-Qwen2-1.5B-instruct is already trained bidirectional (multilingual,
32K). The LM letter still works: one forward pass, letter scores read at an
answer slot at the end (no generation). With the question at the end:
- full bidirectional attention: document tokens see the question and the
  whole document (best spans expected), but no document cache;
- prefix-LM (bidirectional within the document, causal for the question):
  the document cache works and document tokens see the whole document, but
  not the question; evidence must come from the question side (an
  attention-map probe over all of Qwen3's layers, or a reader).
UltraBERT (dmis-lab, 150M / 380M, 32K, Apache 2.0) is English-only, a
problem for MQM and Mu-SHROOM.

Outcome: full bidirectional Qwen3-1.7B collapsed (5.15); Qwen3-Reranker and
Qwen3-Embedding are causal too, so this direction is closed.

### 5.13 Ideas discussed, not yet run

- Span heads for the question at the end: DeFormer-style partial caching
  (now running, 5.16), or a BiDAF-style bi-attention span head;
  a CRF on top for contiguous spans. Other architectures considered:
  Q-Former (BLIP-2 / InstructBLIP), Flamingo gated cross-attention, Perceiver
  IO, Fusion-in-Decoder, ColBERT late interaction.
- Differentiable sufficiency / comprehensiveness: soft masks from the span
  head applied inside Qwen (log m on full-attention keys; the linear-attention
  layers already multiply inputs by a mask), Hard Concrete gates or
  straight-through so masks are near-binary, a size penalty and a random
  same-size baseline; gradients reach the span head directly (DiffMask,
  Bastings et al. 2019).
- Sentence markers `[S n] ... [/S n]`: closing-marker states summarise each
  sentence; sentence-level evidence head, sentence-level ablations (best-of-N,
  ContextCite-style subsets, soft masks), with the token span head for
  word / phrase highlights inside sentences.
- Best-of-N at inference or as training targets: candidates from the span
  head (nested top-k spans are cheap but only pick a cutoff in the head's
  ranking; ContextCite-style random subsets of spans with a sparse linear fit
  handle redundancy and joint evidence), scored by SelfCite's reward on the
  LM letter with a size penalty.
- SelfCite to build synthetic data: score each of the 8 saved samples' span
  sets by the reward on the gold answer and keep the best, instead of
  voting; test on the 300-row test set first (the scorer must believe the
  gold answer: Qwen3.6-27B gave it only 0.60 on average).
- No LoRA (frozen Qwen, train only the span head): needs the zero-shot LM
  letter accuracy first.

### 5.14 Next: a reader built from Qwen's own layers

Possible next step: initialise the reader from Qwen3.5-2B's own
full-attention layers. The model is a hybrid: of 24 layers only 3, 7, 11, 15,
19 and 23 use full attention (8 query heads, 2 key/value heads, head size
256, output gate, rotary positions on 25% of each head); the rest are linear
attention (Gated DeltaNet), which keeps a running state and cannot attend to
another sequence. Copies of layers 11, 15 and 19 (middle-upper, where models
look up earlier context) would read layer-matched document states
(`hidden_states[L]`, the input to layer L) with question queries at the same
depth, shifted to positions after the document; ~45-50M parameters each,
fine-tuned.

### 5.15 Qwen3-1.7B: bidirectional vs causal (question at the end)

Full bidirectional attention (5.12) on Qwen3-1.7B, and the same run with
ordinary causal attention as the control. Both: LoRA + cross head (`SpanHead`),
LM letter, question at the end, 10K balanced data, 3 epochs. Full eval
(18,842 rows), accuracy / span F1 per dataset, with the Qwen3.5-2B 10K runs:

| dataset | Qwen3 bidirectional | Qwen3 causal | Qwen3.5 question first | Qwen3.5 question at end (LM letter) |
|---|---|---|---|---|
| aegis | 0.683 / 0.68 | 0.873 / 0.66 | 0.906 / 0.69 | 0.892 / 0.68 |
| citation_control | 0.351 / 0.00 | 0.952 / 0.36 | 1.000 / 0.48 | 1.000 / 0.62 |
| contractnli | 0.504 / 0.00 | 0.754 / 0.00 | 0.786 / 0.39 | 0.801 / 0.02 |
| customer_control | 0.628 / 0.06 | 1.000 / 0.72 | 1.000 / 0.76 | 1.000 / 0.76 |
| fever | 0.541 / 0.12 | 0.945 / 0.26 | 0.937 / 0.57 | 0.947 / 0.34 |
| hatexplain | 0.330 / 0.38 | 0.514 / 0.47 | 0.601 / 0.53 | 0.582 / 0.51 |
| hotpotqa | 0.544 / 0.00 | 0.756 / 0.18 | 0.854 / 0.64 | 0.840 / 0.25 |
| mqm | 0.297 / 0.00 | 0.397 / 0.00 | 0.384 / 0.06 | 0.444 / 0.05 |
| mushroom | 0.967 / 0.58 | 0.971 / 0.59 | 0.954 / 0.64 | 0.954 / 0.64 |
| nimble_eval | 0.370 / 0.35 | 0.664 / 0.43 | 0.784 / 0.49 | 0.735 / 0.41 |
| prm800k | 0.825 / 0.30 | 0.777 / 0.11 | 0.785 / 0.09 | 0.788 / 0.35 |
| processbench | 0.642 / 0.00 | 0.744 / 0.01 | 0.768 / 0.01 | 0.780 / 0.03 |
| qa_feedback | 0.612 / 0.10 | 0.710 / 0.01 | 0.756 / 0.17 | 0.780 / 0.21 |
| ragtruth | 0.345 / 0.00 | 0.589 / 0.00 | 0.732 / 0.02 | 0.711 / 0.03 |
| wanli | 0.322 / 0.43 | 0.695 / 0.46 | 0.696 / 0.55 | 0.726 / 0.49 |
| **mean** | **0.531 / 0.20** | **0.756 / 0.28** | **0.796 / 0.41** | **0.799 / 0.36** |

(Means over the 18,599 rows all four runs predicted; Qwen3.5 at the end uses
the evidence head, so its spans are not the same head.)

- The collapse comes from switching a causal model to bidirectional
  attention without adapting it, not from Qwen3 in this pipeline: the causal
  control trains normally (0.756 vs 0.531; the bidirectional run mostly gives
  one answer per dataset, and its faithfulness is at random).
- Qwen3-1.7B is still behind Qwen3.5-2B (accuracy -0.04, span F1 -0.08 to
  -0.13), most on RAGTruth, HotpotQA, nimble-eval and HateXplain. Qwen3.5-2B
  stays the base model.
- With the question at the end the one-layer cross head finds no evidence
  on ContractNLI, RAGTruth, MQM and QA-Feedback (0.00-0.01) and little on
  HotpotQA (0.18) and FEVER (0.26): the same gap as in 5.8-5.11.

### 5.16 DeFormer-style split (Cao et al., ACL 2020)

For one document with many questions the document's states must not depend
on the question, which rules out question first in a causal model. DeFormer
split: the lower 24 - N layers read [system ; question] and [system ;
document ; answer tail] as two separate sequences (the document's lower
layers are computed once and reused across questions); the top N layers
read the question-first prompt [system ; question ; document ; tail] built
from those states, with positions renumbered. Cost per question: N/24 of a
pass over the document. Differences from DeFormer: causal decoder (in the
top layers the document reads the question, not the reverse; the letter at
the end reads both), RoPE applied inside each layer, linear-attention layers
in the top block (Qwen3.5-2B's top 6: 18-23, two of them full attention); no
distillation losses yet. Checked: N = 24 gives exactly the question-first
forward; the document's lower states are identical under different
questions. `--deformer 6`, `runs/small_deformer6`, otherwise as the 10K
question-first run (cross head, LM letter).

Full eval (accuracy / span F1, means over the 18,599 rows all runs
predicted):

| dataset | DeFormer, top 6 joint | question first | question at end (evidence head) |
|---|---|---|---|
| aegis | 0.897 / 0.70 | 0.906 / 0.69 | 0.892 / 0.68 |
| citation_control | 0.403 / 0.19 | 1.000 / 0.48 | 1.000 / 0.62 |
| contractnli | 0.530 / 0.01 | 0.786 / 0.39 | 0.801 / 0.02 |
| customer_control | 0.714 / 0.55 | 1.000 / 0.76 | 1.000 / 0.76 |
| fever | 0.802 / 0.25 | 0.937 / 0.57 | 0.947 / 0.34 |
| hatexplain | 0.545 / 0.49 | 0.601 / 0.53 | 0.582 / 0.51 |
| hotpotqa | 0.766 / 0.27 | 0.854 / 0.64 | 0.840 / 0.25 |
| mqm | 0.413 / 0.03 | 0.384 / 0.06 | 0.444 / 0.05 |
| mushroom | 0.963 / 0.67 | 0.954 / 0.64 | 0.954 / 0.64 |
| nimble_eval | 0.537 / 0.51 | 0.784 / 0.49 | 0.735 / 0.41 |
| prm800k | 0.752 / 0.08 | 0.785 / 0.09 | 0.788 / 0.35 |
| processbench | 0.752 / 0.01 | 0.768 / 0.01 | 0.780 / 0.03 |
| qa_feedback | 0.576 / 0.23 | 0.756 / 0.17 | 0.780 / 0.21 |
| ragtruth | 0.598 / 0.01 | 0.732 / 0.02 | 0.711 / 0.03 |
| wanli | 0.491 / 0.44 | 0.696 / 0.55 | 0.726 / 0.49 |
| **mean** | **0.649 / 0.30** | **0.796 / 0.41** | **0.799 / 0.36** |

Comprehensiveness 0.094 (random 0.071), against 0.318 (0.121) question first.

- Worse than both orders on the decision (-0.15) and on spans (0.30). The
  quick evals were still rising (0.52 -> 0.66 from step 200 to 800), so 3
  epochs on 10K are not enough for the top 6 layers to learn the new input.
- Likely cause: the letter is read at the end of the document segment, so
  in the lower 18 layers it has seen the document but not the question; the
  decision rests on 6 layers. With the question at the end the letter sees
  the question in all 24 layers (0.80). Tasks that need the question to read
  the document (WANLI, ContractNLI, citation-control, customer-control) drop
  most; tasks where the document alone mostly decides (Aegis, Mu-SHROOM) do not.
- A variant that keeps the cache and fixes this: lower document sequence
  [system ; document ; question ; tail] (the document still comes before the
  question, so it stays cacheable), top layers [question ; document ;
  question ; tail].

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
6. Does the stronger reader also make answers depend on the evidence
   (comprehensiveness well above random)?
7. A reader initialised from Qwen's full-attention layers 11, 15, 19.
