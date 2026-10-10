# Span-highlighting judge: data generation pipeline design

Date: 2026-10-06
Status: draft for review (rev 3: JEV-style typed questions, scaled question bank, local compute, label reliability, lane D)
Source brief: "Span-highlighting judge: data pipeline brief" (pasted 2026-10-06). This doc refines that brief into a buildable design; where they differ, this doc wins once approved.

## 1. Purpose and success criteria

Build training and evaluation data for a small long-context encoder judge that imitates **JEV** (TypeSafe AI's decision-only judge) and adds what JEV lacks: **highlighted spans that justify every verdict**, so a human can verify a judgment at a glance.

JEV's interface, which this model copies: one shared **state** (text or JSON; here a conversation with optional documents) plus one or more named, **typed questions**; for each question it returns a **label and a calibrated probability distribution over the allowed labels**. Question types:

| Type | Allowed labels | Our addition |
|---|---|---|
| `boolean` | yes / no | Spans supporting the answer |
| `choice` | one of 2–N described options (exclusive) | Spans supporting the chosen option |
| `score` | one of 2–10 described rubric levels | Spans with severities; level also derivable from spans |

Model training is out of scope. Done means:

- ~100K training records and held-out evaluation sets, JSONL, with a data card.
- Every record passes offset validation.
- No source conversation (including near-duplicates and shared upstream dialogues) appears in both train and eval.
- Held-out question clusters are absent from training in every paraphrase.
- Balance report shows: all three question types present, ~1/3 no-span records, several hundred distinct questions, and both single- and multi-turn states.

### 1.1 What spans mean per question type

- `score`: spans mark the text that lowers the level (violations), each with a `severity` in level units. $s_{\text{spans}} = \max(\ell_{\min},\ \ell_{\max} - \sum_i w_i - w_{\text{omission}})$. Clean targets have no spans and the top level. Positive requirements are phrased so that their absence is an omission (no span, `omission.flag = true`).
- `boolean` / `choice`: spans mark the text that supports the answer label, each with `supports: <label id>` and a `weight` (probability drop when the span is neutralized, §8.1). Labels that mean "nothing present" (e.g. `no` to "Does the response contradict the context?") normally have no spans.
- All types: spans live in the target turn; supporting text in earlier turns or documents goes in `evidence`.
- Every record stores both the span-derived answer and the answer distribution (gold or teacher), so training can use either and a consistency loss between them, without regenerating data.

Out of scope for now: pairwise preference questions and preference datasets (user decision, 2026-10-06).

## 2. Decisions

| Decision | Value | Status |
|---|---|---|
| Use | Research only; NC/SA sources allowed; license recorded per record | Decided (user) |
| Judge interface | JEV-style state + typed questions (boolean, choice, score) + spans | Decided (user) |
| Score questions | Store span severities and level distribution; formula vs distribution decided at training time | Decided (user) |
| Preference data | Out of scope | Decided (user) |
| Schema | Rev 2 schema in §4 (replaces brief §3) | Approved in principle (user); confirm on spec review |
| Pipeline structure | File-based stages, content-addressed LLM cache | Proposed |
| Compute | All LLM roles local: vLLM on 3×A100 80GB, overnight runs; Fireworks kept only as an optional backend | Decided (user) |
| Model to train | Own JEV-style encoder (GLiNER-family); JEV itself is not used | Decided (user) |
| Proposer / editor / scorer models | Local open-weights candidates, chosen in the pilot by throughput, edit acceptance, and calibration (§12) | Proposed; decided in pilot |
| Non-English | Kept, tagged `meta.language`, pretrain stage only; config flag can exclude | Default; user may override |
| Lanes | A human labels, B counterfactual, C injection, **D code-verified** (§8.4) | Proposed |
| Budget | Local GPU time, not API spend: pilot = 1 overnight run; full run needs explicit approval and an estimated number of nights | Open (full-run nights) |
| Domains | Multi-domain, balanced (§5.6); no single domain dominates | Decided (user) |
| Question scale | Not limited to dataset rubrics: mined instance rubrics + taxonomy-generated + state-conditioned + code-checkable questions; target 10–30K distinct base questions | Decided (user) |

Scorer probe (2026-10-06, one prompt, Fireworks): with thinking disabled, all tested models except thinking-only ones put ≥0.98 probability mass on the label tokens; assistant prefill breaks most models and is not used. The same probe is repeated on the local candidates in the pilot.

## 3. Architecture

Six stages, each a CLI command that reads inputs from disk and writes outputs plus a manifest. Stages are idempotent and resumable: rerunning a stage skips records already written and LLM calls already cached.

```
0 acquire   data/raw/<source>/          + manifest.json (url, pinned revision/commit, sha256, license, retrieved_at)
1 split     data/registry/              conversation groups, eval assignment, question bank + held-out clusters
2 lane A    data/records/laneA/<source>.jsonl
3 lane B/C/D data/records/laneB|laneC|laneD/<run_id>.jsonl   (+ data/llm_cache/, data/ledger/)
4 merge     data/merged/                validation, dedup, truncation, teacher distributions, control paraphrases, balance report
5 export    data/final/train.jsonl, eval_*.jsonl, DATA_CARD.md
```

Every stage writes rejected items to `data/rejects/<stage>/<source>.jsonl` with a machine-readable `reason`. Reject counts per reason appear in the balance report.

Stage 1 runs before stages 2–3 and its registry is read-only afterwards. Stages 2–3 consult it and never generate train records from eval conversations or held-out questions.

### 3.1 Repository layout

```
pyproject.toml               package "spanjudge"; align requires-python with .python-version (both 3.11)
configs/pilot.yaml, full.yaml
prompts/<role>/<name>.v<N>.md     versioned prompt files; hash stored in record meta
src/spanjudge/
  schema.py           Pydantic models, offset validation, id hashing
  text.py             normalization, quote→offset location, char diff, word-boundary snapping
  registry.py         conversation fingerprints, grouping, split assignment
  llm/client.py       one client: backends, cache, retries, rate limit, cost ledger, budget cap
  llm/backends.py     fireworks (OpenAI-compatible), vllm (local), jev (if available)
  llm/answer.py       label distribution from logprobs for any question type
  acquire/            one module per source: download + manifest
  laneA/              one converter per source
  laneB/              propose, intervene, re-answer, select
  laneC/              select clean turn, inject, verify
  laneD/              code checkers for verifiable constraints
  responses/          response generation (§8.0)
  questions/          question bank build, rewrite, paraphrase, clustering
  merge/              filters, truncation, teacher distributions, control paraphrases, balance report
  export.py           final split files, data card
  cli.py              `python -m spanjudge <stage> --config configs/pilot.yaml`
tests/
  fixtures/<source>/  a few real raw rows per source + expected records
```

## 4. Record schema (rev 2)

One record = one (state, question) pair. Records sharing a state share `state_id`, so the model can be trained JEV-style with several questions per encoded state.

```json
{
  "id": "string, stable and unique",
  "state_id": "string, shared by all records over the same state",
  "source": "dataset name",
  "lane": "human | counterfactual | injection",
  "state": {
    "conversation": [{"role": "user|assistant|system", "content": "..."}],
    "context_docs": ["optional grounding documents"],
    "target_turn": 3
  },
  "question": {
    "id": "base question id",
    "type": "boolean | choice | score",
    "text": "full question text (possibly a paraphrase)",
    "options": [{"id": "yes", "text": "..."}, {"id": "no", "text": "..."}],
    "levels":  [{"value": 1, "text": "level description"}, {"value": 5, "text": "..."}]
  },
  "answer": {
    "label": "yes | option id | level value as string",
    "dist": {"yes": 0.91, "no": 0.09},
    "dist_source": "teacher | smoothed_gold",
    "gold_label": "label from human annotation, or null"
  },
  "spans": [
    {"start": 120, "end": 164, "text": "verbatim substring", "supports": "yes", "weight": 0.42, "severity": null}
  ],
  "evidence": [
    {"where": "turn:1 | doc:0", "start": 10, "end": 52, "text": "verbatim substring"}
  ],
  "omission": {"flag": false, "severity": null},
  "span_score": null,
  "meta": {}
}
```

Conventions:

- `options` is set for `boolean` (fixed yes/no, with question-specific descriptions) and `choice`; `levels` for `score`. Option order is shuffled per record and stored as given; the model must not rely on position.
- `score` spans set `severity` (level units) and `supports: null`; `boolean`/`choice` spans set `supports` and `weight` (probability units) and `severity: null`.
- `span_score` is set only for `score` records: the formula in §1.1.
- `answer.dist` covers every allowed label and sums to 1. `answer.label` is its argmax unless `gold_label` is set, in which case `label = gold_label`.
- Offsets are Python string indices (Unicode code points) into NFC-normalized text, normalized once at load time.
- `id = sha1(source | source_id | target_turn | question.id | lane | variant)[:16]`; `state_id = sha1(normalized state)[:16]`.
- `meta` always contains: `license`, `source_id`, `split`, `stage` (`pretrain` if the state has one user turn before the target, else `finetune`), `language`, `question_family`, `question_paraphrase`, `option_order`, `prompt_hashes` and `models` (lanes B/C and teacher distributions), `deltas` (lane B/C raw measurements), `truncated_turns` (if any), `orig` (lane A native labels).
- Roles: source dialogues between two humans map the knowledge-holding or responding party to `assistant` (FaithDial wizard, DECODE responder) and the other to `user`.

`schema.validate(record)` checks: every span and evidence `text == content[start:end]`; `target_turn` points at an `assistant` turn; spans non-overlapping and sorted; type-specific fields present and consistent (`supports` ∈ option ids; severities in (0, `levels` range]; weights in (0, 1]); `dist` keys equal the allowed labels and sum to 1 ± 1e-6; `score` records with no spans and no omission have `span_score == ℓ_max`; evidence `where` references an earlier turn or an existing doc.

## 5. Shared components

### 5.1 Text utilities (`text.py`)

- `locate(quote, content)`: exact match only; multiple matches require a disambiguating context window from the caller, otherwise the quote is rejected.
- `char_diff(a, b)`: changed regions (character difflib, merged when separated by ≤2 unchanged chars), snapped outward to word boundaries.
- `apply_edits(text, edits)`: applies non-overlapping (start, end, replacement) edits right-to-left; returns new text and an offset map so spans can be re-located after edits.

These carry the correctness of lanes B and C and get property-based tests (§11).

### 5.2 LLM client (`llm/`)

- One `call(role, prompt_id, variables, **params)` entry point. Roles: `proposer`, `editor`, `scorer`. Role→backend/model mapping from config.
- Cache: SQLite at `data/llm_cache/cache.db`, key = sha256(backend, model, prompt file hash, rendered prompt, params).
- Retries with exponential backoff on 429/5xx; per-backend concurrency limit.
- Ledger: every non-cached call appends (time, role, model, tokens in/cached/out, wall-clock GPU seconds) to `data/ledger/<run_id>.jsonl`.
- **Time cap**: a run has a `deadline` (end of the overnight window); the client stops issuing calls at the deadline, stages flush and exit cleanly, and the next night resumes from the cache.
- Default backend: local vLLM (installed: vllm 0.20.2) on 3×A100 80GB. A100 has no NVFP4 support, so checkpoints are BF16 or FP8 (weight-only on Ampere). Fireworks remains an optional backend for small comparisons.

### 5.2.1 Phase scheduling on local GPUs

Roles are not served concurrently. A run executes **phases**: each phase loads one model onto all 3 GPUs (as many data-parallel replicas as fit, tensor-parallel where needed), drains every pending request for that role across all items, writes results, and unloads. Items advance through a per-item state machine (`proposed → edited → scored → reproposed → … → done`), so the iterative steps of lane B (completeness re-proposal, leave-one-in) become extra rounds rather than interleaved calls. vLLM prefix caching makes repeated scorer calls on the same state cheap.

Typical night: generate responses → propose → edit → score (scorer 1) → score (scorer 2) → re-propose round → edit → score. The pilot measures items/hour per phase and the number of nights needed for the full run.

Host RAM is shared and was 937/1007 GB in use on 2026-10-06; each phase checks free host memory before loading and refuses to start below a configured floor.

### 5.3 Answer distributions (`llm/answer.py`)

The scorer role answers any typed question with a distribution over its allowed labels:

- Prompt lists the state, the question, and labels as single-token keys (`boolean`: `Y`/`N`; `choice`: `A`, `B`, …; `score`: level digits). Thinking disabled, `max_tokens=1`, `logprobs` ≥ number of labels (local vLLM returns any number; on Fireworks, which caps at 5, larger label sets are split into blocks and renormalized).
- Two scorers from different model families answer every lane B/C/D-ambiguous item (§5.5); `answer.dist` is their average, and both are kept in `meta.scorer_dists`.
- `dist = softmax over label-token logprobs`; mass on label tokens < 0.9 → one retry with a stricter prompt, then reject (`scorer_mass`).
- For `score`: $\hat{s} = \sum_k k\,p(k)$.
- Generated `choice` questions are capped at 8 options for teacher reliability; lane A choice sets keep their native size.
- Noise: logprob outputs at temperature 0 are near-deterministic, so σ is measured from **neutral control edits** (§8.3): the spread of $\hat{s}$ changes (score) or of $p(\text{label})$ changes (boolean/choice). Thresholds: $τ_{\text{score}} = \max(2σ_{\text{score}}, 0.25)$ level units; $τ_{p} = \max(2σ_p, 0.05)$ probability units. Measured per scorer in the pilot, then fixed in config.

### 5.4 Question bank (`questions/`)

Questions are not limited to the rubrics shipped with datasets. `data/registry/questions.jsonl`, one row per base question:

```json
{"id": "...", "type": "boolean|choice|score", "text": "...", "options": [...], "levels": [...],
 "family": "faithfulness|consistency|instruction_following|relevance|format|safety|style|intent|tone|policy|...",
 "domain": "see §5.6", "scope": "single|multi", "kind": "span|omission|cross_turn|evidence",
 "requires": ["context_docs", "prior_user_constraint", ...], "verifier": "llm|code",
 "source": "laneA_native|prometheus|soda_eval|mtbench101|healthbench|advancedif|rubrichub|wildbench|infobench|ifeval|taxonomy_gen|state_gen|handwritten",
 "instance_state_id": "set for instance-specific questions", "orig_text": "...",
 "paraphrases": ["...", "..."], "cluster_id": "...", "heldout": false, "qc": {"...": "§5.4.2 results"}}
```

#### 5.4.1 Five question sources

Counts measured 2026-10-06/07 from the downloaded data unless marked ≈ (100-row sample).

| # | Source | Raw distinct questions | Multi-turn | Notes |
|---|---|---|---|---|
| 1 | Dataset-native: lane A label sets, Prometheus (~1K rubrics, 5 level descriptions), Soda-Eval (17 types), MT-Bench-101 (13 abilities), handwritten multi-turn kinds (brief §7) | ~1–2K | partly | as in rev 2 |
| 2 | Mined instance rubrics: **HealthBench** (48,562 unique criteria, 5,000 conversations, 42% multi-turn, physician labels on responses in the meta-eval file, MIT), **AdvancedIF** (12,442 rubric questions; 9,450 on 1,243 multi-turn conversations; no responses; CC-BY-NC), **RubricHub_v1** (~2.58M unique criteria, single-turn: Chat 259K, Instruction-Following 94K, Writing 548K, Medical 889K, Science 791K; responses only in separate SFT files; Apache-2.0), WildBench checklists (≈23K), InFoBench (≈1.4K) | ~2.7M raw | ~25–30K native multi-turn | instance-specific; most need filtering (§5.4.2) |
| 3 | Taxonomy-generated: the editor writes questions for each cell of family × type × scope × domain × judged object (assistant turn, user turn, exchange, grounding docs) | configurable | yes | systematic coverage, including JEV-style operational questions (policy compliance, PII leakage, escalation needed, intent, sentiment, refusal type, brand voice) |
| 4 | State-conditioned: given a state, the editor writes 3–5 questions a real user might ask about it, including choice questions with state-specific options | unbounded | yes | always applicable; mirrors users writing their own questions |
| 5 | Code-checkable constraints: IFEval/IFBench instruction types with parameters (argilla ifeval-like 606K prompts, IF_multi_constraints 95K, IFBench multi-turn 3,161, Multi-IF 4,501 × 3 turns) | tens of types × parameters | yes | labels and spans computed by code (lane D, §8.4) |

RubricHub Medical/Science criteria are mostly knowledge-heavy (need domain facts to verify) and are expected to be filtered heavily; Chat, Instruction-Following, and Writing are the main RubricHub contribution. RubricHub is single-turn, so it feeds the pretrain stage and seeds multi-turn rewrites.

#### 5.4.2 Question quality control

Every question, mined or generated, passes these checks before use; results are stored in `qc`:

1. **Reading-checkable**: an LLM classifier labels whether the question can be answered from the state alone (no outside knowledge, proof, or code execution, per brief §10). Fail → drop.
2. **Unambiguous**: on a probe sample of up to 8 applicable states, both scorers agree on the label, and the label is stable across 3 paraphrases of the question. Agreement < 0.8 → drop.
3. **Non-trivial**: across the probe states each label occurs at least once (instance-specific questions are exempt; lane C supplies the other label by flipping).
4. **Choice sets**: options are distinct (pairwise embedding cosine < 0.9) and exhaustive (an "other/none" option is added unless the teacher's mass on existing options is ≥ 0.9 on all probe states).
5. **Rewrite to type form**: booleans and scores phrased so violations are highlightable; positive requirements become omissions.
6. **Dedup and cluster**: embed question text + options/levels, cluster at cosine ≥ 0.85, keep one representative per near-duplicate group.
7. **Hold-out**: ~15% of clusters stratified by type, family, scope, and domain (lane A native questions never held out).

Usage caps: each base question appears in at most 20 training records, so the model learns to read questions rather than memorize them. Target: **10–30K distinct base questions** in training, plus paraphrases.

`requires` drives applicability: lanes B/C/D only pair a question with a state that has the required features.

### 5.5 Label reliability

Each record carries `meta.reliability_tier`; the balance report and data card break down every metric by tier.

| Tier | Label from | Spans from | Examples |
|---|---|---|---|
| 0 | code | code | lane D constraints |
| 1 | humans | humans | RAGTruth, MQM, HalluDial, e-SNLI, HateXplain (tokens marked by ≥2 annotators) |
| 2 | humans | derived from human edits/tags | FAVA tags; FaithDial diffs only where the diff covers ≤ 40% of the turn |
| 3 | humans | teacher (counterfactual, constrained to the human label) | FaithDial originals (measured: median 57% of the turn changed by the human edit, 46% > 60%, so diffs are too coarse), HaluEval, BEGIN, DECODE, DBDC, HealthBench meta-eval physician labels, Prometheus |
| 4 | teacher | teacher | lane B on unlabeled states, lane C |

Checks:

- **Gold anchoring** (tier 3): lane B runs with the human label fixed; if the scorers' argmax disagrees with the human label, the item is rejected (`gold_disagree`) and counted; the rate is reported per source as a teacher-quality measure.
- **Two-scorer agreement** (tiers 3–4): two scorers from different model families must agree on the label and on the direction of every kept span's effect; disagreement → reject (`scorer_disagree`).
- **Counterfactual span test** and **control edits** (§8.1, §8.3).
- **Lane C shortcut test**: a quick classifier (TF-IDF + logistic regression, and a small encoder) tries to tell injected turns from originals without the question. Accuracy well above chance means the injections leave artifacts. Mitigation built in: the clean member of each pair receives a neutral paraphrase edit of similar size, so "was edited" is not a cue.
- **Human-agreement calibration**: teacher distributions are compared with multi-annotator label distributions (HateXplain 3 annotators, SNLI dev/test 5 annotators; ChaosNLI, ~100 annotations per item, added as a calibration-only eval set).
- **Human audit** (pilot): ~50 records per lane × question type, ~300 total; pass thresholds: label precision ≥ 90%, span precision ≥ 80% (span marks real evidence), span recall ≥ 70%. Full run only after passing.

### 5.6 Domains

Multi-domain by decision. Every state gets `meta.domain` from an LLM classifier (spot-checked) over: general assistant, customer support / agents, RAG and document QA, healthcare, education / tutoring, creative writing, professional / business writing, coding help (surface-checkable questions only), science and technical explanation, legal / finance (advice-cautious), safety / moderation, social chit-chat. Mixing constraints in §10: each domain between 3% and 15% of training records; taxonomy- and state-generated questions are sampled to fill under-represented domains.

## 6. Stage 0–1: acquisition and split

### 6.1 Acquisition

Each source module downloads to `data/raw/<source>/`, pins the HF revision or git commit, and records sha256 + license in the manifest. Gated sources (LMSYS-Chat-1M) require the user to accept terms on HF first; acquisition fails with a clear message otherwise.

The verification report (deliverable 1) is generated from the manifests plus a per-source probe: row counts per split, fields, a sample row, license, and mismatches against the brief. Known items from the 2026-10-06 HF probe:

- On HF: RAGTruth (third-party `wandb/RAGTruth-processed`; prefer original GitHub), WMT MQM (`RicardoRei/wmt-mqm-error-spans`, Apache-2.0), FAVA, Prometheus Feedback Collection, HalluDial (`FlagEval/HalluDial`, CC-BY-NC-SA, viewer unavailable), FaithDial (MIT), HaluEval, Dialogue NLI, e-SNLI, HateXplain (CC-BY), ERASER subsets, CoQA, QuAC, Doc2Dial, MultiDoc2Dial, Soda-Eval (`Johndfm/soda_eval`, CC-BY), WildChat (ODC-BY), UltraChat.
- Question and multi-turn sources added in rev 3 (measured 2026-10-06/07): HealthBench (`openai/healthbench`, MIT; 5,000 conversations, 42% multi-turn, 48,562 unique criteria, physician-labeled meta-eval file), AdvancedIF (`facebook/AdvancedIF`, CC-BY-NC; 1,645 conversations, 1,243 multi-turn, 12,442 rubric questions, no responses), RubricHub_v1 (`sojuL/RubricHub_v1`, Apache-2.0; 181K prompts with rubrics, ~2.58M unique criteria, single-turn), WildBench (2,304 tasks with checklists), InFoBench (500), IFBench multi-turn (3,161), IF_multi_constraints_upto5 (95,373), argilla ifeval-like-data (606,339), Multi-IF (4,501 × 3 turns, 8 languages, CC-BY-NC), ChaosNLI (calibration eval only). Multi-turn pools: WildChat-1M 837,989 rows (≈49% multi-turn), UltraChat 200k (~200K unique, avg 3.5 user turns), Soda-Eval (10,000 dialogues, 122,648 turn assessments, 19% with issues), FaithDial (4,094 train dialogues, 18,357 turns).
- Not on HF (check GitHub): DECODE, DiaHalu, DialFact, CI-ToD, BEGIN, DBDC, FED, DIAL-SUMMER, MT-Bench-101, TofuEval (`Dornavineeth/TOFUEval` is the unrelated TOFU unlearning set), cLang-8 (needs Lang-8 by request).

### 6.2 Conversation grouping

Several sources share upstream dialogues (FaithDial, BEGIN and likely HalluDial derive from Wizard-of-Wikipedia; DECODE and Dialogue NLI overlap PersonaChat-style data; confirmed during verification). Splitting per dataset would leak.

- Fingerprint every conversation from every acquired source: normalized hash of each turn, plus MinHash (5-gram shingles, 128 perms) of concatenated turns and of any context doc.
- Union-find groups: join if ≥2 shared exact turn hashes, or MinHash Jaccard ≥ 0.8 on turns, or an identical context doc plus one shared turn.
- Split is assigned per group, never per record.

### 6.3 Evaluation split

Assigned before any lane runs:

- `eval_official`: official test splits of RAGTruth, FaithDial, HalluDial, e-SNLI, HateXplain, ERASER subsets, Dialogue NLI; DECODE test; 10% of WMT MQM by (language pair, document). Any group touching these is eval for all sources.
- `eval_lane`: 5% of groups from raw-conversation sources (WildChat, UltraChat, Doc2Dial, Soda-Eval, Prometheus), so lanes B/C generate eval records with the same procedure as train.
- `eval_heldout_questions`: lane B/C records whose question is in a held-out cluster, generated only from eval groups.

Registry outputs: `groups.parquet`, `questions.jsonl`, and a frozen `REGISTRY_VERSION` hash recorded in every record's meta.

## 7. Stage 2: Lane A converters

Contract: `convert(raw_row) -> list[Record] | Reject`. Each converter registers its native questions (`source: laneA_native`), maps native labels to the question type, emits records without spans where the source has them, and keeps original labels in `meta.orig`. Lane A answers get `gold_label` from humans and `dist_source: smoothed_gold` (ε = 0.05) until the teacher-distribution pass (§10 step 3) adds a teacher distribution.

`score` severity defaults below are replaced after the calibration run (§9) by the median lane-B leave-one-in Δ on that source's matched spans; the mapping version is recorded in meta. Lane A `boolean`/`choice` span weights default to 1/(number of spans) and are replaced by teacher-measured weights where the teacher-distribution pass runs interventions (pilot decides whether that is worth the cost).

Priority sources (brief §9.9 plus classification sources):

| Source | Turns | Type(s) | State / target | Spans | No-span records | Defaults |
|---|---|---|---|---|---|---|
| HalluDial (P1) | multi | `boolean` "contains hallucination?" + `score` faithfulness | response in dialogue; knowledge as doc | localization field located in response | non-hallucinated samples | severity 2.0 |
| FaithDial (P2) | multi | `boolean` faithful? + `choice` BEGIN category | `original_response` as target; knowledge doc | `char_diff(original_response, response)`; >60% of turn changed → no span record | edited `response` | severity 2.0 |
| DECODE (P3) | multi | `boolean` "contradicts an earlier turn?" | contradicting utterance; prior dialogue | whole utterance or annotated clause; `evidence` = conflicting earlier utterances | non-contradicting variants | severity 2.5 |
| RAGTruth (P5) | single | `boolean` + `score` faithfulness; `choice` hallucination type | query as user, output as assistant, source as doc | labeled spans | `hallucination_labels == []` | evident 2.0, subtle 1.0 |
| WMT MQM (P5) | single | `score` per MQM category | "Translate: <src>" as user; MT as assistant | annotation offsets | no-error segments (capped) | minor 0.5, major 2.0, critical 4.0; omission errors → `omission.flag` |
| FAVA | single | `boolean` + `choice` hallucination type | prompt as user + references as docs | keep `<delete>` text, drop `<mark>` text; tagged regions are spans | corrected version | severity 1.5 |
| e-SNLI | single | `choice` entail/neutral/contradict | premise as doc, hypothesis as assistant turn | highlighted hypothesis tokens; premise highlights → `evidence` | none (every pair has a label) | weight split |
| HateXplain | single | `choice` hate/offensive/normal | post as assistant turn | rationale tokens (majority of annotators) | `normal` posts | weight split |
| ERASER FEVER, BoolQ | single | `choice` supports/refutes; `boolean` yes/no | claim/question as user, evidence doc | rationale spans are in the doc → `evidence`; target-turn spans empty | — | — |
| Dialogue NLI | multi | `choice` entail/neutral/contradict vs persona | utterance as target; persona sentences as doc | persona sentence → `evidence`; target utterance span | neutral pairs | weight split |
| CI-ToD | multi | `choice` inconsistency type (history/query/KB/none) | system turn as target | conflicting turn or KB row → `evidence` | consistent turns | weight split |

Per-source record caps in config (initial: MQM 6K, FAVA 4K, RAGTruth 5K, HalluDial 5K, FaithDial 3K, DECODE 3K, e-SNLI 2K, HateXplain 2K, ERASER 2K, Dialogue NLI 2K, CI-ToD 1K ≈ 35K before mixing) keep any one source from dominating.

## 8. Stage 3: Lanes B, C, and D

### 8.0 Response generation

Several question sources come without responses (AdvancedIF, RubricHub prompt files, WildBench, InFoBench, taxonomy- and state-generated questions on truncated conversations). A response-generation phase produces target turns with local models, deliberately spread in quality so both labels occur: 2–3 different generator models, plus a "careless" system prompt variant (ignore one constraint, be verbose, skip part of the request) for a configurable share. The generator and its variant go to `meta.generator`. Generated responses enter lanes B/C/D like any other target turn; they never carry labels themselves.

### 8.1 Lane B: counterfactual filtering

Input pool: Soda-Eval turns (issue labels as hints for question choice), Prometheus Feedback Collection (rubric given), WildChat multi-turn conversations, grounded dialogues (FaithDial, Doc2Dial) for faithfulness questions. Per item: sample 2–4 applicable non-held-out questions over the same state (JEV-style multi-question states), mixing types.

Notation: for a question with answer label $y$, the **signal** is $v(r) = \hat{s}(r)$ for `score` questions and $v(r) = p_y(r)$ for `boolean`/`choice`; threshold $τ$ is $τ_{\text{score}}$ or $τ_p$ accordingly. "Neutralize" means **repair** for score violations (minimal rewrite that fixes the violation) and **neutral rewrite** for boolean/choice evidence (minimal rewrite that removes the cue for $y$ while keeping fluency). Deletion is used only when a rewrite is impossible.

Per (state, question):

1. **Propose** (proposer): answer label $y$, spans as verbatim quotes with `conflicts_with?: {turn, quote}`, optional omission description. Unlocatable quotes are dropped (the item continues); overlapping spans are merged.
2. **Answer the original** with both scorers: `dist(r)`. If the item has a human label (tier 3), $y$ := human label and the item is rejected (`gold_disagree`) when the scorers' argmax differs; otherwise $y$ := scorer argmax, rejected (`teacher_disagree`) if it differs from the proposer's label or between scorers. Rejections are logged for analysis.
3. **No-span case**: for `score` with $\hat{s} ≥ ℓ_{\max} - τ$, or for boolean/choice "nothing present" labels, with no proposed spans → record without spans (subject to the no-span quota).
4. **Neutralize each span** (editor). Validate with `char_diff`: all changes within the span snapped to word boundaries ±10 chars; otherwise one retry, then drop the span (`edit_leak`).
5. **Leave-one-out**: $Δ_i = |v(r_{-i}) - v(r)|$ in the direction away from $y$ (score: increase; boolean/choice: decrease of $p_y$).
6. **Neutral baseline**: $r_0$ = all accepted edits applied; answer it.
7. **Completeness** (`score`): if $\hat{s}(r_0) < ℓ_{\max} - τ$, rerun Propose once on $r_0$, neutralize new spans, recompute; still below → reject (`incomplete`) unless explained by a verified omission (step 10). For boolean/choice: if $p_y(r_0)$ is still above 0.5, the spans do not explain the label → reject (`incomplete`).
8. **Leave-one-in**: $r_{0+i}$ = $r_0$ with only span $i$ restored; $Δ_i^{in} = |v(r_{0+i}) - v(r_0)|$ toward $y$.
9. **Select**: keep span $i$ if $Δ_i > τ$ or $Δ_i^{in} > τ$. `severity` (score) or `weight` (boolean/choice) = $Δ_i^{in}$ clipped to range, falling back to $Δ_i$ when only leave-one-out passes. Both deltas go to `meta.deltas`.
10. **Omission** (`score` only): if declared, the editor appends the missing part (the only edit allowed outside spans); if $\hat{s}$ rises by > τ, set `omission = {flag: true, severity: measured rise}`.
11. **Cross-turn evidence**: locate `conflicts_with` quotes in the referenced earlier turn; unlocatable → keep the span, drop the evidence, `meta.evidence_missing: true`.
12. **Consistency check** (`score`): `|span_score - ŝ(r)|` stored in meta; reject if > 1.5 (spans do not explain the judgment).
13. **Answer**: `answer.dist = dist(r)`, `dist_source: teacher`.

Calls per (state, question) with n spans: 1–2 proposals, n+1 edits, 2n+3 scorer calls. Questions over the same state share the cached conversation prefix.

### 8.2 Lane C: error injection and label flipping

Input pool: WildChat (English, not toxic, not redacted, 2–12 turns), UltraChat, and grounded dialogues (Doc2Dial/MultiDoc2Dial, FaithDial edited responses, CoQA/QuAC) for questions requiring `context_docs`. LMSYS-Chat-1M optional (gated).

Per item:

1. **Pick** (state, target turn t, question q) with q's `requires` satisfied. For `cross_turn` questions, the proposer first returns an **anchor** quote in an earlier turn (user constraint, standing instruction, or assistant claim); no locatable anchor → skip.
2. **Truncate** the conversation after turn t.
3. **Answer the original** with the scorer. Determine the source label $y_0$ (score: must be ≥ $ℓ_{\max} - τ$; boolean/choice: argmax with $p ≥ 0.7$), else skip.
4. **Inject** (editor): minimal rewrite so the answer becomes a chosen target label $y_1 ≠ y_0$ (score: violate the rubric to reach a lower level; boolean: flip; choice: switch to a sampled other option). Cross-turn questions: contradict or ignore the anchor.
5. **Diff**: `char_diff(r, r')` → spans. Reject if > 2 changed regions, changed chars > 40% of the turn, or length change > 30% (`inject_too_broad`).
6. **Verify**: answer $r'$; require the scorer's argmax to be $y_1$ (boolean/choice) or $\hat{s}$ to drop by > τ (score), else reject (`inject_no_effect`). Severity/weight = measured change; with 2 regions, split by leave-one-in from $r'$ and normalize.
7. **Omission questions**: the editor removes the required part; no spans, `omission.flag = true`.
8. **Emit** the injected record and the original (same state up to t, same question) as a pair, both with teacher `answer.dist`. The original carries spans only if its label has positive evidence (boolean/choice); score originals are no-span. Cross-turn records carry the anchor as `evidence`.

Lane C question mix (initial, config): 40% multi-turn kinds, 25% faithfulness to context, 15% choice sets (tone, intent, strategy), 10% relevance, 10% omission.

### 8.3 Control paraphrases (shared, during lanes B/C)

For 5% of lane B/C items, the editor paraphrases a random region outside the selected spans, of similar length, keeping meaning; the scorer re-answers. The distribution of control changes gives σ per signal (§5.3) and the false-positive rate (share of controls exceeding τ). Controls are logged to `data/controls/`, never emitted as records.

### 8.4 Lane D: code-verified constraints

For code-checkable questions (`verifier: code`), labels and spans come from deterministic checkers, not from an LLM (reliability tier 0).

- Checkers implement the IFEval/IFBench instruction types used (length limits, casing, punctuation bans, keyword inclusion/exclusion, bullet/section/JSON format, language, stop-word ratio, start/end phrases, …). Each checker returns the label and the offending character ranges (e.g. every comma when commas are banned; the sentences past a word limit; the first non-JSON character). Constraints that have no localizable violation (e.g. "fewer than N words" when the response is merely too long) mark the overflow region; constraints violated by absence ("must include keyword X") set `omission.flag`.
- Multi-turn: a constraint stated in an earlier user turn (Multi-IF, IFBench multi-turn, or injected into WildChat/UltraChat states as a standing instruction) is checked against later assistant turns; the constraint text goes to `evidence` (`where: turn:k`).
- Question text is generated from the constraint (paraphrased), so lane D questions are not templated.
- Responses come from §8.0; label balance comes from generating with and without the careless variant, and from minimal programmatic edits that introduce a violation (e.g. inserting a comma), with the same paired-control safeguard as lane C.

## 9. Calibration check (before scaling)

Run lane B on a few hundred train items each from RAGTruth, WMT MQM, e-SNLI, and HateXplain (not eval), using each item's native question. Report:

- Character-level span P/R/F1 and span-level F1 (IoU ≥ 0.5, any-overlap) against human spans.
- Teacher label accuracy vs gold labels, and teacher calibration (ECE, Brier) on those labels.
- Correlation of lane B severities with MQM severities.
- Per-source median leave-one-in Δ on matched spans → lane A defaults (§7).

Results go to `reports/calibration.md`. The full lane B run does not start until the user has reviewed them.

## 10. Stage 4–5: merge, filters, export

1. **Validate** every record (§4). Failures → rejects.
2. **Dedup** on (state hash, target_turn, question base id), keeping the higher-reliability record (tier 0 > 1 > 2 > 3 > 4).
3. **Teacher distributions** for lane A: one scorer call per record adds `meta.teacher_dist` (gold stays in `answer.gold_label`). Used for calibration analysis and optionally as soft targets.
4. **Length**: serialize as the encoder will see it (question + options/levels + docs + conversation), count tokens with the ModernBERT tokenizer, budget 8,192. Over budget: drop oldest turns first (re-index `target_turn` and `turn:k`); reject if an evidence turn would be dropped; trim docs from the end only if no evidence span is cut, else reject.
5. **Quotas**: no-span share ~1/3 overall; question-type mix initial target boolean 40% / score 35% / choice 25%; per-source caps; each domain 3–15% (§5.6); each base question ≤ 20 records; multi-turn (finetune stage) ≥ 50% of records.
6. **Mix**: sample to lane targets (initial: A 25%, B 20%, C 40%, D 15% of 100K), keeping states with multiple questions intact.
7. **Balance report** (`reports/balance.md` + JSON): counts per lane, source, question type, family, question, stage (pretrain/finetune), span vs no-span vs omission, language, license, questions per state; label distribution per question type; domain; reliability tier; question source (§5.4.1); reject counts per reason; severity/weight histograms; distinct question count; control false-positive rate; GPU-hours per phase from the ledger.
8. **Export**: `train.jsonl`, `eval_official.jsonl`, `eval_lane.jsonl`, `eval_heldout_questions.jsonl`, `DATA_CARD.md` (sources, licenses, procedures, prompt versions, models, τ values, known limitations). Final assertions: zero group overlap between train and eval, zero held-out question ids in train, every record valid.

## 11. Testing

- Property tests (Hypothesis) for `locate`, `char_diff`, `apply_edits`, offset maps, and `schema.validate`: random edits round-trip; re-located spans equal the edited text.
- `answer.py` tests: distribution extraction for each question type from canned logprob payloads, including >5-label blocks and low-mass rejection.
- Converter tests: each converter on `tests/fixtures/<source>/` must produce the expected records exactly.
- Lane B/C tests with a fake LLM backend covering: unlocatable quotes, edit leaks, saturation (two redundant errors recovered by leave-one-in), incomplete proposals, teacher disagreement, injections with no effect, choice-label flips.
- Registry tests: synthetic near-duplicate conversations across two sources land in the same group.
- Export invariant tests run on every export.

## 12. Pilot and rollout

1. Acquire priority sources (§7 plus HealthBench, AdvancedIF, RubricHub Chat/IF/Writing, Multi-IF, IFBench multi-turn, WildChat, UltraChat); write the verification report.
2. Build registry, question bank with QC (§5.4.2), and domain tags.
3. Lane A converters for §7 priority sources; lane D checkers for the constraint types present in the data.
4. Model line-up: benchmark local candidates on 3×A100 for throughput per phase, label-token mass, σ/τ on ~200 control edits per question type, and edit acceptance; pick the proposer/editor and the two scorers.
5. Calibration check (§9), gold-anchoring disagreement rates (§5.5), lane C shortcut test.
6. Pilot: ~1K records per lane (one overnight run), balance report, control false-positive rate, GPU-hours, projected nights for the full run.
7. Human audit of ~300 pilot records (§5.5).
8. **User review.** Full run only after approval.

## 13. Open items needing user decision

1. **Human audit**: who can review ~300 pilot records (~2–3 hours) on a simple review page.
2. **Full-run duration**: how many overnight runs are acceptable (estimated after the pilot).
3. **Non-English** default (kept, pretrain only; Multi-IF adds 7 non-English languages) is acceptable or should be excluded.
4. **Starting mixes**: question types (boolean 40 / score 35 / choice 25), lanes (A 25 / B 20 / C 40 / D 15), multi-turn ≥ 50%.
