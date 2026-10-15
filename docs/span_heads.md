# Span head architectures

Both heads live in `src/spanjudge/train_joint.py` and sit on top of
Qwen3.5-2B (hidden size 2048) with LoRA. In both, the **answer is the LM
letter**: the next-token logits at the assistant position, read for the
choice letters only (`code_ids`, softmax over the row's choices). The span
head adds one **evidence logit per document token**; p(evidence) =
sigmoid(logit). `--head` picks the head; `--order` picks the prompt order.

Notation: n = document tokens, m = question-side tokens, k = answer choices,
H = 2048 (LLM hidden size), d = 512 (head width), 8 attention heads.

## Shared input: the prompt

```
<system> Classify the context using the question below ... one-letter code.
<user>
  question first (--order question_first):   Question / Choices, then [Part] document sections
  question at the end (--order doc_first):    [Part] document sections, then Question / Choices
  Answer with the letter only.
<assistant> <think>\n\n</think>\n\n  ← LM letter read here
```

`encode` maps every token to: document or not (`doc` mask), question side or
not (`question` mask), and its span target (`char_targets`: human spans 0/1;
synthetic spans = share of the 8 Qwen samples that marked it; rows without
evidence all 0).

---

## 1. `SpanHead` (`--head cross`, default)

Used by `joint_v2` and the 10K question-first runs. 3.4M parameters.

```
LLM last-layer states ─┬─ document tokens  D (n×H) ── Linear H→d ──┐
                       │                                            ├─ cross-attention: document → question (8 heads)
                       └─ question tokens  Q (m×H) ── Linear H→d ──┘          │
                                                                    residual + LayerNorm
                                                                              │
                                                       MLP d→d, GELU, d→1  →  evidence logit per document token
```

```
d   = W_d · D                      (n × d)
q   = W_q · Q                      (m × d)
ctx = MultiheadAttention(query=d, key=q, value=q)
logit = MLP(LayerNorm(d + ctx))    (n)
```

- Document tokens are the queries; question tokens the keys and values.
- With the question first, the document states already depend on the
  question inside the LLM, so this one layer mostly re-reads it.
- With the question at the end (the first, diverged run `joint_v1`), the
  document states never saw the question and this single layer was too weak.

Loss: cross-entropy on the LM letter + mean BCE over document tokens
(`losses`), weighted 1 : `--span-weight` (1.0).

---

## 2. `EvidenceHead` (`--head evidence`, needs `--order doc_first`)

Used by the overnight run and the 10K question-at-the-end runs. 23.1M
parameters: ~11.8M span part, ~11.3M reader (with `--reader-layers 2`).

Two LLM passes per row:

1. **Full prompt** (document first): last-layer states of document tokens
   `h_doc` (question-independent, cacheable) and of question-side tokens
   `h_q` (question, choices, tail; these have read the document), plus the
   **LM letter** ("draft") logits at the assistant position.
2. **Question only** (no document): states of the question tokens `i_q` and
   of each choice line, mean-pooled to one vector per choice `i_choices`.

### 2a. Span part

```
c   = W_iso · i_choices                          (k × d)   choice vectors (document-free)
mem = [ W_mem · h_q ;  Σ_k p_draft(k) · c_k ]     ((m+1) × d) question side + the draft as a soft choice mix
x   = W_doc · h_doc                              (n × d)

repeat 2 blocks (--head-layers):
    x = x + CrossAttn(LN(x), mem)      document tokens read the question, choices and draft
    x = x + SelfAttn(LN(x), LN(x))     document tokens read each other (bidirectional, head only)
    x = x + FFN(LN(x))                 d → 4d → d, GELU

logit = LN → Linear d→d → GELU → Linear d→1      (n)  evidence logit per document token
```

- The draft is the model's own predicted answer (soft), never the gold
  label, so there is no label leak.
- Self-attention inside the head gives document tokens bidirectional context
  that the causal LLM does not.

### 2b. Reader: the strict final answer (optional)

```
bias = log sigmoid(logit)                        log p(evidence) per document token
doc  = W_raw · h_doc                             raw document states (not the span blocks' output)
x_q  = W_iso · i_q                               question tokens encoded WITHOUT the document

repeat 2 reader layers (--reader-layers; 0 = one read by the mean question vector):
    x_q = x_q + CrossAttn(LN(x_q), doc, bias)    question tokens read the document, evidence-weighted
    x_q = x_q + SelfAttn(LN(x_q), LN(x_q))
    x_q = x_q + FFN(LN(x_q))

read  = mean(x_q);  query = mean(W_iso · i_q)
z     = MLP([read ; query])                      (d)
final = (z · c_k) / sqrt(d)                      (k)  final-answer logits
```

- Strict: the reader's queries and the choice vectors never saw the
  document; the document enters only through the evidence-biased reads. The
  draft and the span blocks' document states are not inputs.
- `CrossAttn` / `SelfAttn` are `_Attn`: four d×d projections and PyTorch's
  `scaled_dot_product_attention`; `bias` is added to the attention scores.

### Losses

```
loss = CE(final, gold)                              final answer
     + 1.0 · CE(LM letter, gold)                    --draft-weight
     + 1.0 · mean BCE(span logits, span targets)    --span-weight
     + 1.0 · KL(softmax(LM letter).detach() || softmax(final))   --distill-weight (training only)
     [+ SelfCite policy-gradient term]              --selfcite
```

The final-answer loss also trains the span head, through the evidence bias.

---

## Batched inference

`forward_batch` (`train_joint.py`) runs either head on several rows at once:
left padding, position ids that skip the padding (the linear-attention
layers zero padded inputs; full-attention layers get the padding mask), and
only the last layer's states are kept. `eval_joint` groups rows by length
under `--batch-tokens` (8,192 padded tokens).

## Results in brief

See `docs/learnings.md` (5.8-5.10). Question first with `SpanHead`: best
spans on long claim / multi-hop documents (`joint_v2` span F1 0.56). Question
at the end with `EvidenceHead`, label-balanced 86K data: best LM letter
(mean accuracy 0.836), span F1 0.54, strict reader 0.768.
