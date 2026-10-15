"""Predictions and classification metrics of a joint-model checkpoint.

Usage:
  # predictions, one shard per GPU (resumable):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src .venv/bin/python -m spanjudge.eval_joint \
      predict --checkpoint runs/joint_v2/checkpoints/epoch-2 \
      --data data/splits/eval.jsonl --out runs/joint_v2/preds/eval --shard 0/2
  # metrics over all shards:
  PYTHONPATH=src .venv/bin/python -m spanjudge.eval_joint report \
      --out runs/joint_v2/preds/eval

Per dataset: accuracy, macro F1, per-class precision / recall / F1 and the
confusion matrix (gold rows, predicted columns), plus token-level span
precision / recall / F1 at a 0.5 threshold.

Faithfulness (ERASER, DeYoung et al. 2020), on rows where the model marks
at least one document token (p >= 0.5), for the predicted answer y:
sufficiency = p(y | full) - p(y | prompt with only the marked document
tokens), lower is better; comprehensiveness = p(y | full) - p(y | prompt
with the marked tokens removed), higher is better. A random baseline marks
the same number of random document tokens.
"""

import argparse
import collections
import glob
import json
import pathlib

import random

import torch

from spanjudge import train_joint


def _batches(exs: list, budget: int, cap: int = 64) -> list[list]:
    """Groups items sorted by length so a batch stays under `budget` tokens
    of padded input (and at most `cap` rows)."""
    exs = sorted(exs, key=lambda x: len(x[1]["input_ids"]))
    out, cur = [], []
    for x in exs:
        n = len(x[1]["input_ids"])
        if cur and ((len(cur) + 1) * n > budget or len(cur) == cap):
            out.append(cur)
            cur = []
        cur.append(x)
    return out + ([cur] if cur else [])


def predict(checkpoint: str, data: str, out: str, shard: str, order: str,
            batch_tokens: int = 8192, chunk: int = 512):
    """Writes one prediction per row of this shard to <out>/shard-i.jsonl.

    Rows are processed `chunk` at a time; within a chunk, rows and then
    their ablations run in length-sorted batches of up to `batch_tokens`
    padded tokens (train_joint.forward_batch).
    """
    from spanjudge.joint_viz import _Model

    i, n = map(int, shard.split("/"))
    with open(data) as f:
        rows = [json.loads(line) for line in f][i::n]
    path = pathlib.Path(out) / f"shard-{i}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
    rows = [r for r in rows if r["id"] not in done]
    model = _Model(checkpoint, order)
    pad = model.tok.pad_token_id if model.tok.pad_token_id is not None else 0

    def run(items):
        """{key: (answer logits, span logits, draft logits)} for (key, ex)."""
        res = {}
        with torch.no_grad():
            for b in _batches(items, batch_tokens):
                outs = train_joint.forward_batch(
                    model.model, model.head, [ex for _, ex in b], model.codes,
                    "cuda", pad)
                for (key, _), o in zip(b, outs, strict=True):
                    res[key] = o
        return res

    with open(path, "a") as f:
        for c in range(0, len(rows), chunk):
            part = rows[c:c + chunk]
            exs = {r["id"]: train_joint.encode(r, model.tok, None) for r in part}
            main = run(list(exs.items()))
            # Ablations rerun the whole model (and head) on the shortened
            # prompt, so they measure the answer the model reports; with the
            # evidence head also the draft (the LM letter SelfCite trains on).
            abl, info = [], {}
            for r in part:
                ex = exs[r["id"]]
                logits, span, draft = main[r["id"]]
                chosen = (torch.sigmoid(span) >= 0.5).cpu().tolist()
                n_sel = sum(chosen)
                info[r["id"]] = (chosen, n_sel)
                if not n_sel:
                    continue
                pick = set(random.Random(r["id"]).sample(range(len(chosen)), n_sel))
                rand = [k in pick for k in range(len(chosen))]
                for name, keep in (("only", chosen),
                                   ("drop", [not x for x in chosen]),
                                   ("only_random", rand),
                                   ("drop_random", [not x for x in rand])):
                    abl.append(((r["id"], name), train_joint.subset(ex, keep)))
            ablated = run(abl)
            for r in part:
                ex = exs[r["id"]]
                labels = list(r["labels"])
                logits, span, draft = main[r["id"]]
                probs = torch.softmax(logits, -1).tolist()
                y = max(range(len(probs)), key=probs.__getitem__)
                y_d = None if draft is None else int(draft.argmax())
                pred = (torch.sigmoid(span) > 0.5).cpu().tolist()
                gold = [t >= 0.5 for t, d in zip(ex["target"], ex["doc"],
                                                 strict=True) if d]
                chosen, n_sel = info[r["id"]]
                faith = {}
                if n_sel:
                    for name in ("only", "drop", "only_random", "drop_random"):
                        lg, _, dr = ablated[(r["id"], name)]
                        faith[f"p_{name}"] = torch.softmax(lg, -1)[y].item()
                        if dr is not None:
                            faith[f"pd_{name}"] = torch.softmax(dr, -1)[y_d].item()
                rec = {
                    "id": r["id"],
                    "source": r["source"],
                    "gold": r["label"],
                    "pred": labels[y],
                    "probs": dict(zip(labels, probs, strict=True)),
                    "span_tp": sum(a and g for a, g in zip(pred, gold, strict=True)),
                    "span_pred": sum(pred),
                    "span_gold": sum(gold),
                    **({"draft_probs": dict(zip(labels, torch.softmax(draft, -1).tolist(), strict=True))}
                       if draft is not None else {}),
                    "n_selected": n_sel,
                    "n_doc": sum(ex["doc"]),
                    **faith,
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def report(out: str) -> str:
    """Markdown metrics for all shards in <out>; also written to metrics.md."""
    preds = []
    for path in sorted(glob.glob(f"{out}/shard-*.jsonl")):
        with open(path) as f:
            preds += [json.loads(line) for line in f]
    by: dict[str, list] = collections.defaultdict(list)
    for p in preds:
        by[p["source"]].append(p)
    lines = [f"# Classification metrics: {out}", ""]
    summary = ["| dataset | rows | accuracy | macro F1 | span P | span R | span F1 | marked share | rows marked | sufficiency (rand) | comprehensiveness (rand) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    all_acc, all_macro = [], []
    for name in sorted(by):
        ps = by[name]
        classes = sorted({p["gold"] for p in ps} | {p["pred"] for p in ps})
        conf = collections.Counter((p["gold"], p["pred"]) for p in ps)
        acc = sum(p["gold"] == p["pred"] for p in ps) / len(ps)
        per = []
        for c in classes:
            tp = conf[(c, c)]
            pc = sum(conf[(g, c)] for g in classes)
            gc = sum(conf[(c, q)] for q in classes)
            prec = tp / pc if pc else 0.0
            rec = tp / gc if gc else 0.0
            per.append((c, gc, pc, prec, rec, _f1(prec, rec)))
        gold_classes = [x for x in per if x[1]]
        macro = sum(x[5] for x in gold_classes) / len(gold_classes)
        tp = sum(p["span_tp"] for p in ps)
        sp = sum(p["span_pred"] for p in ps)
        sg = sum(p["span_gold"] for p in ps)
        P, R = (tp / sp if sp else 0.0), (tp / sg if sg else 0.0)
        fr = [p for p in ps if "p_only" in p]

        def mean(xs):
            xs = list(xs)
            return sum(xs) / len(xs) if xs else float("nan")

        p_y = lambda p: max(p["probs"].values())
        share = mean(p["n_selected"] / max(1, p["n_doc"]) for p in ps)
        suff = mean(p_y(p) - p["p_only"] for p in fr)
        suff_r = mean(p_y(p) - p["p_only_random"] for p in fr)
        comp = mean(p_y(p) - p["p_drop"] for p in fr)
        comp_r = mean(p_y(p) - p["p_drop_random"] for p in fr)
        summary.append(
            f"| {name} | {len(ps)} | {acc:.3f} | {macro:.3f} | {P:.2f} | {R:.2f} | {_f1(P, R):.2f} "
            f"| {share:.2f} | {len(fr) / len(ps):.2f} | {suff:.3f} ({suff_r:.3f}) | {comp:.3f} ({comp_r:.3f}) |"
        )
        all_acc.append(acc)
        all_macro.append(macro)
        detail += [f"## {name}", "", "| class | gold | predicted | precision | recall | F1 |", "|---|---|---|---|---|---|"]
        detail += [f"| {c} | {g} | {q} | {p:.3f} | {r:.3f} | {f:.3f} |" for c, g, q, p, r, f in per]
        detail += ["", "Confusion (rows gold, columns predicted):", "", "| gold \\ pred | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
        detail += [f"| {g} | " + " | ".join(str(conf[(g, q)]) for q in classes) + " |" for g in classes]
        detail.append("")
    fr = [p for p in preds if "p_only" in p]
    py = lambda p: max(p["probs"].values())
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    summary.append(
        f"| **all** | {len(preds)} | {sum(all_acc) / len(all_acc):.3f} (mean) | {sum(all_macro) / len(all_macro):.3f} (mean) | | | | "
        f"{avg([p['n_selected'] / max(1, p['n_doc']) for p in preds]):.2f} | {len(fr) / len(preds):.2f} | "
        f"{avg([py(p) - p['p_only'] for p in fr]):.3f} ({avg([py(p) - p['p_only_random'] for p in fr]):.3f}) | "
        f"{avg([py(p) - p['p_drop'] for p in fr]):.3f} ({avg([py(p) - p['p_drop_random'] for p in fr]):.3f}) |"
    )
    fd = [p for p in preds if "pd_only" in p]
    if fd:
        pyd = lambda p: max(p["draft_probs"].values())
        summary += [
            "",
            "Draft answer (LM letter) faithfulness, same highlights: "
            f"sufficiency {avg([pyd(p) - p['pd_only'] for p in fd]):.3f} "
            f"(random {avg([pyd(p) - p['pd_only_random'] for p in fd]):.3f}), "
            f"comprehensiveness {avg([pyd(p) - p['pd_drop'] for p in fd]):.3f} "
            f"(random {avg([pyd(p) - p['pd_drop_random'] for p in fd]):.3f}); "
            f"draft accuracy {avg([max(p['draft_probs'], key=p['draft_probs'].get) == p['gold'] for p in preds]):.3f}",
        ]
    text = "\n".join(lines + summary + [""] + detail)
    (pathlib.Path(out) / "metrics.md").write_text(text)
    return text


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["predict", "report"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--data", default="data/splits/eval.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--batch-tokens", type=int, default=8192,
                        help="padded tokens per forward batch")
    parser.add_argument(
        "--order", default="question_first", choices=["doc_first", "question_first"]
    )
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args.checkpoint, args.data, args.out, args.shard, args.order,
                args.batch_tokens)
    else:
        print(report(args.out))


if __name__ == "__main__":
    main()
