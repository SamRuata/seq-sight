"""Find where the attention model is confidently wrong, and freeze the evidence.

    python -m api.failures

Scores the held-out test split with both models and writes
data/failures.json: the examples the Streamlit Failure Gallery displays, each
with its tokens and attention weights already computed. Precomputed rather
than scored live because the free Render tier should not run 5,000 forward
passes to render a tab.

These rows are public IMDB text, so the file carries the review text itself --
unlike the `predictions` table, which holds a hash of what a visitor typed and
nothing else. The corpus is published; a visitor's input is not.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from dotenv import load_dotenv

from api.models import build_model
from shared.text import CLASS_NAMES, encode, fetch_split, load_vocab, pad_batch

MODELS_DIR = Path("models")
OUT = Path("data/failures.json")

N_CONFIDENT_WRONG = 6      # attention model wrong, and sure of itself
N_DISAGREE = 4             # the two models split


def load_model(name: str):
    ckpt = joblib.load(MODELS_DIR / f"{name}.joblib")
    model = build_model({**ckpt["config"], "vocab_size": ckpt["vocab_size"]})
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_all(model, encoded: List[List[int]], batch_size: int = 256):
    """Return ``(pred [N], confidence [N])`` over the whole split."""
    preds, confs = [], []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        ids, lengths, mask = pad_batch(chunk)
        logits, _, _ = model(ids, lengths, mask)
        probs = torch.softmax(logits, dim=1)
        preds.append(probs.argmax(dim=1).numpy())
        confs.append(probs.max(dim=1).values.numpy())
    return np.concatenate(preds), np.concatenate(confs)


@torch.no_grad()
def explain(model, ids: List[int], tokens: List[str]) -> Dict:
    batch_ids, lengths, mask = pad_batch([ids])
    logits, weights, _ = model(batch_ids, lengths, mask)
    probs = torch.softmax(logits, dim=1)[0]
    label = int(probs.argmax())
    attn = (weights[0][:len(tokens)].tolist() if weights is not None
            else [1.0 / max(1, len(tokens))] * len(tokens))
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": float(probs[label]),
        "attention": [round(float(w), 5) for w in attn],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", default="attention")
    ap.add_argument("--baseline", default="lstm")
    args = ap.parse_args()

    load_dotenv()
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"],
                           os.environ["SUPABASE_SERVICE_KEY"])

    texts, labels = fetch_split(client, "test")
    y = np.asarray(labels)
    vocab = load_vocab()
    encoded, token_lists = [], []
    for t in texts:
        ids, toks = encode(t, vocab)
        encoded.append(ids)
        token_lists.append(toks)
    print(f"test rows {len(texts):,}")

    attn_model, attn_ckpt = load_model(args.attention)
    base_model, base_ckpt = load_model(args.baseline)

    attn_pred, attn_conf = score_all(attn_model, encoded)
    base_pred, base_conf = score_all(base_model, encoded)

    print(f"\n{'model':<12}{'accuracy':>10}{'macro F1':>10}"
          f"{'params':>12}{'train s':>10}")
    for name, ckpt, pred in ((args.attention, attn_ckpt, attn_pred),
                             (args.baseline, base_ckpt, base_pred)):
        print(f"{name:<12}{float((pred == y).mean()):>10.4f}"
              f"{ckpt['metrics']['macro_f1']:>10.4f}"
              f"{ckpt['n_params']:>12,}{ckpt['train_seconds']:>10.0f}")

    agree = int((attn_pred == base_pred).sum())
    print(f"\nthe models agree on {agree:,} of {len(y):,} "
          f"({100 * agree / len(y):.1f}%)")
    print(f"attention right where baseline wrong: "
          f"{int(((attn_pred == y) & (base_pred != y)).sum()):,}")
    print(f"baseline right where attention wrong: "
          f"{int(((base_pred == y) & (attn_pred != y)).sum()):,}")

    # --- pick the examples ------------------------------------------------
    wrong = np.where(attn_pred != y)[0]
    wrong = wrong[np.argsort(-attn_conf[wrong])]          # most confident first

    # Balance the two error directions so the gallery does not show six
    # false negatives and call it a survey of the model's failures.
    chosen_wrong, seen = [], {0: 0, 1: 0}
    for i in wrong:
        direction = int(attn_pred[i])
        if seen[direction] >= N_CONFIDENT_WRONG // 2:
            continue
        seen[direction] += 1
        chosen_wrong.append(int(i))
        if len(chosen_wrong) == N_CONFIDENT_WRONG:
            break

    disagree = np.where(attn_pred != base_pred)[0]
    disagree = disagree[np.argsort(-np.abs(attn_conf[disagree]
                                           - base_conf[disagree]))]
    chosen_disagree = [int(i) for i in disagree[:N_DISAGREE]]

    examples = []
    for bucket, idxs in (("confident_wrong", chosen_wrong),
                         ("disagreement", chosen_disagree)):
        for i in idxs:
            examples.append({
                "bucket": bucket,
                "index": i,
                "text": texts[i],
                "tokens": token_lists[i],
                "n_tokens": len(token_lists[i]),
                "truncated": len(token_lists[i]) >= attn_ckpt["max_len"],
                "true_label": int(y[i]),
                "true_class": CLASS_NAMES[int(y[i])],
                "attention": explain(attn_model, encoded[i], token_lists[i]),
                "baseline": explain(base_model, encoded[i], token_lists[i]),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "attention_model": args.attention,
        "baseline_model": args.baseline,
        "n_test": len(y),
        "attention_accuracy": float((attn_pred == y).mean()),
        "baseline_accuracy": float((base_pred == y).mean()),
        "examples": examples,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1000:.0f} KB), "
          f"{len(examples)} examples")

    print("\nconfidently wrong:")
    for ex in examples[:N_CONFIDENT_WRONG]:
        a = ex["attention"]
        top = sorted(zip(ex["tokens"], a["attention"]),
                     key=lambda p: -p[1])[:5]
        print(f"  true {ex['true_class']:<8} -> {a['class_name']:<8} "
              f"@{a['confidence']:.3f}   top tokens: "
              f"{', '.join(t for t, _ in top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())