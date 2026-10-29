"""Print the comparison tables for the README and model card.

    python -m api.summary

Reads the committed checkpoints, so the tables in the documentation are the
same numbers the models actually produced -- not retyped from a terminal
scroll, which is how a report ends up disagreeing with its own artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib

MODELS_DIR = Path("models")


def load_all() -> Dict[str, dict]:
    out = {}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        out[path.stem] = joblib.load(path)
    return out


def headline(ckpts: Dict[str, dict]) -> None:
    rows = sorted(ckpts.items(),
                  key=lambda kv: -kv[1]["metrics"]["test_accuracy"])
    print("| run | type | test acc | macro F1 | params | best epoch "
          "| train (s) | explainable |")
    print("|---|---|---|---|---|---|---|---|")
    for name, c in rows:
        print(f"| `{name}` | {c['model_type']} "
              f"| {c['metrics']['test_accuracy']:.4f} "
              f"| {c['metrics']['macro_f1']:.4f} "
              f"| {c['n_params']:,} | {c['best_epoch']} "
              f"| {c['train_seconds']:.0f} "
              f"| {'yes' if c['model_type'] == 'attention' else 'no'} |")


def per_class(ckpts: Dict[str, dict], names: List[str]) -> None:
    present = [n for n in names if n in ckpts]
    if len(present) < 2:
        return
    a, b = present[0], present[1]
    print(f"\n| class | {a} P / R / F1 | {b} P / R / F1 | support |")
    print("|---|---|---|---|")
    for ca, cb in zip(ckpts[a]["per_class"], ckpts[b]["per_class"]):
        print(f"| {ca['class_name']} "
              f"| {ca['precision']:.3f} / {ca['recall']:.3f} / {ca['f1']:.3f} "
              f"| {cb['precision']:.3f} / {cb['recall']:.3f} / {cb['f1']:.3f} "
              f"| {ca['support']:,} |")


def curves(ckpts: Dict[str, dict]) -> None:
    for name, c in ckpts.items():
        print(f"\n{name}: ", end="")
        print("  ".join(f"e{h['epoch']} val {h['val_acc']:.4f}"
                        for h in c["history"]))
        over = [h for h in c["history"]
                if h["train_acc"] - h["val_acc"] > 0.10]
        if over:
            print(f"  train/val gap exceeds 10 pts from epoch "
                  f"{over[0]['epoch']} -- memorisation begins here")


def confusion(ckpts: Dict[str, dict]) -> None:
    for name, c in ckpts.items():
        cm = c["confusion"]
        labels = c["class_names"]
        print(f"\n{name} confusion (rows = true):")
        print(f"  {'':<10}" + "".join(f"{l:>10}" for l in labels))
        for i, row in enumerate(cm):
            print(f"  {labels[i]:<10}" + "".join(f"{v:>10,}" for v in row))


def main() -> None:
    ckpts = load_all()
    if not ckpts:
        raise SystemExit("no checkpoints in models/")

    print("## Headline\n")
    headline(ckpts)

    print("\n## Per class\n")
    per_class(ckpts, ["lstm", "attention"])

    print("\n## Validation curves")
    curves(ckpts)

    print("\n## Confusion")
    confusion(ckpts)

    if "attention" in ckpts and "lstm" in ckpts:
        a, l = ckpts["attention"], ckpts["lstm"]
        gap = (a["metrics"]["test_accuracy"]
               - l["metrics"]["test_accuracy"]) * 100
        speed = l["train_seconds"] / max(a["train_seconds"], 1e-9)
        print(f"\n## For the report")
        print(f"  accuracy gap      {gap:+.2f} points (attention - lstm)")
        print(f"  training speedup  {speed:.1f}x")
        print(f"  parameter delta   {a['n_params'] - l['n_params']:,}")
    if "attention_no_pos" in ckpts and "attention" in ckpts:
        d = (ckpts["attention"]["metrics"]["test_accuracy"]
             - ckpts["attention_no_pos"]["metrics"]["test_accuracy"]) * 100
        print(f"  positional encoding worth {d:+.2f} points")


if __name__ == "__main__":
    main()