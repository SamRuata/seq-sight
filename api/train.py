"""Train one configuration, score it, and persist the run.

    python -m api.train --config api/configs/lstm.yaml

Training rows are read from Supabase, as the brief requires. It runs locally
rather than inside the API because the free Render tier has neither the memory
nor the request timeout for it; the resulting checkpoint is committed to git
and loaded at startup.

Every run spends its whole epoch budget. There is no early stopping, because
two models that stop at different times are no longer a matched comparison --
only the best-validation weights are kept.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch import nn

from api.models import build_model, count_params
from shared.text import (CLASS_NAMES, MAX_LEN, N_CLASSES, VOCAB_PATH,
                         build_vocab, encode, fetch_split, load_vocab,
                         pad_batch, save_vocab)

MODELS_DIR = Path("models")

# PyTorch's CPU default is conservative, and an LSTM over long sequences is
# thread-starved without this.
torch.set_num_threads(max(1, os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# Metrics -- hand-rolled so the `api` package stays free of scikit-learn,
# which is not installed on Render.
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n: int = N_CLASSES) -> List[List[int]]:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm.tolist()


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict]:
    out = []
    for c in range(N_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append({
            "label": c,
            "class_name": CLASS_NAMES[c],
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "support": int((y_true == c).sum()),
        })
    return out


def macro_f1(per_class: List[Dict]) -> float:
    return float(np.mean([c["f1"] for c in per_class]))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def encode_split(texts: List[str], vocab: Dict[str, int]) -> List[List[int]]:
    return [encode(t, vocab)[0] for t in texts]


def batches(x: List[List[int]], y: np.ndarray, batch_size: int,
            generator: torch.Generator = None, bucket: bool = True):
    """Yield padded batches, grouping similar lengths when shuffling.

    With purely random batches almost every batch contains at least one
    full-length review, so everything pads to the cap and the LSTM walks
    hundreds of steps over zeros. Sorting by length first and then shuffling
    the ORDER of the batches keeps the randomness that matters -- which batch
    comes when -- while letting each batch pad only to its own longest review.
    """
    n = len(x)
    if generator is None:
        order = list(range(n))
    elif bucket:
        by_length = sorted(range(n), key=lambda i: len(x[i]))
        chunks = [by_length[s:s + batch_size] for s in range(0, n, batch_size)]
        perm = torch.randperm(len(chunks), generator=generator).tolist()
        order = [i for c in perm for i in chunks[c]]
    else:
        order = torch.randperm(n, generator=generator).tolist()

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        ids, lengths, mask = pad_batch([x[i] for i in idx])
        yield ids, lengths, mask, torch.tensor(y[idx], dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, x: List[List[int]], y: np.ndarray,
             batch_size: int = 256) -> Tuple[float, float, np.ndarray]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds = 0.0, []
    for ids, lengths, mask, yb in batches(x, y, batch_size):
        logits, _, _ = model(ids, lengths, mask)
        total_loss += float(loss_fn(logits, yb))
        preds.append(logits.argmax(dim=1).numpy())
    pred = np.concatenate(preds)
    return total_loss / len(y), float((pred == y).mean()), pred


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
def ensure_dataset_row(client, n_rows: int, vocab_size: int, seed: int) -> int:
    """`runs.dataset_id` is a NOT NULL foreign key, so one row must exist."""
    name = "imdb"
    found = (client.table("datasets").select("id").eq("name", name)
             .limit(1).execute())
    if found.data:
        return int(found.data[0]["id"])
    res = client.table("datasets").insert({
        "name": name, "n_rows": n_rows, "seq_len": MAX_LEN,
        "vocab_size": vocab_size, "n_classes": N_CLASSES, "seed": seed,
    }).execute()
    return int(res.data[0]["id"])


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-persist", action="store_true",
                    help="train and checkpoint, but write no Supabase row")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    name = config["name"]
    seed = int(config.get("seed", 42))
    batch_size = int(config["batch_size"])

    load_dotenv()
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"],
                           os.environ["SUPABASE_SERVICE_KEY"])

    print(f"[{name}] reading corpus from Supabase")
    train_texts, train_labels = fetch_split(client, "train")
    val_texts, val_labels = fetch_split(client, "val")
    test_texts, test_labels = fetch_split(client, "test")
    print(f"  train {len(train_texts):,}  val {len(val_texts):,}  "
          f"test {len(test_texts):,}")

    # Vocabulary is built from training rows only and shared by every run, so
    # the two models see an identical input space.
    if VOCAB_PATH.exists():
        vocab = load_vocab()
    else:
        vocab = build_vocab(train_texts)
        save_vocab(vocab)
    print(f"  vocabulary {len(vocab):,}   max_len {MAX_LEN}   "
          f"threads {torch.get_num_threads()}")

    x_train = encode_split(train_texts, vocab)
    x_val = encode_split(val_texts, vocab)
    x_test = encode_split(test_texts, vocab)
    y_train = np.asarray(train_labels)
    y_val = np.asarray(val_labels)
    y_test = np.asarray(test_labels)

    torch.manual_seed(seed)
    config["vocab_size"] = len(vocab)
    model = build_model(config)
    n_params = count_params(model)
    print(f"  {config['model_type']} with {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    history: List[Dict] = []
    best_state, best_acc, best_epoch = None, -1.0, 0
    started = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        epoch_started = time.time()
        running, seen, correct = 0.0, 0, 0

        for ids, lengths, mask, yb in batches(x_train, y_train, batch_size,
                                              generator):
            optimizer.zero_grad()
            logits, _, _ = model(ids, lengths, mask)
            loss = loss_fn(logits, yb)
            loss.backward()
            # Recurrent nets over long sequences can throw a huge gradient on a
            # single batch. The clip costs nothing and is applied to both
            # models, so it stays part of the matched setup.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += float(loss) * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum())
            seen += len(yb)
            if seen % (batch_size * 20) < len(yb):
                rate = seen / max(1e-6, time.time() - epoch_started)
                print(f"\r  epoch {epoch:>2}  {seen:>6,}/{len(x_train):,}"
                      f"  {rate:,.0f} rows/s", end="", flush=True)

        val_loss, val_acc, _ = evaluate(model, x_val, y_val)
        history.append({
            "epoch": epoch,
            "train_loss": running / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        flag = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"\r  epoch {epoch:>2}  train {running / seen:.4f}/"
              f"{correct / seen:.4f}   val {val_loss:.4f}/{val_acc:.4f}"
              f"   {time.time() - epoch_started:.0f}s{flag}")

    elapsed = time.time() - started
    model.load_state_dict(best_state)

    test_loss, test_acc, test_pred = evaluate(model, x_test, y_test)
    per_class = per_class_metrics(y_test, test_pred)
    f1 = macro_f1(per_class)

    print(f"\n[{name}] test accuracy {test_acc:.4f}   macro F1 {f1:.4f}   "
          f"{elapsed:.0f}s   best epoch {best_epoch}")
    for c in per_class:
        print(f"  {c['class_name']:<9} P {c['precision']:.3f}  "
              f"R {c['recall']:.3f}  F1 {c['f1']:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config,
        "model_type": config["model_type"],
        "n_params": n_params,
        "vocab_size": len(vocab),
        "max_len": MAX_LEN,
        "class_names": CLASS_NAMES,
        "metrics": {"test_accuracy": test_acc, "macro_f1": f1,
                    "test_loss": test_loss},
        "history": history,
        "confusion": confusion_matrix(y_test, test_pred),
        "per_class": per_class,
        "train_seconds": elapsed,
        "best_epoch": best_epoch,
    }
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(checkpoint, path)
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if args.no_persist:
        return 0

    dataset_id = ensure_dataset_row(client, len(x_train), len(vocab), seed)
    res = client.table("runs").insert({
        "dataset_id": dataset_id,
        "lr": float(config["lr"]),
        "batch_size": batch_size,
        "epochs": int(config["epochs"]),
        "accuracy": test_acc,
        "macro_f1": f1,
        "model_type": config["model_type"],
        "embed_dim": int(config["embed_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "n_heads": 1 if config["model_type"] == "attention" else None,
        "max_len": MAX_LEN,
        "vocab_size": len(vocab),
        "n_params": n_params,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_seconds": elapsed,
        "history": history,
        "confusion": checkpoint["confusion"],
    }).execute()
    print(f"  runs.id {res.data[0]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())