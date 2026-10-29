"""Load the IMDB review corpus into Supabase.

    python -m db.load                      # 10,000 train / 2,000 val / 5,000 test
    python -m db.load --replace            # wipe `sequences` first
    python -m db.load --limit 100          # quick smoke test
    python -m db.load --archive C:\\path\\aclImdb_v1.tar.gz

The brief requires the training corpus to live in Supabase and the trainer to
read it back out, so this is the only place raw IMDB text enters the system.
It is public, published data -- unlike a review a visitor types into the demo,
which is hashed and discarded. That asymmetry is deliberate and is the privacy
story the README tells.

Two sources are tried in turn because the canonical Stanford host is often
unreachable: the HuggingFace parquet mirror (~40 MB total) and the original
aclImdb tarball (84 MB). Downloads are cached under data/, which is git-ignored.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

HF_BASE = ("https://huggingface.co/datasets/stanfordnlp/imdb/"
           "resolve/main/plain_text")
HF_FILES = {"train": "train-00000-of-00001.parquet",
            "test": "test-00000-of-00001.parquet"}
STANFORD = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

DATA = Path("data")
N_TRAIN, N_VAL, N_TEST = 10_000, 2_000, 5_000
SEED = 42
BATCH = 500

# The corpus was scraped from HTML and still carries line-break tags. This is
# the only edit made to the review text; everything else is stored verbatim.
BR = re.compile(r"<br\s*/?>")

Corpus = Dict[str, List[Tuple[str, int]]]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch(url: str, dest: Path, min_bytes: int) -> bool:
    """Download to ``dest``, returning False instead of raising on failure."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  cached {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with dest.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    pct = (f"{100 * done / total:.0f}%" if total
                           else f"{done / 1e6:.0f} MB")
                    print(f"\r  {dest.name} {pct}", end="", flush=True)
        print()
        return True
    except Exception as exc:                                   # noqa: BLE001
        print(f"\r  {dest.name}: {type(exc).__name__}")
        dest.unlink(missing_ok=True)
        return False


def from_parquet() -> Optional[Corpus]:
    paths = {}
    print("trying the HuggingFace mirror")
    for split, name in HF_FILES.items():
        dest = DATA / name
        if not fetch(f"{HF_BASE}/{name}", dest, 5_000_000):
            return None
        paths[split] = dest

    import pandas as pd

    corpus: Corpus = {}
    for split, path in paths.items():
        df = pd.read_parquet(path)
        corpus[split] = [(BR.sub(" ", str(t)).strip(), int(l))
                         for t, l in zip(df["text"], df["label"])]
    print(f"  read {len(corpus['train']):,} train / {len(corpus['test']):,} test")
    return corpus


def from_tarball(path: Path) -> Corpus:
    """Stream the archive once rather than extracting 50,000 files to disk."""
    corpus: Corpus = {"train": [], "test": []}
    wanted = {
        "aclImdb/train/pos/": ("train", 1),
        "aclImdb/train/neg/": ("train", 0),
        "aclImdb/test/pos/": ("test", 1),
        "aclImdb/test/neg/": ("test", 0),
    }
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            for prefix, (split, label) in wanted.items():
                if member.name.startswith(prefix):
                    fh = tar.extractfile(member)
                    if fh is not None:
                        text = BR.sub(" ", fh.read().decode("utf-8", "replace")).strip()
                        if text:
                            corpus[split].append((text, label))
                    n = len(corpus["train"]) + len(corpus["test"])
                    if n % 5000 == 0:
                        print(f"\r  {n:,} reviews", end="", flush=True)
                    break
    print(f"\r  read {len(corpus['train']):,} train / {len(corpus['test']):,} test")
    return corpus


def acquire(archive: Optional[str]) -> Corpus:
    if archive:
        path = Path(archive)
        if not path.exists():
            raise SystemExit(f"no file at {path}")
        print(f"reading {path}")
        return from_tarball(path)

    corpus = from_parquet()
    if corpus is not None:
        return corpus

    print("mirror unavailable; trying the Stanford archive")
    dest = DATA / "aclImdb_v1.tar.gz"
    if fetch(STANFORD, dest, 50_000_000):
        return from_tarball(dest)

    raise SystemExit(
        "\nNeither source is reachable from this network.\n"
        "Download the archive by hand in a browser:\n"
        f"    {STANFORD}\n"
        "then re-run:\n"
        "    python -m db.load --archive C:\\Users\\zinmo\\Downloads\\aclImdb_v1.tar.gz"
    )


# ---------------------------------------------------------------------------
# Splitting and loading
# ---------------------------------------------------------------------------
def take_balanced(rows: List[Tuple[str, int]], n: int,
                  rng: random.Random, skip: int = 0) -> List[Tuple[str, int]]:
    """Return ``n`` rows, half per label, after a deterministic shuffle.

    ``skip`` discards that many per class first, which is how validation is
    kept disjoint from training without a second shuffle.
    """
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    chosen = pos[skip:skip + half] + neg[skip:skip + half]
    if len(chosen) < n:
        raise SystemExit(f"corpus too small: wanted {n}, got {len(chosen)}")
    rng.shuffle(chosen)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--archive", default=None,
                    help="path to a manually downloaded aclImdb_v1.tar.gz")
    args = ap.parse_args()

    load_dotenv()
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"],
                           os.environ["SUPABASE_SERVICE_KEY"])

    existing = client.table("sequences").select("id", count="exact").limit(1).execute()
    if existing.count and not args.replace:
        print(f"`sequences` already holds {existing.count:,} rows. "
              f"Re-run with --replace to reload.")
        return 1
    if args.replace and existing.count:
        print(f"deleting {existing.count:,} existing rows")
        client.table("sequences").delete().neq("id", 0).execute()

    corpus = acquire(args.archive)

    n_train = args.limit or N_TRAIN
    n_val = args.limit or N_VAL
    n_test = args.limit or N_TEST

    # Validation comes from the official TRAIN split, never from test, so the
    # headline metric is measured on rows nothing has touched.
    train = take_balanced(corpus["train"], n_train, random.Random(SEED))
    val = take_balanced(corpus["train"], n_val, random.Random(SEED),
                        skip=n_train // 2)
    test = take_balanced(corpus["test"], n_test, random.Random(SEED + 1))

    payload = []
    for split, rows in (("train", train), ("val", val), ("test", test)):
        for text, label in rows:
            payload.append({"review_text": text, "label": label,
                            "split": split, "n_chars": len(text)})
    random.Random(SEED).shuffle(payload)

    print(f"inserting {len(payload):,} rows in batches of {BATCH}")
    for start in range(0, len(payload), BATCH):
        client.table("sequences").insert(payload[start:start + BATCH]).execute()
        print(f"\r  {min(start + BATCH, len(payload)):,} / {len(payload):,}",
              end="", flush=True)
    print()

    for split in ("train", "val", "test"):
        res = (client.table("sequences").select("id", count="exact")
               .eq("split", split).limit(1).execute())
        print(f"  {split:<6} {res.count:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())