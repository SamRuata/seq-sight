"""Tokenization, vocabulary, and padded-batch construction for Seq-Sight.

This is the sequence contract. Both models and both tiers go through it, so a
review is turned into token ids exactly the same way at training time and at
serve time -- the single most common source of a model that scores well
offline and behaves strangely in production.

Heavy imports (torch, supabase) live INSIDE functions on purpose: the Streamlit
tier imports this module for `tokenize` and the special-token ids, and must not
drag torch onto a free-tier container.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD, UNK = 0, 1
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

VOCAB_SIZE = 20_000
MAX_LEN = 128
MIN_FREQ = 2

CLASS_NAMES = ["negative", "positive"]
N_CLASSES = 2

VOCAB_PATH = Path("models/vocab.json")

# Words, contractions and standalone punctuation that carries sentiment.
# Deliberately simple and dependency-free: a documented tokenizer that anyone
# can reimplement beats an opaque one, and the attention overlay has to be
# labelled with tokens a reader recognises.
_TOKEN = re.compile(r"[a-z0-9']+|[!?]")


def tokenize(text: str) -> List[str]:
    """Lowercase, then split into word / contraction / `!` / `?` tokens.

    `!` and `?` are kept as tokens of their own because exclamation is a real
    sentiment signal in review text; all other punctuation is dropped.
    """
    return _TOKEN.findall(text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def build_vocab(texts: Iterable[str], vocab_size: int = VOCAB_SIZE,
                min_freq: int = MIN_FREQ) -> Dict[str, int]:
    """Build a word -> id map from TRAINING text only.

    Building this from the full corpus would leak test vocabulary into the
    model's input space, so the caller must pass training rows alone.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD, UNK_TOKEN: UNK}
    for word, freq in counts.most_common():
        if freq < min_freq or len(vocab) >= vocab_size:
            break
        vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: Path = VOCAB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab), encoding="utf-8")
    return path


def load_vocab(path: Path = VOCAB_PATH) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"no vocabulary at {path} -- run `python -m shared.text` first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode(text: str, vocab: Dict[str, int],
           max_len: int = MAX_LEN) -> Tuple[List[int], List[str]]:
    """Return ``(token_ids, tokens)``, truncated to ``max_len``.

    Tokens are returned alongside the ids because the attention explanation is
    only meaningful if each weight can be labelled with the word it belongs to.

    Truncation keeps the FIRST ``max_len`` tokens. A large share of IMDB
    reviews are longer than that, and a review's verdict often sits in its last
    sentence, so this discards signal -- a limitation the model card states and
    the failure analysis returns to, rather than one hidden in a preprocessing
    step. The window is a compute budget: the recurrent baseline costs one
    sequential step per token, so doubling it doubles training time.
    """
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, UNK) for t in tokens]
    return ids, tokens


def pad_batch(sequences: Sequence[Sequence[int]], max_len: int = MAX_LEN):
    """Pad a ragged batch. Returns ``(ids [B,T], lengths [B], mask [B,T])``.

    ``mask`` is True at real tokens and False at padding. Every consumer must
    honour it: the attention block adds -inf at masked positions before its
    softmax, and mean-pooling divides by the true length. Without that, padding
    silently becomes evidence -- the model learns that long reviews have more
    zeros and the attention weights spread onto nothing.
    """
    import torch

    if not sequences:
        raise ValueError("empty batch")

    lengths = [max(1, min(len(s), max_len)) for s in sequences]
    width = max(lengths)
    ids = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = lengths[i]
        if seq:
            ids[i, :n] = torch.tensor(seq[:n], dtype=torch.long)

    # Build the mask from lengths rather than from `ids != PAD`: a review made
    # entirely of out-of-vocabulary words still occupies real positions, and a
    # row whose mask is all False would make the attention softmax undefined.
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True

    return ids, torch.tensor(lengths, dtype=torch.long), mask


# ---------------------------------------------------------------------------
# Corpus access
# ---------------------------------------------------------------------------
def fetch_split(client, split: str, page: int = 1000) -> Tuple[List[str], List[int]]:
    """Read one split out of Supabase, paginated.

    PostgREST caps a response at 1,000 rows, so a 10,000-row training split
    needs explicit ranges -- without them you silently train on a tenth of the
    corpus and never find out.
    """
    texts: List[str] = []
    labels: List[int] = []
    start = 0
    while True:
        res = (client.table("sequences")
               .select("review_text,label")
               .eq("split", split)
               .order("id")
               .range(start, start + page - 1)
               .execute())
        rows = res.data or []
        texts.extend(r["review_text"] for r in rows)
        labels.extend(int(r["label"]) for r in rows)
        if len(rows) < page:
            break
        start += page
    return texts, labels


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv()
    client = create_client(os.environ["SUPABASE_URL"],
                           os.environ["SUPABASE_SERVICE_KEY"])

    texts, labels = fetch_split(client, "train")
    print(f"train rows      {len(texts):,}  (positive {sum(labels):,})")

    lengths = [len(tokenize(t)) for t in texts]
    over = sum(1 for n in lengths if n > MAX_LEN)
    print(f"tokens/review   mean {sum(lengths) / len(lengths):.0f}  "
          f"max {max(lengths):,}")
    print(f"MAX_LEN         {MAX_LEN}")
    print(f"over MAX_LEN    {over:,} ({100 * over / len(lengths):.0f}%)")

    vocab = build_vocab(texts)
    print(f"vocabulary      {len(vocab):,} types")
    print(f"saved to        {save_vocab(vocab)}")

    ids, tokens = encode(texts[0], vocab)
    unk = sum(1 for i in ids if i == UNK)
    print(f"\nsample: {len(tokens)} tokens, {unk} unknown")
    print("  " + " ".join(tokens[:25]) + " ...")