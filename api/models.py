"""The two architectures Seq-Sight compares, and the attention block itself.

The comparison is the product, so the two models are built to differ in ONE
thing: how a sequence of token embeddings becomes a single vector.

    LSTMBaseline         embedding -> packed LSTM -> masked pool -> linear
    AttentionClassifier  embedding -> self-attention -> masked pool -> linear

Everything else is held constant: the same embedding table, the same hidden
width, the same pooling, the same classifier head. Parameter counts are matched
to within a few percent so that any difference in accuracy is attributable to
the mechanism rather than to capacity -- the control that was missing from
Topic 3's first comparison and had to be added as an extra run.

`torch.nn.MultiheadAttention` is deliberately not used; the scaled dot-product
block below is written out so the scaling factor is visible in the code that
produces the numbers.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from shared.text import MAX_LEN, N_CLASSES, PAD, VOCAB_SIZE

EMBED_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor,
                   dim: int = -1) -> torch.Tensor:
    """Softmax that cannot put weight on padding.

    Padding is filled with a large negative number BEFORE the softmax, not
    zeroed after it. Zeroing afterwards leaves the surviving weights summing to
    less than one, which quietly rescales the context vector by how much
    padding a review happened to have.
    """
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=dim)


# ---------------------------------------------------------------------------
# Scaled dot-product self-attention -- written out, not imported
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    r"""One single-head scaled dot-product self-attention block.

        Q = X W_q,  K = X W_k,  V = X W_v
        A = softmax( Q Kᵀ / sqrt(d_k) )
        Z = A V

    **Why divide by sqrt(d_k).** If the components of q and k are independent
    with zero mean and unit variance, then q·k is a sum of d_k such products,
    so it has variance d_k and standard deviation sqrt(d_k). At d_k = 128 the
    raw scores therefore spread over roughly +/- 11 instead of +/- 1. Softmax
    over scores that large is effectively an argmax: one weight goes to ~1, the
    rest underflow, and their gradients vanish. Dividing by sqrt(d_k) returns
    the scores to unit variance and keeps the softmax in the region where it
    still has a usable gradient. `saturation_demo` below measures this.

    Single-head on purpose: the returned [T, T] matrix is shown to a user, and
    eight heads would mean eight competing stories about the same sentence.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = math.sqrt(dim)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x`` [B,T,D], ``mask`` [B,T] -> ``(output [B,T,D], attn [B,T,T])``."""
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # [B,T,T]

        # A padded KEY must never receive weight from any query, so the mask is
        # broadcast across the query axis: [B,1,T].
        attn = masked_softmax(scores, mask.unsqueeze(1), dim=-1)

        out = self.w_o(torch.matmul(attn, v))                        # [B,T,D]
        return out, attn


class AttentionPool(nn.Module):
    """Collapse [B,T,D] to [B,D] with a learned query, returning the weights.

    This is the layer whose weights are shown to the user. The [T,T]
    self-attention matrix says which words informed which other words; THIS
    vector says how much each position contributed to the decision, which is
    the question a reader of an explanation is actually asking.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(x, self.query) / math.sqrt(x.size(-1))  # [B,T]
        weights = masked_softmax(scores, mask, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)       # [B,D]
        return context, weights


def sinusoidal_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal positional encodings, [max_len, dim].

    Self-attention is permutation-invariant: without this, "not good at all"
    and "good, not at all" are the same bag of vectors. The LSTM gets order for
    free from its recurrence, so adding this is what makes the comparison fair
    rather than a handicap match.
    """
    pos = torch.arange(max_len).unsqueeze(1).float()
    i = torch.arange(0, dim, 2).float()
    div = torch.exp(-math.log(10_000.0) * i / dim)
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# The two models
# ---------------------------------------------------------------------------
class LSTMBaseline(nn.Module):
    """Recurrent baseline. Order comes from the recurrence, not from a mask.

    Uses a packed sequence so the LSTM stops at each review's true length
    instead of running over padding. Feeding padded zeros into an LSTM lets the
    final hidden state drift after the text ends, which is the classic silent
    bug in variable-length batches.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, n_classes: int = N_CLASSES,
                 embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, bidirectional: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                           bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                             Optional[torch.Tensor]]:
        emb = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=ids.size(1))

        # Mean over real positions only.
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        pooled = summed / lengths.unsqueeze(-1).clamp(min=1)

        logits = self.classifier(self.dropout(pooled))
        return logits, None, None          # no attention: that is the point


class AttentionClassifier(nn.Module):
    """The recurrent cell replaced by one self-attention block.

    Returns the pooling weights (one per token, the explanation) and the full
    [T,T] self-attention matrix (what the block itself computed).
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, n_classes: int = N_CLASSES,
                 embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, positional: bool = True,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.project = (nn.Identity() if embed_dim == hidden_dim
                        else nn.Linear(embed_dim, hidden_dim))
        self.positional = positional
        if positional:
            self.register_buffer("pe", sinusoidal_encoding(max_len, hidden_dim))

        self.attention = SelfAttention(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # Position-wise feed-forward. Present so the parameter count lands
        # within a few percent of the LSTM's; a thinner block would win or lose
        # on capacity rather than on mechanism.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPool(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
        x = self.project(self.embedding(ids))
        if self.positional:
            x = x + self.pe[:x.size(1)].unsqueeze(0)

        attended, attn_matrix = self.attention(x, mask)
        x = self.norm(x + attended)                 # residual
        x = self.norm2(x + self.ffn(x))

        context, weights = self.pool(x, mask)
        logits = self.classifier(self.dropout(context))
        return logits, weights, attn_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(config: Dict) -> nn.Module:
    kind = config["model_type"]
    common = dict(
        vocab_size=config.get("vocab_size", VOCAB_SIZE),
        n_classes=N_CLASSES,
        embed_dim=config.get("embed_dim", EMBED_DIM),
        hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
        dropout=config.get("dropout", DROPOUT),
    )
    if kind == "lstm":
        return LSTMBaseline(bidirectional=config.get("bidirectional", False),
                            **common)
    if kind == "attention":
        return AttentionClassifier(positional=config.get("positional", True),
                                   max_len=config.get("max_len", MAX_LEN),
                                   **common)
    raise ValueError(f"unknown model_type {kind!r}")


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def saturation_demo(d_k: int = HIDDEN_DIM, n: int = 4096,
                    seed: int = 0) -> Dict[str, float]:
    """Measure what the sqrt(d_k) divisor actually does. Feeds the Concepts tab.

    Draws random unit-variance query/key pairs and reports the spread of the
    dot products and the peakiness of the resulting softmax, with and without
    the scaling. These are measured numbers, not quoted ones.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, d_k, generator=g)
    k = torch.randn(n, d_k, generator=g)
    raw = (q * k).sum(dim=-1)

    def peak(scores: torch.Tensor) -> Tuple[float, float]:
        row = scores[: min(64, n)].unsqueeze(0)
        p = torch.softmax(row, dim=-1)
        entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
        return float(p.max()), entropy

    unscaled_max, unscaled_entropy = peak(raw)
    scaled_max, scaled_entropy = peak(raw / math.sqrt(d_k))

    return {
        "d_k": d_k,
        "sqrt_d_k": math.sqrt(d_k),
        "unscaled_std": float(raw.std()),
        "scaled_std": float((raw / math.sqrt(d_k)).std()),
        "unscaled_max_softmax": unscaled_max,
        "scaled_max_softmax": scaled_max,
        "unscaled_entropy": unscaled_entropy,
        "scaled_entropy": scaled_entropy,
        "uniform_entropy": math.log(min(64, n)),
    }


if __name__ == "__main__":
    from shared.text import pad_batch

    ids, lengths, mask = pad_batch([[5, 9, 12, 7], [3, 8], [11] * 10])

    for kind in ("lstm", "attention"):
        model = build_model({"model_type": kind})
        model.eval()
        with torch.no_grad():
            logits, weights, matrix = model(ids, lengths, mask)
        print(f"{kind:<10} params {count_params(model):>9,}  "
              f"logits {tuple(logits.shape)}  "
              f"weights {tuple(weights.shape) if weights is not None else '--'}  "
              f"matrix {tuple(matrix.shape) if matrix is not None else '--'}")
        if weights is not None:
            # Padding must receive exactly zero weight, and the rest must sum to 1.
            print(f"           pad weight {float(weights[1, 2:].sum()):.2e}  "
                  f"row sum {float(weights[1].sum()):.6f}")

    print()
    demo = saturation_demo()
    print(f"d_k={demo['d_k']}  sqrt(d_k)={demo['sqrt_d_k']:.2f}")
    print(f"  dot-product std   {demo['unscaled_std']:.2f} -> "
          f"{demo['scaled_std']:.2f}")
    print(f"  max softmax       {demo['unscaled_max_softmax']:.3f} -> "
          f"{demo['scaled_max_softmax']:.3f}")
    print(f"  entropy (nats)    {demo['unscaled_entropy']:.2f} -> "
          f"{demo['scaled_entropy']:.2f}   uniform = "
          f"{demo['uniform_entropy']:.2f}")