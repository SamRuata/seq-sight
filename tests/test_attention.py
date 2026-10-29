"""Unit tests for the scaled dot-product attention block.

The block is hand-written, so these assert the properties the derivation
claims rather than trusting a library to have them.
"""
from __future__ import annotations

import math

import torch

from api.models import (AttentionPool, SelfAttention, masked_softmax,
                        saturation_demo, sinusoidal_encoding)


def test_attention_rows_are_distributions():
    torch.manual_seed(0)
    block = SelfAttention(16)
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out, attn = block(x, mask)

    assert out.shape == (2, 5, 16)
    assert attn.shape == (2, 5, 5)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert (attn >= 0).all()


def test_padded_keys_receive_no_weight():
    """The property the whole masking design exists to guarantee."""
    torch.manual_seed(0)
    block = SelfAttention(16)
    x = torch.randn(1, 6, 16)
    mask = torch.tensor([[True, True, True, False, False, False]])
    _, attn = block(x, mask)

    assert float(attn[0, :, 3:].sum()) < 1e-6
    assert torch.allclose(attn[0, :, :3].sum(dim=-1), torch.ones(6), atol=1e-5)


def test_masked_softmax_does_not_rescale_survivors():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, False]])
    w = masked_softmax(scores, mask)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert float(w[0, 2]) == 0.0


def test_scaling_divides_by_sqrt_d_k():
    block = SelfAttention(64)
    assert abs(block.scale - math.sqrt(64)) < 1e-9


def test_scaling_keeps_softmax_out_of_saturation():
    """The numerical claim the Concepts tab makes, asserted."""
    demo = saturation_demo(d_k=128, seed=0)
    assert demo["unscaled_std"] > 8.0          # ~sqrt(128) = 11.3
    assert demo["scaled_std"] < 1.5            # ~1
    # Unscaled, one position takes nearly everything; scaled, it does not.
    assert demo["unscaled_max_softmax"] > demo["scaled_max_softmax"]
    assert demo["scaled_entropy"] > demo["unscaled_entropy"]


def test_attention_pool_weights_sum_to_one_and_ignore_padding():
    torch.manual_seed(0)
    pool = AttentionPool(16)
    x = torch.randn(2, 5, 16)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
    context, weights = pool(x, mask)

    assert context.shape == (2, 16)
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert float(weights[1, 2:].sum()) < 1e-6


def test_self_attention_is_permutation_equivariant_without_positions():
    """Why positional encoding is not optional.

    Shuffle the inputs and the outputs shuffle with them -- the block has no
    idea the order changed. This is the structural reason the report gives for
    attention losing to the recurrent baseline on negation.
    """
    torch.manual_seed(0)
    block = SelfAttention(8)
    x = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    perm = torch.tensor([2, 0, 3, 1])

    out_a, _ = block(x, mask)
    out_b, _ = block(x[:, perm], mask)
    assert torch.allclose(out_a[:, perm], out_b, atol=1e-5)


def test_positional_encoding_shape_and_range():
    pe = sinusoidal_encoding(32, 16)
    assert pe.shape == (32, 16)
    assert float(pe.abs().max()) <= 1.0
    assert not torch.allclose(pe[0], pe[1])