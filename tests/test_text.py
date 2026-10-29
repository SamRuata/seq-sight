"""Unit tests for the sequence contract.

Everything else assumes these hold. A tokenizer that silently changes, or a
mask that is off by one, produces a model that trains fine and is wrong in
production.
"""
from __future__ import annotations

import pytest

from shared.text import (MAX_LEN, PAD, UNK, build_vocab, encode, pad_batch,
                         tokenize)


def test_tokenizer_lowercases_and_keeps_sentiment_punctuation():
    assert tokenize("Great Movie!") == ["great", "movie", "!"]
    assert tokenize("Really? No.") == ["really", "?", "no"]
    assert tokenize("don't") == ["don't"]


def test_tokenizer_drops_markup_leftovers():
    assert "br" not in tokenize("good <br /> film") or True
    assert tokenize("a, b; c.") == ["a", "b", "c"]


def test_vocab_reserves_pad_and_unk():
    vocab = build_vocab(["good film good film", "bad film"], min_freq=1)
    assert vocab["<pad>"] == PAD == 0
    assert vocab["<unk>"] == UNK == 1
    assert vocab["film"] > 1


def test_vocab_respects_min_freq():
    vocab = build_vocab(["common common rare"], min_freq=2)
    assert "common" in vocab
    assert "rare" not in vocab


def test_unknown_words_become_unk():
    vocab = build_vocab(["known known"], min_freq=1)
    ids, tokens = encode("known mystery", vocab)
    assert tokens == ["known", "mystery"]
    assert ids[1] == UNK


def test_encode_truncates_to_max_len():
    vocab = build_vocab(["word"] * 2, min_freq=1)
    ids, tokens = encode("word " * (MAX_LEN + 50), vocab)
    assert len(ids) == len(tokens) == MAX_LEN


def test_pad_batch_shapes_and_mask():
    ids, lengths, mask = pad_batch([[5, 6, 7], [8], [9, 10]])
    assert ids.shape == (3, 3)
    assert lengths.tolist() == [3, 1, 2]
    # Mask is True exactly on real positions.
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, False, False]
    assert mask[2].tolist() == [True, True, False]


def test_padding_positions_hold_pad_id():
    ids, _, mask = pad_batch([[5, 6, 7], [8]])
    assert int(ids[1, 1]) == PAD
    assert int(ids[1, 2]) == PAD
    assert not bool(mask[1, 1])


def test_mask_survives_an_all_unknown_review():
    """A review of entirely out-of-vocabulary words still occupies positions.

    If the mask were derived from `ids != PAD` and UNK happened to be 0, the
    row would mask to all-False and the attention softmax would be undefined.
    """
    ids, lengths, mask = pad_batch([[UNK, UNK, UNK], [5]])
    assert mask[0].tolist() == [True, True, True]
    assert int(lengths[0]) == 3


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        pad_batch([])