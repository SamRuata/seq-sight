# Model Card — Seq-Sight

**Models:** `attention` (served by default) and `lstm` (the baseline), plus the `attention_no_pos` ablation
**Owner:** Lal Ruata, CST-435
**Served at:** https://seq-sight-api-usf4.onrender.com
**Dataset:** IMDB — see [`DATASET_CARD.md`](DATASET_CARD.md)
**Last updated:** October 2026

---

## Model details

Two architectures that differ in exactly one thing: how a sequence of token embeddings becomes a single vector. Everything else — the embedding table, the hidden width, the pooling, the classifier head, the optimizer, the budget — is held constant.

| | `lstm` (baseline) | `attention` (deployed) |
|---|---|---|
| Sequence mechanism | single-layer LSTM over packed sequences | one hand-written scaled dot-product self-attention block |
| Pooling | mean over unpadded positions | learned-query attention pooling |
| Order information | intrinsic to the recurrence | sinusoidal positional encoding, added |
| Parameters | **2,692,354** | **2,692,354** |
| Embedding | 20,000 × 128, `padding_idx=0` | identical |
| Hidden width | 128 | 128 |
| Dropout | 0.3 | 0.3 |
| Loss / optimizer | cross-entropy; Adam, lr 1e-3, grad-norm clip 5.0 | identical |
| Budget | 10,000 reviews, 5 epochs, batch 128, seed 42, 128-token window | identical |
| Checkpoint | `models/lstm.joblib` | `models/attention.joblib` |

The parameter counts are equal to the digit, not approximately. The attention block carries a position-wise feed-forward layer sized so that its total matches the LSTM's gate matrices. This is the control that makes the comparison mean something: no result below can be attributed to one model being larger.

### The attention block

`torch.nn.MultiheadAttention` is deliberately not used. The block is written out in `api/models.py` so the scaling factor is visible in the code that produces the numbers:

    Q = XW_Q,  K = XW_K,  V = XW_V
    A = softmax( Q Kᵀ / sqrt(d_k) )
    Z = A V

Single-head on purpose. The `[T, T]` matrix is shown to a user, and eight heads would mean eight competing accounts of the same sentence.

**Why the divisor.** With independent, zero-mean, unit-variance components, `q·k` is a sum of `d_k` such products, so it has standard deviation `sqrt(d_k)`. Measured on this deployment at `d_k = 128`: the raw dot products have σ = 11.39 and the scaled ones σ = 1.01; the largest softmax weight drops from 0.611 to 0.108; entropy rises from 0.67 to 3.69 nats against a uniform ceiling of 4.16. Unscaled, the softmax is effectively an argmax and the gradient through the losing positions vanishes. `GET /scaling_demo` recomputes this live.

**Masking.** Padded positions are filled with the dtype minimum *before* the softmax, never zeroed after it. Zeroing afterwards leaves the surviving weights summing to less than one, which silently rescales the context vector by how much padding a review happened to have. `tests/test_attention.py` asserts that padded keys receive less than 1e-6 of the total weight.

## Performance

Held-out test split: 5,000 reviews never used in training or model selection.

| run | type | test accuracy | macro F1 | parameters | best epoch | train (s) |
|---|---|---|---|---|---|---|
| `lstm` | recurrent | **0.7990** | 0.7989 | 2,692,354 | 5 | 398 |
| `attention_no_pos` | attention | 0.7890 | 0.7890 | 2,692,354 | 5 | 61 |
| `attention` | attention | 0.7804 | 0.7804 | 2,692,354 | 4 | **53** |

### Per class

| class | `lstm` P / R / F1 | `attention` P / R / F1 | support |
|---|---|---|---|
| negative | 0.789 / 0.817 / 0.803 | 0.782 / 0.777 / 0.780 | 2,500 |
| positive | 0.810 / 0.781 / 0.795 | 0.778 / 0.784 / 0.781 | 2,500 |

Neither model is meaningfully better at one class than the other — the corpus is balanced and so are the errors. The LSTM's advantage is spread evenly rather than concentrated in a class.

### Confusion, 5,000 held-out reviews

| | predicted negative | predicted positive |
|---|---|---|
| **`lstm`** — true negative | 2,042 | 458 |
| **`lstm`** — true positive | 547 | 1,953 |
| **`attention`** — true negative | 1,942 | 558 |
| **`attention`** — true positive | 540 | 1,960 |

## Which model is deployed, and why it is not the better one

`GET /predict` serves `attention` by default, although `lstm` is 1.86 points more accurate.

That is a deliberate trade and worth stating plainly rather than burying. The product's purpose is to show *where a model looked*, and the recurrent baseline has nothing to show — no per-position weights exist to return. A visitor arriving at Try It should see an explanation beside the first prediction they make, not after discovering a dropdown.

The dropdown is still there, the baseline is one click away, and the Baseline vs. Attention tab leads with the accuracy gap rather than hiding it. **A deployment optimising for accuracy rather than for teaching should serve `lstm` instead**, and the only change required is one constant in `api/main.py`.

## Explainability, and its limits

Two things are returned, and they answer different questions:

- **`attention`** — one weight per token from the pooling layer, summing to 1. How much each position contributed to the decision.
- **`attention_matrix`** — the `[T, T]` self-attention matrix, optional because at T = 128 it is 16,384 numbers. Row *i* is what position *i* attended to.

**The weights are not explanations, and the product says so.** In the attention model's most confident errors, the highest-weighted tokens are words like "alien", "season", "critters", "when" — plot nouns and function words, not anything a reader would point at. If attention were an explanation, it would concentrate on sentiment-bearing words. It does not, and the model is confidently wrong anyway.

The honest reading is that a weight distribution shows where the pooling layer put mass, not why the classifier decided. The Failure Gallery makes this argument inside the product rather than in a footnote, because a tool that ships a heatmap beside a wrong answer gives its user false assurance — worse than giving them nothing.

## Intended use

**Primary:** a teaching demo. A visitor pastes a review, sees a label with its full probability distribution and a per-token heatmap, and can switch architectures to see the comparison directly.

**Out of scope.** Anything consequential. This model classifies English-language movie reviews into two fixed classes. It has never seen a survey answer, a support ticket, a performance review, or any ambivalent text at all.

## Limitations

- **No abstain path.** Every input gets one of two labels. A weather report returns a sentiment with a confidence attached.
- **Trained on a corpus with no middle.** The IMDB split excludes reviews rated 5 and 6 of 10, so the model has never seen ambivalence and does not hedge — it picks a pole at high confidence. Its most confident errors sit above 0.98.
- **Probabilities are softmax outputs, not calibrated frequencies.** 0.98 does not mean 98 of 100 such predictions are right.
- **A 128-token window.** The mean review is 236 tokens, so most are truncated, and a review's verdict is often its last sentence. The API reports `truncated: true` when this happens; the model has no way to know what it missed.
- **The ranking is budget-specific.** Both models were still improving at epoch 5 of 5. A longer budget could reorder them, and could plausibly rescue the positional encoding.
- **Positional encoding currently costs accuracy.** `attention_no_pos` beats `attention` by 0.86 points, because sinusoidal values are the same magnitude as a freshly initialised embedding and perturb word identity early in training. The encoding is necessary in principle — self-attention is provably permutation-equivariant without it — but at this budget it did not pay for itself.

## Privacy: what is and is not stored

**Never written to the database:** the text of anything a visitor submits, in any encoding.

**Written to the database:** a SHA-256 of the input, the predicted class and probability, the full probability vector, the attention weight vector, the token count, latency, and the id of the training run that served it.

This is enforced, not merely documented: a test inspects every stored string column across the most recent predictions and fails if any exceeds 500 characters, on the grounds that nothing legitimate in a metadata row is that long. A review is.

The cost is visible in the product. The Recent Predictions drilldown plots a past explanation against token *position* with no words attached, because the words are gone. The live Try It tab shows words because they are still in the request.

## Ethical considerations

A two-class sentiment model is low-stakes, but the family it belongs to is not — the same architecture, retrained, is what ranks support tickets, screens job applications and moderates content. Three things in this design transfer:

The explanation is surfaced beside every prediction rather than available on request. The system stores the minimum it needs to be audited rather than the maximum it could. And the failures are shown first, in their own tab, rather than being discoverable only by someone who already suspected a problem.

The fuller argument is in the Engineering Report section of [`README.md`](README.md).