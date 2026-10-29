# Seq-Sight — A Sequence-Modeling Service with Attention Explanations

**Name:** Lal Ruata
**Course / Section:** CST-435 
**GitHub:** https://github.com/SamRuata/seq-sight
**Streamlit Cloud (UI):** https://<fill-after-deploy>.streamlit.app
**Render.com (API):** https://<fill-after-deploy>.onrender.com — docs at [`/docs`](https://<fill-after-deploy>.onrender.com/docs), health at [`/healthz`](https://<fill-after-deploy>.onrender.com/healthz)
**Supabase project ref:** `ruyydsxkddxiwjttbtaj`
**Product presentation (video):** https://<fill-after-recording>
**Engineering report:** below, in this README
**Model card:** [`MODEL_CARD.md`](MODEL_CARD.md) · **Dataset card:** [`DATASET_CARD.md`](DATASET_CARD.md)

> The Render API runs on the free tier and sleeps when idle. The first request after a quiet period can take up to 60 seconds; every UI tab handles that timeout rather than erroring.

## Product pitch

**Variant B — sentiment classification on IMDB.**

A campus operations team collects free-text answers on student surveys and wants them scored. A number alone is not enough: before anyone acts on "63% negative," they need to see which words drove each call, and they need every scored response kept so a weekly review can ask whether the model is still behaving.

Seq-Sight is that tool. Paste a review, get a label with the full probability distribution and a per-token attention heatmap showing where the model put its weight. Two models are served side by side — a recurrent LSTM baseline and a hand-written scaled dot-product attention block — so a visitor can see not just which is better but *how they differ*. A Failure Gallery shows, up front, six held-out reviews the attention model got confidently wrong.

**Customer story.** *As a student-affairs analyst, I want each free-text survey answer scored with the words that drove the score, so that I can decide whether a flagged response deserves a human read before I report a trend to the dean.*

No review a visitor types is ever stored. Only its SHA-256.

---

## Architecture

```
┌───────────────────────┐   JSON / text    ┌───────────────────────────┐  service-role  ┌───────────────────┐
│  Streamlit Cloud      │ ───────────────▶ │  FastAPI on Render         │ ─────────────▶ │  Supabase         │
│  ui/app.py            │   POST /predict  │  api/main.py               │   full access  │  Postgres         │
│  thin client:         │                  │  LSTM + attention (torch)  │                │  sequences        │
│  no torch, no model   │ ◀─────────────── │  hashes the text,          │ ◀───────────── │  runs             │
│  plots w/ matplotlib  │  label + probs   │  discards the words        │                │  predictions      │
└──────────┬────────────┘  + attention     └───────────────────────────┘                └─────────▲─────────┘
           │                                                                                      │
           └─────────────── anon key, read-only: Recent Predictions tab ──────────────────────────┘
```

The Streamlit tier imports no model code. The **service-role** key exists only in Render's environment; the **anon** key, limited by row-level security to `SELECT`, is the only credential a browser ever sees.

**Training reads from Supabase, as the brief requires, but runs locally.** `api/train.py` pulls the corpus out of Postgres and writes the resulting `runs` row; it is a CLI rather than a `POST /train` endpoint because Render's free tier has neither the memory nor the request timeout for a 400-second job. Checkpoints are versioned in git and loaded at startup.

## Data persisted / not persisted

**Persisted**

- `sequences` — the IMDB corpus: 10,000 train / 2,000 validation / 5,000 test rows, each with its review text, label and split. This is published data; storing it is what makes training reproducible.
- `runs` — one row per training job: model type, every hyperparameter, test accuracy, macro-F1, per-epoch curves, confusion matrix, parameter count, wall-clock seconds.
- `predictions` — one row per served request: SHA-256 of the input, predicted class and probability, the full attention vector, token count, serving run, latency.

**Deliberately not persisted**

- **The text of anything a visitor submits, in any form.** Only its hash. A hash recognises a repeat submission, reproduces which model decided what, and detects a replay. It cannot reconstruct what anyone wrote.
- Trained weights — checkpoints live in `models/` in git, not in Postgres.

**The cost of that choice is visible in the product.** The Recent Predictions drilldown plots a past explanation against token *position*, with no words, because the words are gone. The live Try It tab shows words because they are still in the request. That asymmetry is the privacy decision made legible rather than argued for.

## Repository layout

```
seq-sight/
├── api/
│   ├── main.py            # 10 endpoints, OpenAPI-tagged, validation layer
│   ├── models.py          # LSTMBaseline, SelfAttention, AttentionPool
│   ├── train.py           # CLI: reads Supabase, trains, writes a runs row
│   ├── failures.py        # CLI: precomputes the Failure Gallery
│   ├── summary.py         # CLI: regenerates the tables in this README
│   └── configs/           # lstm, attention, attention_no_pos
├── db/
│   ├── migrations/        # 001 (template) + 002 (sequences, predictions, RLS)
│   ├── load.py            # IMDB -> Supabase
│   └── check.py           # setup verification
├── shared/
│   ├── text.py            # tokenizer, vocabulary, padding masks
│   └── schemas.py         # Pydantic contract
├── models/                # 3 checkpoints + vocab.json
├── ui/app.py              # 6-tab thin client
└── tests/                 # 36 tests
```

## API endpoints

`POST /predict` · `POST /predict_batch` · `GET /models` · `GET /runs` ·
`GET /predictions` · `GET /classes` · `GET /scaling_demo` · `GET /healthz` · `GET /version`

Grouped under four OpenAPI tags — prediction, models, meta, ops — so `/docs` reads as documentation.

## Configuration comparison

Generated from the committed checkpoints by `python -m api.summary`, not typed by hand:

| run | type | test acc | macro F1 | params | best epoch | train (s) | explainable |
|---|---|---|---|---|---|---|---|
| `lstm` | lstm | **0.7990** | 0.7989 | 2,692,354 | 5 | 398 | no |
| `attention_no_pos` | attention | 0.7890 | 0.7890 | 2,692,354 | 5 | 61 | yes |
| `attention` | attention | 0.7804 | 0.7804 | 2,692,354 | 4 | **53** | yes |

Every run: seed 42, the same 10,000 reviews in the same order, 5 epochs, batch 128, Adam at 1e-3, 128-token window, 20,000-word vocabulary.

| class | lstm P / R / F1 | attention P / R / F1 | support |
|---|---|---|---|
| negative | 0.789 / 0.817 / 0.803 | 0.782 / 0.777 / 0.780 | 2,500 |
| positive | 0.810 / 0.781 / 0.795 | 0.778 / 0.784 / 0.781 | 2,500 |

## Tests

`pytest -q` — 36 tests, all passing. Input-length and dtype validation (`test_validation.py`), the attention matrix's shape and row-stochasticity plus a frozen reference pinned to ±1e-3 (`test_predict.py`), a Supabase round trip whose `served_by_run_id` must resolve to a `runs` row (`test_supabase_roundtrip.py`), unit tests for the hand-written attention block including masking and permutation-equivariance (`test_attention.py`), and unit tests for the tokenizer and padding masks (`test_text.py`).

Two are worth singling out. One asserts that **no stored string column exceeds 500 characters**, which makes the no-raw-text invariant a build failure rather than a promise. Another breaks the Supabase client mid-request and asserts `/predict` still returns 200 — it caught a real bug, where client construction sat outside the logging try block and a database outage would have failed every prediction.

## Running locally

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt joblib pyarrow matplotlib
cp .env.example .env                      # SUPABASE_URL + SERVICE key
python -m db.check                        # verify setup
python -m db.load                         # IMDB -> Supabase
python -m shared.text                     # build models/vocab.json
python -m api.train --config api/configs/lstm.yaml
python -m api.train --config api/configs/attention.yaml
python -m api.failures                    # precompute the gallery
uvicorn api.main:app --reload --port 8000 # terminal 1
streamlit run ui/app.py                   # terminal 2
```

---

# Engineering Report

## Engineering analysis

Both models were trained under one budget: seed 42, the same 10,000 reviews read from Supabase in the same order, five epochs, batch 128, Adam at 1e-3, a 128-token window, and one 20,000-word vocabulary built from training rows only. Parameter counts are **identical to the digit — 2,692,354** — because the attention block carries a position-wise feed-forward sized to match the LSTM's gate matrices. Nothing below can be explained by capacity.

**The recurrent baseline is more accurate: 0.7990 against 0.7804, a 1.86-point gap.** This follows from how each model composes. The LSTM's cell state passes through positions in order, so "not" alters the state that the following token then updates; negation modifies what comes after it as a matter of mechanism. A single self-attention block computes a weighted average of value vectors, and an average is permutation-invariant — the block literally cannot tell a reordering from the original. A unit test in `test_attention.py` asserts exactly this: shuffle the inputs and the outputs shuffle with them. Sentiment turns on negation and contrast, so the cost lands where the architecture is weakest.

**Attention trains 7.6× faster: 53 seconds against 398.** Also structural, and in the opposite direction. The LSTM cannot begin step *t* until step *t−1* has finished, so its sequential depth is O(T) — 128 steps that no amount of hardware parallelises. Attention computes every position in two matrix multiplications with O(1) sequential depth. Because the parameter counts are equal, no one can attribute the gap to model size: it is the recurrence itself.

**Positional encoding cost 0.86 points.** This was not the expected result. `attention_no_pos` scored 0.7890 against the encoded model's 0.7804, and the per-epoch curves say why: at epoch 1 the encoded model is at 0.6535 validation accuracy while the order-blind one is already at 0.7310. Sinusoidal values lie in [−1, 1] — the same order of magnitude as a freshly initialised embedding — so early in training the encoding is a large perturbation of the word-identity signal. IMDB sentiment is close to a bag-of-words problem, and five epochs is not enough to recover from the handicap. The encoding is necessary in principle, as the permutation test shows; at this budget it did not pay for itself.

One caveat limits all three findings: both models were still improving when the budget ran out — the LSTM reached its best validation score at epoch 5 of 5. These are rankings at this budget, not ceilings.

## Risk and ethics

A model trained on IMDB carries that corpus's shape into whatever it is pointed at. The split is **polarised by construction**: reviews rated 5 and 6 out of 10 were excluded when the dataset was built, so the model has never seen ambivalence. Asked to score a genuinely mixed survey answer, it does not hedge — it picks a pole and reports high confidence, because every training example was at a pole. The Failure Gallery shows this directly: the model's most confident errors sit at 0.98 and above.

It also learns topic as sentiment. The tokens receiving the most attention in those failures are words like "alien," "season," "critters" — plot nouns that happened to correlate with a label in the training data, not words any reader would point to. A deployment scoring survey text would carry those correlations into a domain where they mean nothing.

Three disclosures a deployment owes its users. **What it was trained on** — movie reviews, not survey answers, so accuracy here says little about accuracy there. **That the confidence figure is a softmax output, not a frequency** — 0.98 does not mean 98 of 100 such calls are right. **That the heatmap is not an explanation** — the Failure Gallery makes this case inside the product rather than in a footnote, because a tool that ships an attention map beside a wrong answer gives its user false assurance, which is worse than giving them nothing.

The persistence rule follows from who is exposed. The corpus is stored in full: it is published data, and reproducibility requires it. A submitted review is hashed and the characters discarded, because the database is the component that accumulates a permanent record of what real people wrote. A hash supports every legitimate audit and nothing an illegitimate one wants. The cost is real and visible: a stored heatmap cannot be labelled with its words, so the Recent Predictions tab plots shape without content.

## Worldview reflection

"Do not boast about tomorrow, for you do not know what a day may bring" (Proverbs 27:1, ESV). A classifier's output is a claim, stated without hesitation, about something it cannot see — and the number beside it reads as a measure of how sure it is entitled to be.

**The uncertainty this product shows:** the full probability distribution rather than a bare label; the attention weights; a warning when a review was truncated; and a Failure Gallery that leads with six confident mistakes instead of six successes.

**The uncertainty it hides, and I would rather name than let a user discover:** the softmax probability is not calibrated, so 0.98 is a shape of the output layer and not a frequency. There is no abstain path — every input gets one of two labels, including input the model has no business judging. The attention weights look like reasons and are not. And a single accuracy figure describes IMDB-like text only.

Building this honestly meant reporting that my attention model lost, that its positional encoding made things worse, and that its explanations point at the wrong words. Each of those was a result I could have buried by not running the comparison. Stewardship of a model's uncertainty is mostly the discipline of running the test that could embarrass you, and then publishing what it said.