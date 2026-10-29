# Dataset Card — IMDB Large Movie Review Dataset (Seq-Sight)

**Dataset:** IMDB Large Movie Review Dataset (aclImdb v1)
**Used by:** Seq-Sight, CST-435 Topic 4, variant B
**Maintainer of this card:** Lal Ruata

---

## Why this dataset

The brief offered Jena Climate hourly forecasting or IMDB sentiment. IMDB was chosen for two reasons.

**The explanation is legible.** An attention weight over a word is something a reader can judge — you can look at which words the model weighted and decide for yourself whether they are the ones that carry the sentiment. An attention weight over "hour −17" of a temperature series is not checkable in the same way. Since the product's whole claim is that it shows *where the model looked*, the task had to be one where a visitor can tell whether the model looked somewhere sensible. That turned out to matter more than expected: the finding that the weights land on the wrong words is only available because the right words are obvious.

**The comparison has somewhere to go.** Sentiment turns on negation, contrast and scope — "not good", "good, until". Those are exactly the compositions a recurrent cell handles by construction and a single self-attention block does not, so the architectural difference has a place to show up. A 24-hour temperature window is short enough that the recurrence advantage and the parallelism advantage both nearly vanish.

## Source and provenance

- **Origin:** Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011. Stanford AI Lab.
- **Composition:** 50,000 labelled reviews, balanced, split 25,000 train / 25,000 test by the publishers. A further 50,000 unlabelled reviews ship in the archive and are **not** used here.
- **Labels:** binary. Negative is a rating of 4 or lower out of 10; positive is 7 or higher. **Reviews rated 5 and 6 were excluded when the dataset was built.** See "Known biases" — this is the single most consequential fact about this corpus.
- **Ceiling on reviews per film:** at most 30, so no single popular title dominates.
- **Licence:** the dataset is distributed for research use; see the Stanford page for terms.
- **Access path:** `python -m db.load`. The canonical Stanford host (`ai.stanford.edu`) was unreachable from the development network, so the loader tries the HuggingFace parquet mirror of the same corpus first (~40 MB across two files) and falls back to the 84 MB Stanford tarball, which it streams rather than extracting 50,000 files to disk. Either download is cached under `data/`, which is git-ignored, and `--archive` accepts a manually downloaded copy.

## What this project actually used

The full 25,000-row training split was not used. Every run trains on the same deterministic subsample so that differences between runs are attributable to the models rather than to the data.

| Split | Rows | Source |
|---|---|---|
| Train | 10,000 | official train split, balanced, shuffled with seed 42 |
| Validation | 2,000 | also from the official **train** split, disjoint from the above |
| Test | 5,000 | official **test** split, untouched by training or model selection |

Validation is carved from training rows, not test rows, so no test review influences which epoch's weights are kept. Each split is balanced 50/50 by construction rather than by chance.

The corpus lives in Supabase in a `sequences` table (`review_text`, `label`, `split`, `n_chars`), and `api/train.py` reads it back out over PostgREST — paginated in 1,000-row pages, because the API caps a response there and a trainer that silently used a tenth of its corpus would never announce it.

## Preprocessing

**Tokenization** is deliberately simple and dependency-free, in `shared/text.py`:

    _TOKEN = re.compile(r"[a-z0-9']+|[!?]")

Lowercase, then split into words, contractions, and standalone `!` and `?`. Exclamation and question marks are kept as tokens of their own because they carry sentiment in review prose; all other punctuation is dropped. A documented tokenizer anyone can reimplement was preferred to an opaque one — and the attention overlay has to be labelled with tokens a reader recognises.

**The one edit to the text itself:** the corpus was scraped from HTML and still carries `<br />` tags, which are replaced with a space at load time. Everything else is stored verbatim.

**Vocabulary:** 20,000 types, built from **training rows only**, minimum frequency 2. Ids 0 and 1 are reserved for `<pad>` and `<unk>`. Building the vocabulary from the whole corpus would leak test vocabulary into the model's input space, which is a quiet and common way to inflate a score.

**Window:** the first 128 tokens. The mean review is 236 tokens, so most are truncated. This is a compute budget, not a modelling claim — the recurrent baseline costs one sequential step per token, and at 200 tokens a single training run took long enough to make the comparison impractical. The consequence is stated rather than hidden: reviews often put their verdict in the last sentence, and that sentence is frequently not seen. The API returns `truncated: true` so a caller knows, and the Failure Gallery marks which failures were truncated.

**Padding and masking:** batches are padded to the longest member, with a boolean mask built from true lengths rather than from `ids != PAD`. A review consisting entirely of out-of-vocabulary words still occupies real positions; a mask derived from ids could mark such a row empty and leave the attention softmax undefined. Batches are formed by sorting on length and shuffling the batch *order*, so each batch pads to its own longest review instead of to the global cap.

## Known biases and limitations

- **No middle.** Ratings of 5 and 6 were excluded at construction, so the corpus contains no ambivalence at all. A model trained here has never seen a mixed opinion and has no representation for one. Asked to score genuinely mixed text, it picks a pole and reports high confidence — not because it is sure, but because every example it ever saw was at a pole. This is the most important limitation in this card.
- **Topic correlates with label, and the model learns it.** The tokens receiving the most attention in the model's confident errors are words like "alien", "season", "critters" — genre and plot nouns, not sentiment words. Horror and sci-fi B-movies are disproportionately reviewed negatively in this corpus, so their vocabulary becomes a negative signal. Pointed at text from another domain, those correlations are noise that still votes.
- **One domain, one register, one era.** English-language movie reviews written on a single website, largely before 2011. Nothing about accuracy here transfers to survey answers, support tickets or clinical notes — the deployment this project imagines scores student survey text, and this card exists partly to make that gap explicit.
- **Self-selected reviewers.** People who post film reviews online are not a random sample of filmgoers, and the dataset carries whatever that population's language, slang and reference points are.
- **Binary labels on a 10-point source.** A 4 and a 1 are both "negative" and the model cannot distinguish strength of feeling from direction of feeling.

## Privacy and what is stored

This corpus contains published reviews under screen names, not personal data, and the project's privacy design is not primarily about protecting it — it is about what a **visitor** submits to the live demo.

**Stored in Supabase:** the corpus in full, in `sequences`. It is published data and reproducibility requires it.

**Never stored:** the text of anything a visitor types into Try It. Only a SHA-256 of it, alongside the prediction, the attention vector and the token count. The database is the component that accumulates a permanent record of what real people wrote; a hash supports every legitimate audit — recognising a repeat, reproducing which model decided what, detecting a replay — and nothing an illegitimate one wants.

**Committed to git:** `data/failures.json`, which carries the text of ten held-out reviews so the Failure Gallery can display them. These are public corpus rows in a public repository, not user input in a growing database.

**Not committed:** the downloaded archive, which is fetched on demand and git-ignored.

## Reproducing

```bash
python -m db.check                        # verify credentials and schema
python -m db.load                         # corpus -> Supabase
python -m shared.text                     # build models/vocab.json
python -m api.train --config api/configs/lstm.yaml
python -m api.train --config api/configs/attention.yaml
python -m api.train --config api/configs/attention_no_pos.yaml
python -m api.failures                    # precompute the Failure Gallery
```

Seed 42 throughout. The same seed produces the same subsample, the same vocabulary, the same batch order and the same initialisation on any machine.