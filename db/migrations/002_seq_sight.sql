-- 002_seq_sight.sql
-- Reshapes the Attend-It template schema for Seq-Sight (variant B: IMDB sentiment).
--
-- The template stored synthetic sequences as generation PARAMETERS and
-- regenerated them from a seed. This product uses a real corpus, so the corpus
-- itself must live in Postgres and training must read it back out.

-- ---------------------------------------------------------------------------
-- sequences: the IMDB training corpus. This is public dataset text, not user
-- input -- the distinction that governs what may be persisted here.
-- ---------------------------------------------------------------------------
create table if not exists sequences (
    id          bigint generated always as identity primary key,
    review_text text    not null,
    label       integer not null check (label in (0, 1)),
    split       text    not null check (split in ('train', 'val', 'test')),
    n_chars     integer not null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_sequences_split on sequences (split);

-- ---------------------------------------------------------------------------
-- runs: extended so one row fully describes a model in the comparison.
-- model_type is the axis the whole report turns on: 'lstm' vs 'attention'.
-- ---------------------------------------------------------------------------
alter table runs add column if not exists model_type    text;
alter table runs add column if not exists embed_dim     integer;
alter table runs add column if not exists hidden_dim    integer;
alter table runs add column if not exists n_heads       integer;
alter table runs add column if not exists max_len       integer;
alter table runs add column if not exists vocab_size    integer;
alter table runs add column if not exists n_params      integer;
alter table runs add column if not exists seed          integer;
alter table runs add column if not exists best_epoch    integer;
alter table runs add column if not exists train_seconds double precision;
alter table runs add column if not exists history       jsonb;
alter table runs add column if not exists confusion     jsonb;

-- ---------------------------------------------------------------------------
-- predictions: one row per served request.
--
-- input_hash, never the text. A visitor's review is hashed and the characters
-- are discarded; only the hash, the model's output and the attention vector
-- survive. The corpus above is stored in full because it is published data the
-- model was trained on; a review someone types into the demo is not.
-- ---------------------------------------------------------------------------
create table if not exists predictions (
    id                    bigint generated always as identity primary key,
    served_by_run_id      bigint  not null references runs (id) on delete cascade,
    input_hash            text    not null,
    predicted_label       integer not null,
    predicted_class       text    not null,
    confidence            double precision not null,
    probs                 jsonb   not null,
    attention_vector_json jsonb   not null,
    n_tokens              integer not null,
    true_label            integer,
    correct               boolean,
    source                text    not null default 'user',
    latency_ms            integer,
    created_at            timestamptz not null default now()
);

create index if not exists idx_predictions_run     on predictions (served_by_run_id);
create index if not exists idx_predictions_created on predictions (created_at desc);

-- ---------------------------------------------------------------------------
-- Superseded template tables.
--   run_artifacts      -- this product's embedding matrix alone is ~8 MB per
--                         model; base64 in a text column is the wrong home for
--                         it. Checkpoints are versioned in git instead.
--   sequence_metadata  -- replaced by `predictions`, which additionally stores
--                         the attention vector the brief requires.
-- ---------------------------------------------------------------------------
drop table if exists run_artifacts;
drop table if exists sequence_metadata;

-- ---------------------------------------------------------------------------
-- RLS. The UI reads `runs`, `predictions` and `sequences` with the ANON key.
-- Corpus rows are public IMDB text, so anonymous SELECT on them discloses
-- nothing that is not already published. All writes stay server-side behind
-- the service-role key.
-- ---------------------------------------------------------------------------
alter table sequences   enable row level security;
alter table predictions enable row level security;

drop policy if exists "anon can read sequences" on sequences;
create policy "anon can read sequences"
    on sequences for select to anon using (true);

drop policy if exists "anon can read predictions" on predictions;
create policy "anon can read predictions"
    on predictions for select to anon using (true);