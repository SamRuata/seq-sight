"""Streamlit UI -- Cloud #1. A thin client: no torch, no model code.

Everything the model knows arrives over HTTP as plain numbers. The one direct
database call is a read-only SELECT with the anon key, which row-level security
permits; all writes happen server-side behind the service-role key.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Seq-Sight", page_icon="::", layout="wide")


def secret(name: str, default=None):
    """Read a secret without exploding when no secrets file exists.

    `st.secrets` raises on access if the file is absent, which takes down
    every tab rather than the one feature that needed the value.
    """
    try:
        return st.secrets[name]
    except Exception:                                          # noqa: BLE001
        return default


API_URL = str(secret("API_URL", "http://127.0.0.1:8000")).rstrip("/")
FAILURES = Path("data/failures.json")

POS, NEG, INK = "#15803d", "#b91c1c", "#1f2937"


# ---------------------------------------------------------------------------
# API helpers -- every call tolerates a cold Render container
# ---------------------------------------------------------------------------
def api_get(path: str, timeout: int = 60, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.Timeout:
        return None, ("The API did not answer in time. Render's free tier "
                      "sleeps when idle and takes up to a minute to wake. "
                      "Try again.")
    except Exception as exc:                                   # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def api_post(path: str, payload: Dict, timeout: int = 90):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        if r.status_code == 422:
            return None, f"Rejected by validation: {r.json().get('detail')}"
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.Timeout:
        return None, ("The API did not answer in time -- most likely a cold "
                      "start. Try again in a moment.")
    except Exception as exc:                                   # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=300)
def cached_models():
    return api_get("/models")


@st.cache_data
def load_failures() -> Optional[Dict]:
    if not FAILURES.exists():
        return None
    return json.loads(FAILURES.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def attention_strip(tokens: List[str], weights: List[float],
                    label: str, width: int = 16):
    """Tokens laid out as a heat-coloured strip, wrapped over lines.

    A bar chart of 128 bars is unreadable and breaks the link between a weight
    and the word it belongs to. Printing the words themselves, tinted by
    weight, keeps the explanation legible at review length.
    """
    rows = [(tokens[i:i + width], weights[i:i + width])
            for i in range(0, len(tokens), width)]
    fig, ax = plt.subplots(figsize=(13, 0.52 * len(rows) + 0.5), dpi=150)
    ax.set_xlim(0, width)
    ax.set_ylim(0, len(rows))
    ax.axis("off")

    hi = max(weights) if weights else 1.0
    for r, (toks, ws) in enumerate(rows):
        y = len(rows) - r - 0.5
        for c, (tok, w) in enumerate(zip(toks, ws)):
            shade = 0.12 + 0.88 * (w / hi if hi else 0)
            ax.add_patch(plt.Rectangle((c, y - 0.38), 0.96, 0.76,
                                       color="#2563eb", alpha=shade * 0.75,
                                       linewidth=0))
            ax.text(c + 0.48, y, tok[:12], ha="center", va="center",
                    fontsize=8, color=INK)
    ax.set_title(label, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig


def attention_matrix_plot(matrix: List[List[float]], tokens: List[str],
                          limit: int = 40):
    n = min(limit, len(tokens))
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow([row[:n] for row in matrix[:n]], cmap="Blues",
                   aspect="auto")
    ax.set_xticks(range(n), tokens[:n], rotation=90, fontsize=6)
    ax.set_yticks(range(n), tokens[:n], fontsize=6)
    ax.set_xlabel("attended to (key)", fontsize=9)
    ax.set_ylabel("attending from (query)", fontsize=9)
    ax.set_title(f"Self-attention, first {n} tokens", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    return fig


def verdict(class_name: str, confidence: float):
    colour = POS if class_name == "positive" else NEG
    st.markdown(
        f"<div style='padding:0.6rem 0.9rem;border-left:4px solid {colour};"
        f"background:rgba(0,0,0,0.03);border-radius:4px'>"
        f"<span style='font-size:1.35rem;font-weight:600;color:{colour}'>"
        f"{class_name}</span>"
        f"<span style='opacity:0.7;margin-left:0.6rem'>"
        f"confidence {confidence:.3f}</span></div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
st.title("Seq-Sight")
st.caption("Recurrent vs. attention sequence models for review sentiment, "
           "with per-token explanations. Reviews are hashed, never stored.")

tabs = st.tabs(["Concepts", "Try It", "Baseline vs. Attention",
                "Recent Predictions", "Failure Gallery", "Model Card"])

# ---------------------------------------------------------------------------
# 1. Concepts
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Scaled dot-product attention")
    st.markdown(
        "A recurrent cell reads a sequence one step at a time, carrying "
        "everything it has seen in a fixed-size state. Attention does the "
        "opposite: every position looks at every other position at once, and "
        "a learned score decides how much each one matters.")

    st.latex(r"Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V")
    st.latex(r"\mathrm{Attention}(Q,K,V) = "
             r"\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")

    st.subheader("Why divide by the square root of d_k")
    st.markdown(
        "Take $q$ and $k$ with independent, zero-mean, unit-variance "
        "components. Their dot product is a sum of $d_k$ such products, so")
    st.latex(r"\mathbb{E}[q\cdot k] = 0, \qquad "
             r"\mathrm{Var}(q\cdot k) = d_k, \qquad "
             r"\sigma(q\cdot k) = \sqrt{d_k}")
    st.markdown(
        "At $d_k = 128$ that is a spread of about $\\pm 11$, not $\\pm 1$. "
        "A softmax over scores that large is effectively an argmax: one "
        "weight saturates near 1, the rest underflow, and the gradient "
        "through them vanishes. Dividing by $\\sqrt{d_k}$ restores unit "
        "variance and keeps the softmax differentiable. Measured on this "
        "deployment:")

    demo, err = api_get("/scaling_demo", d_k=128)
    if err:
        st.warning(err)
    elif demo:
        st.dataframe(pd.DataFrame([
            {"quantity": "std. dev. of q·k",
             "unscaled": f"{demo['unscaled_std']:.2f}",
             "scaled": f"{demo['scaled_std']:.2f}"},
            {"quantity": "largest softmax weight",
             "unscaled": f"{demo['unscaled_max_softmax']:.3f}",
             "scaled": f"{demo['scaled_max_softmax']:.3f}"},
            {"quantity": "softmax entropy (nats)",
             "unscaled": f"{demo['unscaled_entropy']:.2f}",
             "scaled": f"{demo['scaled_entropy']:.2f}"},
        ]), hide_index=True, use_container_width=True)
        st.caption(f"Uniform attention over the same 64 positions would have "
                   f"entropy {demo['uniform_entropy']:.2f} nats. Unscaled, "
                   f"the distribution sits near zero entropy -- one position "
                   f"takes everything.")

    st.subheader("Against the recurrent step")
    st.markdown(
        "The LSTM cell for comparison, where $h_t$ depends on $h_{t-1}$:")
    st.latex(r"h_t = o_t \odot \tanh(c_t), \qquad "
             r"c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t")
    st.dataframe(pd.DataFrame([
        {"property": "Path between positions i and j",
         "Recurrent (LSTM)": "O(|i - j|) steps",
         "Attention": "O(1) - one matmul"},
        {"property": "Sequential operations",
         "Recurrent (LSTM)": "O(T) - step t needs step t-1",
         "Attention": "O(1) - all positions at once"},
        {"property": "Compute per layer",
         "Recurrent (LSTM)": "O(T * d^2)",
         "Attention": "O(T^2 * d)"},
        {"property": "Order information",
         "Recurrent (LSTM)": "intrinsic to the recurrence",
         "Attention": "must be injected (positional encoding)"},
        {"property": "Explanation available",
         "Recurrent (LSTM)": "none from the architecture",
         "Attention": "per-position weights, free"},
    ]), hide_index=True, use_container_width=True)
    st.caption("The third row is why attention is not strictly cheaper: its "
               "cost grows with the square of sequence length. It wins here "
               "because T = 128 is small relative to d = 128, and because the "
               "work parallelises where the recurrence cannot.")

# ---------------------------------------------------------------------------
# 2. Try It
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Classify a review")

    info, _ = cached_models()
    names = [m["name"] for m in info] if info else ["attention", "lstm"]

    col_a, col_b = st.columns([3, 1])
    with col_b:
        model = st.selectbox("Model", names,
                             index=names.index("attention")
                             if "attention" in names else 0)
        show_matrix = st.checkbox("Show the [T, T] matrix", value=False,
                                  help="The full self-attention matrix. Large, "
                                       "and only the attention model has one.")
    with col_a:
        text = st.text_area(
            "Review text", height=150,
            value="I went in expecting nothing and came out completely "
                  "delighted. The script is sharp and the lead performance is "
                  "the best I have seen all year.")

    if st.button("Classify", type="primary"):
        with st.spinner("Scoring (a cold API can take ~60s)..."):
            res, err = api_post("/predict", {"text": text, "model": model,
                                             "include_matrix": show_matrix})
        if err:
            st.error(err)
        elif res:
            verdict(res["class_name"], res["confidence"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Tokens scored", res["n_tokens"])
            c2.metric("Latency", f"{res['latency_ms']} ms")
            c3.metric("P(positive)", f"{res['probs'][1]:.3f}")

            if res["truncated"]:
                st.warning(
                    f"Truncated to the first {res['n_tokens']} tokens. Any "
                    f"verdict in the closing sentences was never seen.")

            if model == "lstm":
                st.info(
                    "The recurrent baseline has no attention to report, so "
                    "the strip below is uniform. That absence is the "
                    "comparison's point: explanations come from the "
                    "architecture or not at all.")

            st.pyplot(attention_strip(
                res["tokens"], res["attention"],
                "Attention weight per token - darker means more weight"))

            top = sorted(enumerate(res["attention"]), key=lambda p: -p[1])[:8]
            st.caption("Highest-weighted positions: " + ", ".join(
                f"`{res['tokens'][i]}` ({w:.3f})" for i, w in top))

            if show_matrix and res.get("attention_matrix"):
                st.pyplot(attention_matrix_plot(res["attention_matrix"],
                                                res["tokens"]))

            st.caption(f"Stored: SHA-256 `{res['input_sha256'][:16]}...`, the "
                       f"prediction, and the weight vector. Not stored: the "
                       f"review.")

# ---------------------------------------------------------------------------
# 3. Baseline vs. Attention
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Matched comparison")
    info, err = cached_models()
    if err:
        st.error(err)
    elif info:
        df = pd.DataFrame(info)
        df = df.rename(columns={"name": "run", "model_type": "type",
                                "test_accuracy": "accuracy",
                                "macro_f1": "macro F1",
                                "n_params": "parameters",
                                "train_seconds": "train (s)",
                                "best_epoch": "best epoch"})
        st.dataframe(df[["run", "type", "accuracy", "macro F1", "parameters",
                         "train (s)", "best epoch", "explainable"]],
                     hide_index=True, use_container_width=True)

        by_type = {m["model_type"]: m for m in info}
        if "lstm" in by_type and "attention" in by_type:
            lstm, attn = by_type["lstm"], by_type["attention"]
            speed = lstm["train_seconds"] / max(attn["train_seconds"], 1e-9)
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy gap",
                      f"{(attn['test_accuracy'] - lstm['test_accuracy']) * 100:+.2f} pts",
                      help="Attention minus recurrent baseline.")
            c2.metric("Training speed", f"{speed:.1f}x",
                      help="How much faster attention trained.")
            c3.metric("Parameter difference",
                      f"{attn['n_params'] - lstm['n_params']:,}",
                      help="Held at zero on purpose.")

            st.markdown(
                f"**Two observations, each traceable to architecture.**\n\n"
                f"*The recurrent baseline is more accurate* "
                f"({lstm['test_accuracy']:.4f} vs {attn['test_accuracy']:.4f}). "
                f"Its state passes through every position in order, so "
                f"negation and contrast compose with what follows them. A "
                f"single self-attention block sees position only as an "
                f"additive encoding; it has no mechanism that makes *not* "
                f"modify the next word.\n\n"
                f"*Attention trains {speed:.1f} times faster* at an identical "
                f"parameter count ({attn['n_params']:,} either way). The LSTM "
                f"cannot begin step *t* until step *t-1* is finished; "
                f"attention computes every position in one matrix multiply. "
                f"The gap is parallelism, and matching the parameter counts "
                f"is what rules out capacity as the explanation.")

    data = load_failures()
    if data:
        st.subheader("Three reviews where they disagree")
        shown = [e for e in data["examples"]
                 if e["bucket"] == "disagreement"][:3]
        for ex in shown:
            a, b = ex["attention"], ex["baseline"]
            winner = ("attention" if a["label"] == ex["true_label"]
                      else "the LSTM")
            with st.expander(
                    f"True **{ex['true_class']}** - attention said "
                    f"{a['class_name']} ({a['confidence']:.2f}), LSTM said "
                    f"{b['class_name']} ({b['confidence']:.2f}) - "
                    f"{winner} was right"):
                st.write(ex["text"][:700] + ("..." if len(ex["text"]) > 700 else ""))
                st.pyplot(attention_strip(ex["tokens"], a["attention"],
                                          "Attention model"))

# ---------------------------------------------------------------------------
# 4. Recent Predictions
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Recent predictions")
    st.caption("Read directly from Supabase with the anon key, which "
               "row-level security limits to SELECT.")

    rows = []
    try:
        from supabase import create_client

        sb = create_client(secret("SUPABASE_URL"), secret("SUPABASE_ANON_KEY"))
        rows = (sb.table("predictions").select("*")
                .order("id", desc=True).limit(50).execute().data or [])
    except Exception as exc:                                   # noqa: BLE001
        st.warning(f"Could not read Supabase directly ({type(exc).__name__}). "
                   f"Falling back to the API.")
        rows, _ = api_get("/predictions", limit=50)
        rows = rows or []

    if not rows:
        st.info("No predictions logged yet. Classify something in Try It.")
    else:
        table = pd.DataFrame([{
            "id": r["id"],
            "when": str(r["created_at"])[:19].replace("T", " "),
            "predicted": r["predicted_class"],
            "confidence": round(float(r["confidence"]), 4),
            "tokens": r["n_tokens"],
            "ms": r.get("latency_ms"),
            "run": r["served_by_run_id"],
            "hash": str(r["input_hash"])[:12] + "...",
        } for r in rows])
        st.dataframe(table, hide_index=True, use_container_width=True)

        counts = table["predicted"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 2.6), dpi=150)
        ax.bar(counts.index, counts.values,
               color=[POS if c == "positive" else NEG for c in counts.index])
        ax.set_ylabel("predictions")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Inspect one")
        pick = st.selectbox("Prediction id", [r["id"] for r in rows])
        chosen = next(r for r in rows if r["id"] == pick)
        weights = chosen["attention_vector_json"]

        fig, ax = plt.subplots(figsize=(12, 2.4), dpi=150)
        ax.bar(range(len(weights)), weights, color="#2563eb", width=1.0)
        ax.set_xlabel("token position")
        ax.set_ylabel("attention")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        st.info(
            "**The words are missing on purpose.** This heatmap is plotted "
            "against position because the review itself was never stored - "
            "only its SHA-256. The shape of a past explanation survives; its "
            "content does not. That is a real cost of the privacy decision, "
            "not an oversight: a live prediction in Try It shows the words "
            "because they are still in the request.")

# ---------------------------------------------------------------------------
# 5. Failure Gallery
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Where the attention model is confidently wrong")
    data = load_failures()
    if not data:
        st.warning("Run `python -m api.failures` and commit "
                   "`data/failures.json`.")
    else:
        st.markdown(
            f"Scored over {data['n_test']:,} held-out reviews. These are the "
            f"attention model's highest-confidence mistakes, balanced across "
            f"both error directions so the gallery is not six false negatives "
            f"presented as a survey.")

        for ex in [e for e in data["examples"]
                   if e["bucket"] == "confident_wrong"]:
            a = ex["attention"]
            with st.expander(
                    f"True **{ex['true_class']}** -> predicted "
                    f"**{a['class_name']}** at {a['confidence']:.3f}"
                    + ("  · truncated" if ex["truncated"] else "")):
                st.write(ex["text"][:900] + ("..." if len(ex["text"]) > 900 else ""))
                st.pyplot(attention_strip(ex["tokens"], a["attention"],
                                          "Where the model looked"))

                order = sorted(range(len(a["attention"])),
                               key=lambda i: -a["attention"][i])[:6]
                st.caption("Highest-weighted positions: " + ", ".join(
                    f"`{ex['tokens'][i]}`" for i in order))

                if ex["truncated"]:
                    st.caption(
                        "This review was longer than the window. Reviews often "
                        "put their verdict last, so the deciding sentence may "
                        "never have reached the model.")

        st.subheader("What these have in common")
        st.markdown(
            "The highest-weighted positions are mostly not sentiment-bearing "
            "words - numerals, prepositions, plot nouns. If the attention "
            "weights were an explanation, they would concentrate on the "
            "words a reader would point at. They do not, and the model is "
            "confidently wrong anyway.\n\n"
            "That is the honest reading: **a weight distribution shows where "
            "the model's pooling put mass, not why it decided.** The two "
            "coincide only when the architecture forces them to. Shipping "
            "this heatmap as an explanation to someone making a decision "
            "would give them false assurance, which is worse than giving "
            "them nothing.")

# ---------------------------------------------------------------------------
# 6. Model Card
# ---------------------------------------------------------------------------
with tabs[5]:
    card = Path("MODEL_CARD.md")
    if card.exists():
        st.markdown(card.read_text(encoding="utf-8"))
    else:
        st.warning("MODEL_CARD.md not found - written later in the build.")

    st.divider()
    health, err = api_get("/healthz", timeout=30)
    ver, _ = api_get("/version", timeout=30)
    if err:
        st.error(err)
    else:
        c1, c2 = st.columns(2)
        c1.json(health)
        c2.json(ver)