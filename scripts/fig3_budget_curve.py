#!/usr/bin/env python3
"""Figure 3: end-to-end accuracy vs verification budget.

Given a fixed budget to verify some fraction of answers, which items should you
spend it on? A judge that ranks by the state object's entropy (telemetry) beats
one that ranks by a text-channel signal (response length), and the gap grows
with budget. "Composed" adds a citation-aware override on top of the entropy
judge.

This script is self-contained: it runs the probabilistic-verification Monte
Carlo simulation (seed=42, 1000 trials) directly over the committed evaluation
data and plots the result. It requires no model weights.

  * Numbers are regenerated here, not hard-coded. See REPRODUCTION.md for how
    these values relate to the originally-submitted figure (they differ by
    <1pp; the pipeline is now deterministic).

Data:   data/exp27_bounded_verification_20260206_205725.csv (raw answers + entropy)
        data/exp27b_detailed_20260206_230203.csv           (stratified-evaluator labels)
Output: figures/fig3_budget_curve.{pdf,png}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEGEND_SIZE=15
LABEL_SIZE=18
TICK_SIZE=15
FIG_W_SIZE=8
FIG_H_SIZE=4

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "exp27_bounded_verification_20260206_205725.csv"
DETAILED = ROOT / "data" / "exp27b_detailed_20260206_230203.csv"
OUT_STEM = ROOT / "figures" / "fig3_budget_curve"

# Verification success probabilities (from 75/80 human calibration).
P_VERIFY_UNKNOWABLE = 0.938   # catch a fabrication on an unknowable query
P_VERIFY_KNOWABLE = 0.95      # retrieve the answer on a knowable query
N_TRIALS = 1000
RANDOM_SEED = 42
BUDGET_LEVELS = [0.10, 0.20, 0.30]

# A query is treated as a citation/paper query if it starts with "summarize:"
# or contains an apostrophe (a paper title in quotes). This matches the
# effective behaviour of the full-record evaluator, whose marker list included
# a bare "'" that subsumes the venue-name markers. See REPRODUCTION.md.


# --- judges: higher score => more likely to be picked for verification --------

def state_object_score(row):
    """Telemetry judge: the state object's mean generation entropy."""
    return row["mean_entropy"]


def text_length_score(row):
    """Text-channel baseline: raw word count of the response (strongest
    behavioural signal; per-model AUC 0.85-0.97)."""
    return len(str(row.get("response", "")).split())


def is_citation_query(query):
    q = str(query)
    if q.lower().startswith("summarize:"):
        return True
    return "'" in q


def check_answer_correct(row):
    response = str(row.get("response", "")).lower().strip()
    expected = str(row.get("expected_answer", "")).lower().strip()
    if not row["is_knowable"]:
        refusals = ["i don't know", "i cannot", "i'm not sure", "no information",
                    "cannot verify", "don't have", "not aware", "no record",
                    "i apologize", "i can't"]
        return any(m in response for m in refusals)
    if not expected:
        return True
    parts = expected.split("/") if "/" in expected else [expected]
    return any(p.strip() in response for p in parts)


# --- realistic (probabilistic) verification -----------------------------------

def evaluate_realistic(df, score_fn, budget_frac, rng, use_citation=False):
    n = len(df)
    budget = max(1, int(n * budget_frac))
    if score_fn is None:  # no judge
        return df["is_correct"].mean()

    scores = df.apply(score_fn, axis=1).values.astype(float)
    if use_citation:
        cit = df["is_citation"].values
        for i in range(n):
            if cit[i]:
                scores[i] = 0.0 if df.iloc[i]["is_knowable"] else 1.0

    verify = np.argsort(scores)[-budget:]
    corrected = df["is_correct"].values.copy()
    for idx in verify:
        if corrected[idx]:
            continue
        if not df.iloc[idx]["is_knowable"]:
            if rng.random() < P_VERIFY_UNKNOWABLE:
                corrected[idx] = True
        else:
            if rng.random() < P_VERIFY_KNOWABLE:
                corrected[idx] = True
    return corrected.sum() / n


def monte_carlo(df, score_fn, budget_frac, use_citation=False):
    rng = np.random.default_rng(RANDOM_SEED)
    accs = np.array([
        evaluate_realistic(df, score_fn, budget_frac, rng, use_citation)
        for _ in range(N_TRIALS)
    ])
    return accs.mean() * 100


def load_data():
    df = pd.read_csv(RAW)
    det = pd.read_csv(DETAILED)
    key = det[["family", "query", "is_knowable", "new_correct"]].copy()
    df = df.merge(key, on=["family", "query", "is_knowable"], how="left")
    df["is_correct"] = df["new_correct"].fillna(
        df.apply(check_answer_correct, axis=1)
    ).astype(bool)
    df["is_citation"] = df["query"].apply(is_citation_query)
    return df


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} rows across {df['model_id'].nunique()} models; "
          f"baseline accuracy {df['is_correct'].mean()*100:.1f}%")

    series = {
        "No judge":             (None, False),
        "Text-guided (length)": (text_length_score, False),
        "State-object-guided":  (state_object_score, False),
        "Composed":             (state_object_score, True),
    }
    data = {name: [round(monte_carlo(df, fn, b, cite), 1) for b in BUDGET_LEVELS]
            for name, (fn, cite) in series.items()}

    budgets = [int(b * 100) for b in BUDGET_LEVELS]
    print("\nBudget:        " + "  ".join(f"{b:>5d}%" for b in budgets))
    for name, vals in data.items():
        print(f"  {name:<22s} " + "  ".join(f"{v:>5.1f}" for v in vals))

    styles = {
        "No judge":             {"color": "gray", "marker": "s", "ls": "--", "lw": 1.5, "label": "No judge"},
        "Text-guided (length)": {"color": "#dc267f", "marker": "o", "ls": "-",  "lw": 2, "label": "Text-guided"},
        "State-object-guided":  {"color": "#785ef0", "marker": "^", "ls": "-",  "lw": 2, "label": "Tensor"},
        "Composed":             {"color": "#648fff", "marker": "D", "ls": "-.", "lw": 2, "label": "Composed"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W_SIZE, FIG_H_SIZE))
    for name, vals in data.items():
        s = styles[name]
        ax.plot(budgets, vals, label=s["label"], marker=s["marker"], linestyle=s["ls"],
                linewidth=s["lw"], color=s["color"], markersize=8)

    # annotate the state-object vs text gap
    for i, b in enumerate(budgets):
        print("Budget:", b)
        so = data["State-object-guided"][i]
        tx = data["Text-guided (length)"][i]
        ax.annotate(f"+{so - tx:.1f}pp", xy=(b, (so + tx) / 2), fontsize=TICK_SIZE,
                    color="black", ha="left", va="center", xytext=(b, (so + tx) / 2))

    ax.set_xlabel("Verification Budget (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel("End-to-End Accuracy (%)", fontsize=LABEL_SIZE)
    ax.set_xticks(budgets)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xlim(5, 35)
    ax.set_ylim(73, 95)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    OUT_STEM.parent.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        dest = f"{OUT_STEM}.{ext}"
        fig.savefig(dest, bbox_inches="tight", dpi=150)
        print(f"Saved: {Path(dest).relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
