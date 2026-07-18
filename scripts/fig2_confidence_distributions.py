#!/usr/bin/env python3
"""Figure 2: self-reported confidence distributions, knowable vs unknowable.

For each of four model families, histograms the model's *stated* confidence on
knowable vs unknowable queries. The distributions overlap heavily -- the text
(self-report) channel does not separate what the model knows from what it does
not. Regenerates deterministically from the committed evaluation data; no model
weights required.

Data: data/exp27_bounded_verification_20260206_205725.csv
Output: figures/fig2_confidence_distributions.{pdf,png}
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEGEND_SIZE=15
LABEL_SIZE=18
TICK_SIZE=15
FIG_W_SIZE=16
FIG_H_SIZE=8

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "exp27_bounded_verification_20260206_205725.csv"
OUT_STEM = ROOT / "figures" / "fig2_confidence_distributions"

FAMILIES = {
    "OLMo": "allenai/olmo-3-7b-instruct",
    "Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen3-4B-Instruct-2507",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def load_rows():
    with open(DATA_FILE) as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_SIZE, FIG_H_SIZE))
    fig.suptitle("Self-Reported Confidence: Knowable vs Unknowable",
                 fontsize=LABEL_SIZE, fontweight="bold")

    color_know = "#fe6100"     # orange
    color_unknow = "#785ef0"   # purple
    bins = np.linspace(0, 1, 20)

    for ax, (label, model_id) in zip(axes.flat, FAMILIES.items()):
        know = [float(r["self_report_confidence"]) for r in rows
                if r["model_id"] == model_id and r["category"] == "knowable"]
        unknow = [float(r["self_report_confidence"]) for r in rows
                  if r["model_id"] == model_id and r["category"] == "unknowable"]

        ax.hist(know, bins=bins, alpha=0.7, label="Knowable", color=color_know,
                edgecolor="black", linewidth=0.5, hatch="//", density=True)
        ax.hist(unknow, bins=bins, alpha=0.7, label="Unknowable", color=color_unknow,
                edgecolor="black", linewidth=0.5, hatch="\\\\", density=True)

        ax.set_title(label, fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xlabel("Self-Reported Confidence", fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    OUT_STEM.parent.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        dest = f"{OUT_STEM}.{ext}"
        fig.savefig(dest, bbox_inches="tight", dpi=150)
        print(f"Saved: {Path(dest).relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
