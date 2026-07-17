# Provenance

This repository is a **curated artifact**, not the original research record. It
was assembled by selecting, from the full record, exactly the interface code and
data needed to regenerate the three figures in the PACMI 2026 paper.

## Source

- **Full research record:** https://github.com/fsgeek/ai-honesty
- **Archived:** [DOI 10.5281/zenodo.21314137](https://doi.org/10.5281/zenodo.21314137), commit `9799e22` (v1.0.0)

The full record contains the complete, unpruned experimental history — every
experiment, exploratory run, dead end, and raw trace. This artifact is a strict
subset of it. Nothing here contradicts the full record; where you want the
complete evidentiary trail (e.g. to check that a figure was not selected to
flatter), follow the DOI above.

## What was included

- `src/observability/interface.py` — the state-object interface, lifted from the
  full record's `scripts/tensor_interface.py` and renamed (see "Terminology").
- The three figures' generation scripts and their exact input data.

## What was intentionally excluded

Exploratory experiments beyond the three paper figures, review notes, WIP paper
drafts, presentation material, and internal tooling. These are process, not
evidence; they live in the full record.

## Terminology

This artifact uses **"state object"** for the thing exposed (the model's internal
computational state, as an inspectable object) and **"observability"** for the
framing. The full record and its TLA+ specification use the older word "tensor"
for the same construct; that term drew a (fair) objection that it has a specific
mathematical meaning the activation arrays do not satisfy. The paper's title
already uses "Epistemic Observability."

## Budget-curve provenance (Figure 3)

The figure originally submitted with the paper carried hard-coded numbers that
**do not reproduce from any run committed to the full record.** They came from an
earlier run that no longer exists in the record.

The committed simulation (`experiment27_realistic_verification.py` in the full
record, ported into `scripts/fig3_budget_curve.py` here) is **deterministic**: a
fresh run today is byte-identical to the run archived in February 2026. It
produces numbers that differ from the originally-submitted figure by **less than
1 percentage point**, and change **no claim** — the telemetry judge still beats
the text baseline, the gap still grows with budget.

Rather than preserve orphaned numbers, this artifact plots what the deterministic
pipeline actually produces, and the paper's figure is being updated to match.
This reflects roughly six months of improvement in the reproducibility of the
methodology between submission and release: the current pipeline is the more
trustworthy one. Full details are in [`REPRODUCTION.md`](REPRODUCTION.md).

## Reciprocal link

The full record's README points back to this artifact's DOI, so the provenance
chain is walkable in both directions.
