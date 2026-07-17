# pacmi26-observability

A minimal, self-contained reproduction artifact for the PACMI 2026 paper
**"Epistemic Observability in Language Models"** (Mason & Anand).

It ships the **state-object observability interface** and the code + data to
regenerate all three figures in the paper. It is curated from the full research
record ([DOI 10.5281/zenodo.21314137](https://doi.org/10.5281/zenodo.21314137));
see [`PROVENANCE.md`](PROVENANCE.md) for the exact chain and
[`REPRODUCTION.md`](REPRODUCTION.md) for how each figure's numbers were verified.

## The idea in one paragraph

A model's *testimony* — what it says, including how confident it claims to be —
is gameable: it can state a fabrication fluently and confidently. Its
*telemetry* — the entropy of the actual next-token distribution, the attention
of the actual forward pass — is not, because the model cannot present a
different computation than the one it performed. The **state object** exposes
that telemetry. Figure 1 makes this visible: on a query the model answers with a
confident-sounding fabrication, the state object still spikes in entropy exactly
where the model improvises.

## Quickstart

```bash
uv sync            # installs matplotlib, numpy, pandas (no model weights needed)
make figures       # regenerates all three figures into figures/
```

Every figure regenerates deterministically from committed data on a laptop — no
GPU, no model weights.

## The three figures

| Figure | Script | What it shows | Regenerated from |
|---|---|---|---|
| 1 — entropy trace | `scripts/fig1_entropy_trace.py` | Per-token entropy of a confident fabrication, as coloured LaTeX | `data/entropy_trace_wombat.json` |
| 2 — confidence distributions | `scripts/fig2_confidence_distributions.py` | Self-reported confidence overlaps for knowable vs unknowable, 4 model families | `data/exp27_bounded_verification_*.csv` |
| 3 — budget curve | `scripts/fig3_budget_curve.py` | End-to-end accuracy vs verification budget; telemetry judge beats the text baseline | `data/exp27_*.csv` (seeded Monte Carlo) |

## Two-tier reproduction

- **`make figures`** — regenerate the figures from committed data. Deterministic,
  laptop, no weights. This is what a reviewer or reader runs.
- **`make traces`** — re-run the model itself to regenerate the underlying
  entropy trace (`scripts/generate_trace.py`). Needs model weights and,
  realistically, a GPU: `uv sync --extra traces`. This is the "regenerate the
  data, not just the figure" path.

## Layout

```
src/observability/   state-object interface (StateObjectInterface -> StateObservation)
scripts/             one script per figure, plus generate_trace.py (tier 2)
data/                committed inputs the figures regenerate from
figures/             regenerated outputs (committed for convenience)
PROVENANCE.md        what this is curated from, and the budget-curve provenance note
REPRODUCTION.md      the precise, per-figure reproduction process
```

## License & citation

MIT (see [`LICENSE`](LICENSE)). If you use this, please cite the paper and the
software — see [`CITATION.cff`](CITATION.cff).
