# Reproduction notes

This document records the *precise* process used to reproduce each figure's
values from the full research record, so a third party does not have to
rediscover it. Every figure regenerates deterministically with `make figures`
(no model weights). What follows is *how we know* each regeneration is faithful.

## Figure 2 — confidence distributions (direct)

The most straightforward. The full record's `regenerate_confidence_distributions.py`
reads `exp27_bounded_verification_20260206_205725.csv` and histograms the
`self_report_confidence` column by `category` (knowable / unknowable) for each of
the four model families. `scripts/fig2_confidence_distributions.py` is that code,
unchanged except for paths. The committed CSV is the same one; the figure is a
pure function of it.

## Figure 3 — budget curve (regenerated from a seeded simulation)

This figure required the most care; see also
[`PROVENANCE.md`](PROVENANCE.md#budget-curve-provenance-figure-3).

1. **The submitted numbers were orphaned.** The full record's
   `regenerate_budget_curve.py` had the four series' values hard-coded. Those
   numbers match **no** committed run — not the realistic Monte Carlo output, not
   the oracle output. They came from a run that is not in the record.

2. **The real simulation was located.** `experiment27_realistic_verification.py`
   is the source: 1000 Monte Carlo trials, `seed=42`, budgets 10/20/30%,
   probabilistic verification calibrated to the 75/80 human evaluation
   (`P(catch fabrication)=0.938`, `P(retrieve answer)=0.95`).

3. **Determinism was verified.** Re-running it produced a CSV byte-identical to
   the one archived on 2026-02-14. The pipeline is fully deterministic.

4. **The three judges were pinned down:**
   - *State-object-guided* ranks by the state object's `mean_entropy`.
   - *Text-guided (length)* ranks by **raw word count** of the response — the
     strongest text-channel signal. (The submitted figure's text baseline was
     length-only, not the full record's hedge+length `_fixed` scorer; raw word
     count is the faithful length-only judge.)
   - *Composed* is the entropy judge plus a citation-aware override. **Subtlety:**
     the full record's `is_citation_query` marker list contains a bare `'`
     (apostrophe), which makes its *effective* rule "starts with `summarize:` or
     contains an apostrophe." An initial port that required a venue name instead
     flagged fewer queries and shifted the Composed series by ~1pp. Running the
     figure and diffing against the sim caught this; the rule here matches the
     effective original.

5. **Resulting reproducible values** (`make figures` prints these):

   | Series (10/20/30%) | Value |
   |---|---|
   | No judge | 75.8 / 75.8 / 75.8 |
   | Text-guided (length) | 78.5 / 82.1 / 87.5 |
   | State-object-guided | 81.7 / 86.8 / 90.9 |
   | Composed | 80.2 / 87.1 / 91.5 |

   These differ from the originally-submitted figure by <1pp per point and change
   no claim. The paper figure is being updated to match.

## Figure 1 — entropy trace (rendered from a committed trace)

1. **The raw data is committed.** `data/entropy_trace_wombat.json` is the 43-token
   trace for "What shape is wombat scat?", extracted from the full record's
   `epistemic_trace_demo_20260208_184656.json` (`traces[1]`). Each token carries
   its raw generation entropy (nats).

2. **The entropy→bin(0–6) thresholds were reverse-engineered, not guessed.** The
   full record commits both the raw entropies (JSON) and the paper's rendered
   colours (`epistemic_trace_latex.tex`). Pairing them, the bins are cleanly
   monotone in entropy with a gap between every level:

   | bin | entropy range (nats) |
   |---|---|
   | 0 | ≤ 0.14 |
   | 1 | 0.24 – 0.35 |
   | 2 | 0.58 – 0.95 |
   | 3 | 1.12 – 1.49 |
   | 4 | 1.63 – 1.85 |
   | 5 | 2.19 – 2.68 |
   | 6 | 3.62 – 4.32 |

   `scripts/fig1_entropy_trace.py` uses fixed edges placed in those gaps
   (`[0.19, 0.46, 1.04, 1.56, 2.02, 3.15]`), which reproduce the paper's colours
   with **zero** bin mismatches across all rendered tokens.

3. **The `<|endoftext|>` token is excluded from rendering** but counted in the
   header statistics ("43 tokens, mean H=1.147, max H=4.315"), matching the paper,
   which renders 42 visible tokens.

## Verifying it yourself

```bash
make figures
```

- fig1: diff `figures/fig1_entropy_trace.tex`'s `\etok{bin}{...}` sequence against
  the paper's trace — the bins match token-for-token.
- fig3: the printed table matches the values above exactly (the simulation is seeded).
- fig2: the PDF matches the paper's confidence-distribution figure.
