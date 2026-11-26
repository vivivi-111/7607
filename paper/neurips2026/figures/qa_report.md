# Figure QA report

Backend: Python / Matplotlib only. Final width: 183 mm. All quantitative
source tables are exported under `source_data/`.

## Figure: controlled baselines

Core conclusion: all four open text models distinguish correct from incorrect
feedback, but the magnitude and authority interaction are heterogeneous.

| Panel | Evidence role | Center / interval | Unit | QA |
|---|---|---|---|---|
| a | Descriptive controlled adoption | Question averages | 538 paired questions | Pass |
| b | Primary truth-discrimination contrast | Estimate + question-clustered bootstrap 95% CI | Question | Pass |
| c | Boundary / heterogeneity test | Authority-by-truth estimate + question-clustered bootstrap 95% CI | Question | Pass |

- Source preflight: pass.
- Alignment gate: pass at 1.5 pt tolerance.
- Rendered glyph floor: 6.5 pt, above the 5 pt requirement.
- Collision audit: pass, no fail or warning.
- Final-size visual review: axes, model labels, uncertainty, zero references,
  and the hero Qwen2.5-14B effect are legible.

## Figure: controlled prompt mitigations

Core conclusion: independent self-review improves truth-selective updating
while retaining substantial correct-feedback adoption.

| Panel | Evidence role | Center / interval | Unit | QA |
|---|---|---|---|---|
| a | Controlled behavior comparison | Question averages | 538 paired questions | Pass |
| b | Paired mitigation effect | Difference + question-clustered bootstrap 95% CI | Question | Pass |
| c | Observational boundary check | FR and CR under adaptive denominators | Natural first-turn stratum | Pass |

- Source preflight: pass.
- Alignment gate: pass at 1.5 pt tolerance.
- Rendered glyph floor: 6.5 pt, above the 5 pt requirement.
- Collision audit: no failures; one `text-fill-edge` warning for the direct
  `Incorrect` label was visually reviewed. The label is outside the data line
  and remains fully legible; the detector catches proximity to the endpoint
  marker fill.
- Final-size visual review: the correct/incorrect series use shape as well as
  color, the paired M2 effect is visually dominant, and observational rates are
  clearly separated from the controlled estimand.

## Figure: M4 revalidation

Core conclusion: across all three question-disjoint DPO runs, M4 reduces
incorrect-feedback adoption while preserving correct-feedback adoption; the
independent capability holdout shows no significant regression.

| Panel | Evidence role | Center / interval | Unit | QA |
|---|---|---|---|---|
| a | Controlled adoption by feedback truth | Three seed points; mean + seed SD | Seed and paired question | Pass |
| b | Observational FR/CR boundary check | Three seed points; mean + seed SD | Natural first-turn stratum | Pass |
| c | Independent capability regression | Three paired seed estimates; mean + seed SD | Held-out question | Pass |

- Source preflight: 21 pass, 0 warning, 0 failure.
- Alignment gate: pass at 1.5 pt tolerance.
- Rendered glyph floor: 6.5 pt, above the 5 pt requirement.
- Collision audit: pass, with no failures or warnings.
- Final-size visual review: seed points, mean uncertainty, legends, zero
  reference, and all three dataset labels are legible. Color is redundant with
  marker shape in panels a and b.

## Export bundle

Each figure has editable PDF and SVG, 600-dpi TIFF, 300-dpi PNG preview,
alignment JSON and overlay, collision JSON and diagnostic overlay, plotting
source, and source-data CSV files. Diagnostic overlays are QA artifacts and
must not replace the manuscript PDFs.
