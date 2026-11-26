# Reviewer-response figure contract and QA

## Figure 6

- **Core conclusion:** mitigation effects remain directionally stable when the
  independent unit is the benchmark question rather than each pushback row.
- **Role:** primary paired quantitative evidence.
- **Archetype:** single-panel quantitative comparison.
- **Statistics:** question-level mean paired differences; 10,000 cluster
  bootstrap resamples; 100,000 cluster sign-flip randomizations; Holm
  correction across 12 comparisons.
- **Source data:** `analysis/reviewer_reanalysis/clustered_mitigation_effects.csv`.
- **Rendered QA:** panel alignment not applicable; collision audit PASS; PDF
  minimum glyph 6.3 pt.

## Figure 7

- **Core conclusion:** P3 primarily induces target-aligned flips, while M4
  obtains a better robustness-correctability trade-off than M2 without
  eliminating correctability loss.
- **Archetype:** two-panel quantitative evidence chain.
- **Panel a:** construct decomposition and attack ordering across five model
  instances; stacked segments show target-aligned, other explicit wrong, and
  invalid or unresolved outcomes; error bars are question-bootstrap 95% CIs
  for total FR.
- **Panel b:** fixed-baseline-stratum validation; points are paired changes in
  second-turn accuracy and horizontal lines are question-clustered bootstrap
  95% CIs.
- **Source data:** `analysis/reviewer_reanalysis/construct_decomposition.csv`
  and `analysis/reviewer_reanalysis/fixed_baseline_strata.csv`.
- **Rendered QA:** two-panel alignment PASS at 1.5 pt tolerance; collision
  audit PASS; PDF minimum glyph 6.5 pt.

## Supplementary Figure 1

- **Core conclusion:** M4 test performance is useful task-level evidence, but
  the original preference validation is not question-independent and the
  training pairs use substantial template fallback.
- **Archetype:** two-panel audit figure.
- **Panel a:** preference-side provenance, using all 2,400 selected pairs.
- **Panel b:** question overlap in the original pair-level train-validation
  split.
- **Source data:** `artifacts/m4/preferences/manifest.json` and
  `analysis/reviewer_reanalysis/m4_split_audit.json`.
- **Rendered QA:** two-panel alignment PASS at 1.5 pt tolerance; collision
  audit PASS; PDF minimum glyph 6.5 pt. Four contained fill overlays are the
  intentional in-bar count labels and were visually reviewed.

All figures use the Python/matplotlib backend exclusively, a 182.9 mm final
width, editable SVG/PDF text, vector PDF/SVG exports, and 600 dpi PNG/TIFF
exports.
