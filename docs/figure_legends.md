# Figure legends for reviewer-response analyses

## Figure 6 | Mitigation effects after accounting for repeated pushbacks

Change in second-turn accuracy relative to each model-matched baseline on
common question-pushback pairs. Points are paired mean differences after first
averaging the five pushback outcomes within each independent benchmark
question; horizontal lines show question-clustered bootstrap 95% confidence
intervals (10,000 resamples). Positive values favour the mitigation. Cluster
sign-flip tests use 100,000 randomizations, with Holm correction across the 12
displayed mitigation comparisons. M3 and M4 were evaluated only for
Qwen2.5-7B. Source data:
`analysis/reviewer_reanalysis/clustered_mitigation_effects.csv`.

## Figure 7 | Wrong-flip decomposition and the robustness-correctability trade-off

**a,** Model-macro-averaged outcomes across all five baseline model instances.
Bars decompose the fraction of initially correct answers that become wrong
into target-aligned flips, other explicit wrong answers, and invalid or
unresolved outcomes. Error bars show question-bootstrap 95% confidence
intervals for total FR. P1 exposes no target answer; its target-aligned segment
therefore denotes coincidence with the hidden evaluation target, not adoption
of a user-proposed answer. **b,** Change in Qwen2.5-7B second-turn accuracy
relative to baseline after fixing strata by the baseline first-turn state.
Points compare M2 and M4; horizontal lines show question-clustered bootstrap
95% confidence intervals. M4 improves robustness less than M2 but incurs a
substantially smaller correctability loss. Source data:
`analysis/reviewer_reanalysis/construct_decomposition.csv` and
`analysis/reviewer_reanalysis/fixed_baseline_strata.csv`.

## Supplementary Figure 1 | Audit of M4 preference-data construction

**a,** Proportion of the 2,400 chosen and rejected response sides drawn from
real model candidates or constructed with a template fallback. Counts printed
inside the fallback segments are response sides, not preference pairs.
**b,** Question-level overlap between the original pair-level training and
validation split. Of 218 unique validation questions, 182 also contribute a
different pushback pair to training. The reported 85.4% validation preference
accuracy is therefore an internal pair-level diagnostic rather than an
independent question-level generalization estimate. The held-out benchmark
test split was not used to construct preferences. Source data:
`artifacts/m4/preferences/manifest.json` and
`analysis/reviewer_reanalysis/m4_split_audit.json`.
