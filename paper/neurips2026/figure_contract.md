# Figure contract

Backend: Python/Matplotlib exclusively. Target width: 183 mm. Minimum rendered glyph: 5 pt. Every multi-panel export requires a strict 1.5 pt alignment manifest, PDF glyph audit, collision audit, and panel-by-panel review.

## Figure 1 — Controlled measurement of selective updating

**Results-level question:** Can the design separate feedback truth from authority and rationale?

**Claim:** The factorial benchmark identifies adoption responses to feedback truth and social cues without conditioning those factors on the model's natural first-turn correctness.

- **a — setup:** \(2\times2\times2\times2\) design; role: task definition.
- **b — primary evidence:** correct- versus incorrect-feedback adoption for four open text models; role: establish truth discrimination and heterogeneity.
- **c — boundary:** authority-by-truth interaction with question-clustered 95% CIs; role: show that the authority effect is not universal.

Archetype: schematic-led quantitative composite. Panel b is the hero. A model-universal authority claim would be overturned by an interaction interval spanning or changing sides of zero; that boundary must remain visible.

## Figure 2 — Mitigations change response policy, not merely flip rates

**Results-level question:** Which interventions improve selective updating on Qwen2.5-7B-Instruct?

**Claim:** Controlled adoption contrasts distinguish genuine truth sensitivity from blanket answer persistence.

- **a — primary comparison:** incorrect-feedback adoption versus correct-feedback adoption for baseline, M1, M2, and revalidated M4.
- **b — paired effect:** truth-discrimination gain relative to baseline with question-clustered 95% CIs.
- **c — qualification:** observational second-turn accuracy or fixed-first-state effects, showing why the adaptive FR/CR protocol can rank methods differently.

Archetype: asymmetric quantitative grid. Panel a is the hero; consistent method colors are used across panels.

## Figure 3 — M4 revalidation and capability boundary

**Results-level question:** Does the M4 result survive data leakage controls and training randomness without broad capability loss?

**Claim:** Report only what is supported by three question-disjoint, real-response-only runs and the independent 1,000-question capability holdout.

- **a — seed robustness:** per-seed incorrect- and correct-feedback adoption,
  with mean \(\pm\) seed SD; their separation is the controlled truth
  discrimination contrast.
- **b — observational benchmark:** per-seed FR and CR, retaining both axes rather than only \(D\).
- **c — external validity boundary:** per-seed paired MMLU/GSM8K capability
  differences with mean \(\pm\) seed SD; question-level 95\% CIs remain in the
  source table and main text.

Archetype: validation envelope. If seed variability is large or capability CIs indicate material regression, the figure and main claim must foreground that failure boundary.

## Supplementary figures

- Observational P1–P5 results for four comparable open text models; descriptive only.
- Target-aligned versus other wrong flips and invalid outputs.
- Weight-sensitivity curves for \(D\) and final accuracy.
- Locked extraction-audit sampling composition and checksum manifest; no accuracy claim until new human labels are returned.
