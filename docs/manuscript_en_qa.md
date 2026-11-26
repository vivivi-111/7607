# English manuscript QA

## Argument

In a controlled factual pushback benchmark, we show that reliable mitigation
requires selective updating rather than answer persistence, supported by
question-clustered comparisons of four interventions and bounded by templated
English interactions, limited mitigation tuning, and retrospective extraction
validation.

## Claim–evidence map

- **Authority is the strongest implemented pushback.** Supported by the
  model-macro-averaged P3 FR of 65.0%, the P1 comparator of 15.3%, and the
  P3 target-aligned flip rate of 59.4%.
- **Scale does not monotonically reduce FR.** Supported by Qwen2.5 FR values
  of 66.6%, 48.3%, and 50.0% at 1.5B, 7B, and 14B.
- **M3 did not generalize in this implementation.** Supported by a paired
  second-turn accuracy change of −2.31 percentage points; the Holm-adjusted
  result is not significant (`p=0.114`), so no general harmful effect is
  claimed.
- **M4 provides the best tested Qwen2.5-7B trade-off.** Supported by
  +11.87 percentage points paired accuracy (95% CI +9.42 to +14.32;
  Holm-adjusted `p<0.001`) and a smaller correctability loss than M2.
- **Extraction passed its threshold retrospectively.** Supported by 200/200
  agreement with adjudicated labels (Wilson 95% CI 98.12–100.00%), explicitly
  qualified because the same audit informed rule revision.

## Main-text discipline audit

- Core evidence retained: pushback ordering, dataset and scale heterogeneity,
  FR–CR trade-off, M3/M4 paired effects, fixed baseline strata.
- Necessary qualifications retained: M3 multiplicity correction, M4
  train-validation overlap, qualitative subgroup size, retrospective
  extraction audit.
- Implementation and reproducibility details moved to Methods.
- Extended cases remain in `analysis/report_assets/case_studies.md` rather
  than enlarging the manuscript.
- Figures 1–7 and Supplementary Figure 1 are linked and present.

## Verification

- Manuscript: 4,477 words.
- Figures: 8/8 paths resolve.
- References: 9 entries, all inherited from the verified project report.
- PDF: 13 pages; all pages contain extractable text.
- No unsupported novelty language or universal claim was detected.

## Author input required before submission

1. Author names, order, affiliations, and corresponding-author email.
2. Permanent repository URL or DOI, software/data license, and access terms.
3. Target journal and its exact formatting, word-count, and disclosure rules.
4. Human-author review and approval of every claim, number, citation, and
   interpretation.
5. An AI-assisted writing disclosure appropriate to the selected journal;
   AI systems must not be listed as authors.
