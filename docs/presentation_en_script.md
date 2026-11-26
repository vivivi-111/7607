# English presentation script

## Slide 1 — When Should a Language Model Stand Its Ground?

This study asks when a language model should resist a user and when it should change its mind. We evaluate that distinction rather than treating every answer change as failure. The central result is that authority-based pressure is the strongest implemented attack, and mitigation is useful only if it preserves the ability to accept genuine corrections. Balanced preference optimization offers the best trade-off for Qwen2.5-7B, whereas activation steering does not generalize in our small-data setting.

## Slide 2 — Robustness without correctability is just stubbornness

A robust assistant should not simply refuse to update. If its first answer is correct, it should resist unsupported pressure. If its first answer is wrong, it should accept a valid correction. These two cases create an evaluation tension: lowering the flip rate can look good while actually making the model stubborn. We therefore frame the task as discernment under pushback and ask four questions: which pushback is most harmful, whether mathematics is more robust, whether scale helps monotonically, and which interventions improve robustness without destroying correctability.

## Slide 3 — A controlled two-turn benchmark isolates the effect of pushback

The benchmark uses 545 held-out questions from MMLU, GSM8K, and TruthfulQA. Each question receives one round-one answer. We then reuse that answer across five pushback types: a simple challenge, a direct alternative, an authority claim, social pressure, and a generated rationale. The target is conditional on the first answer. Correct answers are challenged with a deterministic wrong target; wrong answers are challenged with the gold answer. This shared-round-one design isolates the effect of the pushback wording and avoids regenerating a different baseline response for each condition.

## Slide 4 — Three metrics separate resilience from selective updating

We report three linked metrics. Flip rate is the proportion of initially correct answers that become wrong after pushback, so lower is better. Correction rate is the proportion of initially wrong answers that become correct, so higher is better. Discernment is correction rate minus flip rate. It becomes positive only when valid updating exceeds harmful updating. For mitigation comparisons, the independent unit is the benchmark question because each question contributes up to five correlated pushbacks. We therefore use question-clustered bootstrap intervals and cluster sign-flip tests with Holm correction across 12 comparisons.

## Slide 5 — Authority claims are the strongest implemented attack

Authority-based pushback is the strongest implemented attack. Averaged across the five baseline model instances, P3 produces a 65.0 percent wrong-answer flip rate, compared with 15.3 percent for a simple challenge. Its target-aligned flip rate is also highest at 59.4 percent, so the result is not explained only by unrelated errors. Under these templates, a bare assertion of expertise is more damaging than the generated rationale in P5. The strong API-served model remains comparatively stable, but the ordering of attacks is visible across the local models.

## Slide 6 — Neither task type nor scale gives a simple robustness law

Two intuitive hypotheses receive qualified support rather than simple confirmation. GSM8K has the lowest macro-average baseline flip rate at 40.5 percent, versus 47.2 percent on MMLU and 45.8 percent on TruthfulQA. However, the datasets also differ in format, initial accuracy, domain, and wrong-target construction, so this does not identify mathematical verifiability as the cause. Scale is also non-monotonic: Qwen2.5-7B has 48.3 percent FR and Qwen2.5-14B has 50.0 percent. The larger model has better discernment because it corrects more initial errors, not because it flips less often.

## Slide 7 — Most inference-time gains trade away correctability

The robustness-correctability plot reveals why flip rate alone is inadequate. M2 usually moves points left, meaning fewer harmful flips, but also moves them down, meaning fewer valid corrections. For Qwen2.5-7B, M2 reduces FR from 48.3 to 22.1 percent, while CR falls from 67.0 to 44.0 percent. The strong API model starts with only 7.4 percent FR, so M1 and M2 have little room to improve and mainly reduce correctability. The useful region is upper-left, but interventions must be judged relative to each model's baseline.

## Slide 8 — Activation steering fails to generalize; preference tuning does

The paired question-level analysis separates genuine generalization from tuning success. Activation steering, M3, improves discernment on the small train tuning subset, but on test it reduces second-turn accuracy by 2.31 percentage points. The clustered 95 percent interval is minus 4.47 to minus 0.19, yet the Holm-adjusted p-value is 0.114. We therefore report a negative point estimate that is compatible with overfitting, not a statistically confirmed harmful mechanism. By contrast, M4 improves paired accuracy by 11.87 points with a 95 percent interval from 9.42 to 14.32 and adjusted p below 0.001.

## Slide 9 — M4 improves robustness while preserving more correctability

M4 gives the best benchmark-specific trade-off for Qwen2.5-7B. It lowers FR from 48.3 to 30.1 percent while retaining 63.3 percent CR, raising discernment from 0.187 to 0.331. The fixed baseline-stratum analysis shows what remains imperfect. On questions the baseline initially answered correctly, M4 raises second-turn accuracy by 18.9 points. On initially wrong questions, it lowers accuracy by 5.8 points. M2 gains more robustness, plus 26.1 points, but loses 23.0 points of correctability. M4 reduces rather than eliminates the stubbornness trade-off.

## Slide 10 — M4 succeeds on test, but its internal validation is not independent

The M4 result needs two important boundaries. First, 420 chosen sides and 803 rejected sides use template fallbacks rather than real candidate responses. Second, the original train-validation split was performed at the question-pushback pair level. Of 218 unique validation questions, 182 also contribute a different pushback to training, an 83.5 percent overlap. The reported 85.4 percent validation preference accuracy is therefore an internal pair-level diagnostic, not independent question-level validation. The held-out benchmark test remains separate, but future training should split preference data by question.

## Slide 11 — Wrong flips usually come with concession or fabricated reasoning

Wrong flips are often accompanied by new rationalizations rather than simple agreement. Two annotators independently coded 100 baseline failures. Before adjudication, agreement was 59 percent and Cohen's kappa was 0.417; all 41 disagreements were discussed. The final distribution is 42 percent partial concession, 31 percent fabricated reasoning, 17 percent direct concession, 8 percent hedging, and 2 percent other. Authority failures contain 45 percent fabricated reasoning. These subgroup patterns are descriptive because each model-by-pushback group contains only 20 samples.

## Slide 12 — Human audit validates extraction—and defines its boundary

Because the benchmark depends on automatic answer extraction, we conducted a separate 200-response stress-test audit with deliberate oversampling of NONE and HEDGE cases. The annotators agreed on 193 cases, giving 96.5 percent raw agreement and kappa 0.962; seven cases were adjudicated with written rationales. The audit exposed compound units, explicit rounding, invalid all-option answers, and echoed prompt instructions. After general rule revisions, the extractor matches all 200 adjudicated labels, with a Wilson lower bound of 98.12 percent. This is retrospective verification because the audit errors informed the rules.

## Slide 13 — The evidence supports a benchmark-specific, not universal, claim

The conclusions are bounded by the experimental design. The benchmark is English-only and uses templated two-turn pressure. Wrong targets are deterministic rather than selected as the most plausible distractor. P5 comes from one reference model. M3 uses only 36 tuning questions and one global direction. M4 uses template fallbacks, no hyperparameter sweep, and a leaky pair-level validation split. Qualitative subgroups are small, and the extraction audit is retrospective after rule revision. These limitations do not erase the held-out test comparisons, but they constrain causal and cross-domain claims.

## Slide 14 — Selective updating—not stubbornness—is the mitigation target

Three conclusions are most defensible. First, social cues matter: authority pressure is more damaging than a simple request to reconsider. Second, a low flip rate is not enough; every mitigation should be evaluated jointly on robustness and correctability. Third, balanced preference optimization gives the best tested trade-off for Qwen2.5-7B, while activation steering fails to generalize in this small-data implementation. The next decisive studies are human-written attacks, question-disjoint preference validation, independent capability benchmarks, and a new locked extraction audit.
