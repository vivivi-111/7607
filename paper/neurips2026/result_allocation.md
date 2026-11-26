# Main-text result allocation

## One-sentence argument

In two-turn factual QA, selective updating must be measured by independently varying feedback truth and social cues; this controlled design reveals model-specific sensitivity to authority and distinguishes interventions that evaluate feedback from those that merely suppress answer changes.

## Shortest sufficient evidence chain

1. The original adaptive benchmark exposes frequent wrong-answer reversals but cannot identify the causal effect of authority or evidence because feedback truth, target naming, and first-answer state are coupled.
2. A \(2\times2\times2\times2\) controlled benchmark independently varies initial-answer correctness, feedback correctness, authority, and rationale over 538 questions per model.
3. Baseline models adopt correct feedback more often than incorrect feedback, but the magnitude and authority interaction vary by model.
4. On Qwen2.5-7B-Instruct, M1 modestly improves truth discrimination, whereas M2 produces a much larger gain by reducing incorrect-feedback adoption while retaining substantial correct-feedback adoption.
5. M4 is reported only after question-disjoint, real-response-only, three-seed revalidation and an independent MMLU/GSM8K capability check.

## Allocation

- **Core discovery — main text:** controlled truth-discrimination effects and authority-by-truth interactions across four comparable open text models.
- **Core intervention evidence — main text:** Qwen2.5-7B baseline versus M1, M2, and revalidated M4 under the controlled design.
- **Necessary qualification — main text:** the original FR/CR protocol is observational; \(D=CR-FR\) is a balance summary, not independent evidence of discernment.
- **Necessary qualification — main text:** model heterogeneity; do not claim that authority universally impairs selective updating.
- **Necessary support — main text, brief:** original observational FR/CR results establish practical prevalence and connect to prior benchmark protocols.
- **Robustness — appendix/SI:** fixed-baseline strata, metric-weight sensitivity, all factorial contrasts, and per-seed tables.
- **Measurement provenance — appendix/SI:** retrospective 200-response extraction audit and the fresh locked audit package awaiting new human labels.
- **Qualitative context — appendix or one compact main-text paragraph:** 100 adjudicated failure categories; subgroup patterns are descriptive because each subgroup has \(n=20\).
- **Exploratory only — appendix/SI:** API-served qwen3.8-27b; exclude it from the primary open-text-model macro-average.

## Claim boundaries

- English templated two-turn factual QA only.
- Authority and rationale effects are prompt- and model-dependent.
- Controlled initial answers test response policy conditional on supplied conversational state; they do not estimate each model's natural first-turn knowledge.
- Single-model mitigation evidence does not establish architecture-wide generalization.
- A three-seed training estimate quantifies run variability but is not a hyperparameter sweep.
