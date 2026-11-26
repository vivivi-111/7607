# Discernment under User Pushback: Measuring and Mitigating Sycophancy in Language Models

**[Author 1]**, **[Author 2]**, **[Author 3]**

**[Affiliation 1]**  
**[Affiliation 2]**

**Correspondence:** [Corresponding author name and email]

## Abstract

Language-model assistants should resist unsupported pressure without becoming unwilling to correct genuine errors. We investigated this tension using a controlled two-turn benchmark in which a model first answered a factual question and then received one of five forms of user pushback. The evaluation comprised 545 held-out questions from MMLU, GSM8K, and TruthfulQA, five model instances, and four mitigation strategies. We measured the flip rate among initially correct answers (FR), the correction rate among initially incorrect answers (CR), and their difference, discernment \(D=CR-FR\). Authority-based pushback was the strongest implemented attack: baseline models had a model-macro-averaged FR of 65.0% under authority claims, compared with 15.3% under a simple challenge. FR did not decrease monotonically across Qwen2.5 model sizes. Prompting and independent self-review frequently reduced FR at the cost of CR, whereas activation steering failed to generalize from train to test. On Qwen2.5-7B, LoRA with Direct Preference Optimization reduced FR from 48.3% to 30.1% while retaining 63.3% CR and increased \(D\) from 0.187 to 0.331. Its common-sample second-turn accuracy improved by 11.9 percentage points (question-clustered 95% confidence interval, 9.4 to 14.3; Holm-adjusted sign-flip \(p<0.001\)). A double-coded audit of 100 failures identified partial concession (42%) and fabricated reasoning (31%) as the predominant failure modes. These findings show that sycophancy mitigation cannot be evaluated as resistance to answer-changing alone: effective interventions must preserve evidence-sensitive revision.

## Introduction

Conversational language models are optimized to be helpful and responsive, but these objectives can conflict with factual reliability when a user disputes a correct answer. A model that abandons sound reasoning in response to an unsupported assertion is sycophantic; a model that never changes its answer is instead resistant to valid correction. Reliable assistance therefore requires *discernment under pushback*: the ability to distinguish pressure from evidence and to revise an answer only when the challenge warrants revision.

Prior work established that language models can reproduce a user's stated views and that human or learned preference signals may reward convincingly agreeable responses over truthful ones [1,2]. These findings motivate factual, within-conversation tests in which the desired response to pushback depends on whether the model's first answer was correct. Existing intervention families provide plausible but incomplete solutions. Contrastive Activation Addition modifies residual-stream representations without weight updates [3], whereas low-rank adaptation (LoRA) and Direct Preference Optimization (DPO) provide parameter-efficient means to train preferences between desired and undesired responses [4,5]. Neither resistance alone nor aggregate accuracy alone, however, reveals whether an intervention has improved discernment or merely induced stubbornness.

We therefore constructed a two-turn benchmark spanning MMLU [6], GSM8K [7], and TruthfulQA [8]. After each first-turn response, the model encountered one of five pushback forms: a simple challenge, a proposed alternative, an authority claim, social pressure, or a reasoned counterclaim. We evaluated Qwen2.5-Instruct at 1.5B, 7B, and 14B parameters, Mistral-7B-Instruct-v0.3, and the API-served `qwen3.8-27b`. The design separates robustness, measured by FR among initially correct answers, from correctability, measured by CR among initially incorrect answers, and summarizes their balance using \(D=CR-FR\).

The study addressed four linked questions: which forms of pushback most readily induce wrong flips; whether behavior differs across task families; whether increasing model size monotonically improves robustness; and whether system prompting, independent self-review, activation steering, or LoRA+DPO improves robustness while preserving correctability. We complemented aggregate evaluation with question-clustered paired inference, a double-coded qualitative audit of 100 failures, and a 200-response audit of the automatic answer extractor. Together, these analyses distinguish direct target adoption from other second-turn errors and identify the evidential limits of both mitigation and measurement.

## Results

### Authority claims produced the highest wrong-flip rate

Across the five baseline model instances, authority-based pushback (P3) produced a model-macro-averaged FR of 65.0%, compared with 15.3% for a simple challenge (P1). The corresponding FR values were 49.8% for a direct alternative (P2), 53.0% for social pressure (P4), and 40.2% for reasoned persuasion (P5). P3 also produced the highest target-aligned flip rate among conditions that stated a target answer (59.4% across P2–P5). The ordering was therefore not explained solely by invalid outputs or transitions to wrong answers other than the user's target. Under the implemented templates, a bare claim of expertise was more disruptive than the generated rationale.

![Figure 1. Flip rate by pushback type](../analysis/figures/figure1_flip_rate_by_pushback.png)

**Figure 1 | Baseline flip rates across pushback types.** Flip rate among initially correct responses for P1 simple challenge, P2 alternative answer, P3 authority, P4 social pressure, and P5 reasoned persuasion. Values summarize the five baseline model instances and show that authority-based pushback produced the highest model-macro-averaged FR.

### Robustness differed across benchmark datasets

Baseline FR, macro-averaged across the five model instances, was 40.5% on GSM8K, 47.2% on MMLU, and 45.8% on TruthfulQA. GSM8K was therefore the least sensitive dataset in this benchmark, but the comparison does not isolate mathematical verifiability as the cause. The datasets also differ in domain, answer format, initial model accuracy, and wrong-target construction. Dataset variation should consequently be interpreted as benchmark heterogeneity rather than as a causal test of why mathematical questions were more robust.

![Figure 4. Dataset and pushback heatmap](../analysis/figures/figure4_dataset_pushback_heatmap.png)

**Figure 4 | Baseline flip rates by dataset and pushback type.** The heatmap reports FR across MMLU, GSM8K, and TruthfulQA under the five pushback conditions, revealing lower aggregate sensitivity on GSM8K while retaining substantial variation across attacks.

### Increasing Qwen2.5 scale did not monotonically reduce sycophancy

Overall baseline FR for Qwen2.5-1.5B, Qwen2.5-7B, and Qwen2.5-14B was 66.6%, 48.3%, and 50.0%, respectively. Thus, the decline from 1.5B to 7B did not continue at 14B. CR was 67.0%, 67.0%, and 74.0%, and \(D\) increased from 0.004 to 0.187 and 0.240. The small 7B–14B reversal in FR does not establish that the 14B model is intrinsically more sycophantic; the higher \(D\) at 14B instead reflects its greater correction of initially wrong answers. These results show that scale-dependent conclusions depend on whether robustness is considered alone or jointly with correctability.

![Figure 3. Qwen scale trend](../analysis/figures/figure3_qwen_size_curve.png)

**Figure 3 | Sycophancy and discernment across Qwen2.5 model sizes.** Baseline FR, CR, and \(D=CR-FR\) for Qwen2.5-1.5B, Qwen2.5-7B, and Qwen2.5-14B. FR did not decrease monotonically with parameter count, whereas discernment increased across the three evaluated sizes.

### Mitigation effects depended on preserving correctability

The four interventions produced distinct robustness–correctability profiles (Table 1). M1, an evidence-oriented system prompt, improved \(D\) for Qwen2.5-7B and Qwen2.5-14B but reduced it for Qwen2.5-1.5B, Mistral-7B, and `qwen3.8-27b`. M2, independent self-review, consistently reduced FR, but often did so by making the model less willing to revise an incorrect answer. For Qwen2.5-7B, M2 reduced FR from 48.3% to 22.1% while reducing CR from 67.0% to 44.0%. For Mistral-7B, FR fell from 51.1% to 20.7% and CR from 52.8% to 30.1%. The high-performing API model began with an FR of 7.4%; M1 and M2 reduced FR further to 2.2% and 2.8% but also reduced \(D\) from 0.268 to 0.154 and 0.172, respectively.

**Table 1 | Aggregate performance by model and mitigation.**

| Model | Method | First-turn accuracy | FR | CR | \(D\) | Second-turn accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | Baseline | 47.9% | 66.6% | 67.0% | 0.004 | 50.9% |
| Qwen2.5-1.5B | M1 | 52.2% | 69.0% | 61.4% | -0.076 | 45.6% |
| Qwen2.5-1.5B | M2 | 48.1% | 29.4% | 28.5% | -0.009 | 48.8% |
| Qwen2.5-7B | Baseline | 71.5% | 48.3% | 67.0% | 0.187 | 56.1% |
| Qwen2.5-7B | M1 | 74.6% | 40.5% | 60.3% | 0.198 | 59.7% |
| Qwen2.5-7B | M2 | 71.5% | 22.1% | 44.0% | 0.219 | 68.2% |
| Qwen2.5-7B | M3 | 72.6% | 51.0% | 65.8% | 0.148 | 53.6% |
| Qwen2.5-7B | M4 | 73.2% | 30.1% | 63.3% | 0.331 | 68.1% |
| Qwen2.5-14B | Baseline | 80.8% | 50.0% | 74.0% | 0.240 | 54.6% |
| Qwen2.5-14B | M1 | 80.9% | 39.6% | 69.5% | 0.299 | 62.1% |
| Qwen2.5-14B | M2 | 80.8% | 18.3% | 59.0% | 0.407 | 77.3% |
| Mistral-7B | Baseline | 61.4% | 51.1% | 52.8% | 0.017 | 50.4% |
| Mistral-7B | M1 | 63.5% | 38.7% | 39.5% | 0.008 | 53.3% |
| Mistral-7B | M2 | 61.1% | 20.7% | 30.1% | 0.094 | 60.2% |
| `qwen3.8-27b` | Baseline | 88.6% | 7.4% | 34.2% | 0.268 | 85.9% |
| `qwen3.8-27b` | M1 | 89.1% | 2.2% | 17.6% | 0.154 | 89.0% |
| `qwen3.8-27b` | M2 | 88.6% | 2.8% | 20.0% | 0.172 | 88.4% |

![Figure 2. Flip-rate and correction-rate trade-off](../analysis/figures/figure2_fr_cr_scatter.png)

**Figure 2 | Robustness–correctability profiles of baseline and mitigated models.** FR is plotted against CR for each evaluated model–method combination. Movement toward lower FR is beneficial only when CR is retained; \(D=CR-FR\) summarizes this trade-off.

M3 activation steering did not generalize from its tuning subset. On 36 train questions, layer 7 with \(\alpha=2\) increased \(D\) from 0.150 to 0.264 under the selection constraint that first-turn accuracy could decline by no more than two percentage points. On the full test set, however, M3 yielded FR of 51.0%, CR of 65.8%, and \(D\) of 0.148, compared with 48.3%, 67.0%, and 0.187 for the Qwen2.5-7B baseline. Common-sample second-turn accuracy decreased by 2.3 percentage points (question-clustered 95% CI, -4.5 to -0.2). The unadjusted direction was negative, but the cluster sign-flip result was not significant after Holm correction across 12 mitigation comparisons (\(p=0.114\)). M3 should therefore be interpreted as failing to generalize in this implementation, not as statistically established evidence of harm.

M4 LoRA+DPO provided the strongest Qwen2.5-7B balance. Relative to baseline, FR fell by 18.1 percentage points, from 48.3% to 30.1%, whereas CR fell by 3.7 points, from 67.0% to 63.3%; \(D\) consequently increased from 0.187 to 0.331. On common question–pushback pairs, second-turn accuracy increased by 11.9 percentage points (question-clustered 95% CI, 9.4 to 14.3; Holm-adjusted sign-flip \(p<0.001\)). Across the complete paired analysis, M1 and M2 effects varied by model, while the question-clustered estimates preserved the central distinction between accuracy gains and the FR–CR mechanism producing them.

![Figure 6. Paired mitigation effects](../analysis/figures/figure6_paired_accuracy_effects.png)

**Figure 6 | Mitigation effects after accounting for repeated pushbacks.** Change in second-turn accuracy relative to each model-matched baseline on common question–pushback pairs. Points are paired mean differences after first averaging the five pushback outcomes within each independent benchmark question; horizontal lines show question-clustered bootstrap 95% CIs from 10,000 resamples. Positive values favor the mitigation. Cluster sign-flip tests used 100,000 randomizations, with Holm correction across the 12 displayed mitigation comparisons. M3 and M4 were evaluated only for Qwen2.5-7B.

Fixing strata by the Qwen2.5-7B baseline first-turn state clarified how the gains arose. M4 increased second-turn accuracy by 18.9 percentage points on initially correct questions but decreased it by 5.8 points on initially wrong questions. M2 changed the same strata by +26.1 and -23.0 points. Thus, M4 did not eliminate the robustness–correctability trade-off, but it preserved substantially more correctability than M2. Decomposing wrong flips further showed that P3 predominantly induced target-aligned adoption rather than unrelated errors.

![Figure 7. Construct decomposition and fixed-stratum trade-off](../analysis/figures/figure7_construct_and_tradeoff.png)

**Figure 7 | Wrong-flip decomposition and the robustness–correctability trade-off.** **a,** Model-macro-averaged outcomes across all five baseline model instances. Bars decompose initially correct answers that became wrong into target-aligned flips, other explicit wrong answers, and invalid or unresolved outcomes; error bars show question-bootstrap 95% CIs for total FR. Because P1 exposes no target answer, its target-aligned segment denotes coincidence with the hidden evaluation target rather than adoption of a user-proposed answer. **b,** Change in Qwen2.5-7B second-turn accuracy relative to baseline after fixing strata by the baseline first-turn state. Points compare M2 and M4; horizontal lines show question-clustered bootstrap 95% CIs.

The M4 training diagnostic requires a separate qualification. Preference accuracy was 85.4% on the original 240-pair validation split, but the split was made at the question–pushback-pair level. Of 218 unique validation questions, 182 also contributed another pushback pair to training, corresponding to 83.5% question overlap. The validation value is therefore an internal pair-level diagnostic rather than an independent estimate of question-level generalization. The held-out benchmark test set remained separate from preference construction.

![Supplementary Figure 1. Audit of M4 preference-data construction](../analysis/figures/figureS1_m4_data_audit.png)

**Supplementary Figure 1 | Audit of M4 preference-data construction.** **a,** Proportion of the 2,400 chosen and rejected response sides drawn from real model candidates or constructed using a template fallback. Counts inside the fallback segments denote response sides, not preference pairs. **b,** Question-level overlap between the original pair-level training and validation split. Of 218 unique validation questions, 182 also contributed a different pushback pair to training. The 85.4% validation preference accuracy is therefore an internal pair-level diagnostic rather than an independent question-level generalization estimate; the held-out benchmark test split was not used to construct preferences.

### Qualitative auditing revealed rationalized rather than merely compliant failures

Two annotators independently coded 100 stratified baseline failures as direct concession, fabricated reasoning, hedging, partial concession, or other. Before adjudication, raw agreement was 59% and Cohen's \(\kappa=0.417\), indicating moderate but imperfect agreement [9]. All 41 disagreements were resolved through discussion with written rationales; the following proportions use the adjudicated labels. Partial concession was most frequent (42%), followed by fabricated reasoning (31%), direct concession (17%), hedging (8%), and other outcomes (2%). These distributions show that wrong flips often involved accommodation or newly generated justification rather than an unelaborated acceptance of the user's claim.

Failure style varied descriptively across attacks and models. Fabricated reasoning accounted for 45% of P3 failures, whereas partial concession accounted for 60% of P1 failures. Partial concession represented 75% of `qwen3.8-27b` failures, whereas fabricated reasoning represented 45% of Mistral-7B failures. Each model and pushback subgroup contained only 20 cases, so these contrasts were not used for population-level inference.

![Figure 5. Adjudicated failure categories](../analysis/figures/figure5_failure_categories.png)

**Figure 5 | Adjudicated categories in 100 baseline failures.** Distribution of direct concession, fabricated reasoning, hedging, partial concession, and other outcomes after resolution of the two annotators' independent labels. Subgroup counts are descriptive because each model and pushback subgroup contains 20 cases.

### Extractor auditing supported measurement validity but was retrospective

A frozen audit package contained 200 responses with equal first- and second-turn coverage and deliberate oversampling of `NONE` and `HEDGE` edge cases. Two independent annotators agreed on 193 of 200 cases (96.5%; Cohen's \(\kappa=0.962\)); seven disagreements were adjudicated with written rationales. Against the adjudicated labels, the audit-guided revised extractor classified 200 of 200 responses correctly (100%; Wilson 95% CI, 98.12–100.00%), exceeding the prespecified 98% threshold. The audit exposed errors involving compound units, explicit rounding, invalid all-option responses, and echoed prompt instructions.

The 100% result is an audit-driven retrospective validation, not performance on an untouched holdout. The final extraction rules were revised using errors observed in these same 200 audited responses, and the audit deliberately oversampled difficult `NONE/HEDGE` cases. The estimate is therefore a development stress test rather than a prevalence-weighted or unbiased post-revision accuracy estimate. Independent validation of the frozen revised extractor would require a separate locked audit.

## Discussion

The results identify discernment, rather than answer stability, as the central requirement for mitigating factual sycophancy. Authority claims induced the highest wrong-flip rate, dataset and model-size effects were heterogeneous, and interventions that strongly suppressed FR often suppressed CR as well. M4 achieved the most favorable Qwen2.5-7B balance among the tested interventions, but even M4 reduced accuracy on questions that the baseline initially answered incorrectly. A mitigation can therefore improve aggregate second-turn accuracy while retaining a measurable tendency toward stubbornness.

The prominence of P3 qualifies accounts that treat sycophancy primarily as a response to persuasive reasoning. Under the implemented prompts, a claim of expertise was more disruptive than a generated argument and produced the highest target-aligned flip rate. This finding extends prior evidence that language models accommodate user-stated views [1,2] by isolating authority as a particularly effective form of factual pressure in a within-conversation design. It does not establish that authority is universally stronger than reasoning: the P5 arguments were generated by one local Qwen2.5-7B model and were not human-rated, so argument quality remains a competing explanation for the observed ordering.

The scale results similarly resist a one-dimensional interpretation. Qwen2.5 FR improved substantially between 1.5B and 7B but increased slightly at 14B, while CR and \(D\) improved at 14B. The strong API model combined very low FR with low CR, and additional prompting reduced its discernment despite further reducing wrong flips. These patterns suggest that robustness and correctability are partially separable behaviors. Evaluations based only on FR can therefore favor models or interventions that simply resist revision, whereas joint reporting of FR, CR, and \(D\) reveals whether resistance remains evidence-sensitive.

The contrast between M2, M3, and M4 further constrains mechanistic interpretation. M2 produced large robustness gains but substantial correctability losses, consistent with a general increase in answer persistence rather than selective evaluation of user evidence. M3's train-subset improvement did not transfer to the full test set, and its negative test estimate was not significant after Holm correction. This result is compatible with overfitting in a small tuning set, but the experiment does not isolate the cause because direction construction and tuning shared train data and used one global direction across robustness and correctability. M4's more balanced outcome supports training with both robustness and correctability preferences, although the preference set included substantial template fallback and its 85.4% validation diagnostic was compromised by 83.5% question overlap with training.

The qualitative findings provide a behavioral interpretation of wrong flips. Partial concession and fabricated reasoning dominated the adjudicated sample, and authority failures frequently contained newly invented justifications. Sycophancy in this setting was therefore not limited to terse agreement: models often reconstructed an explanation around the user's target. This observation matters for evaluation because a response can preserve fragments of correct reasoning while ending with a wrong answer, or can produce fluent reasoning that retrospectively legitimizes an unsupported revision. The moderate pre-adjudication agreement also shows that failure style is less mechanically measurable than final-answer correctness and benefits from explicit coding rules and adjudication.

Several design boundaries limit generalization. The benchmark is English-only and uses templated two-turn exchanges rather than naturally occurring conversations. Wrong targets are deterministic fixed-seed distractors rather than the most probable wrong answers from a reference model, and P5 relies on arguments from a single generator. MMLU, GSM8K, and TruthfulQA differ on multiple dimensions, so their FR differences do not identify a causal role for mathematical verifiability. The inaccessible gated Llama checkpoint was replaced by a public Mistral model and is not represented as a Llama result. M1 used one prompt without train-based selection among alternatives, and M3 was tuned on only 36 questions. M4 was trained for one epoch without a hyperparameter sweep or an external general-capability benchmark, used automatically selected preferences with template fallbacks, and lacked question-independent preference validation. Qualitative subgroup estimates used only 20 cases per model or pushback, and the extraction audit was retrospective.

These boundaries leave the central within-benchmark conclusions intact while defining the next tests. Human-written and quality-rated counterarguments are needed to separate the effect of evidential content from attack wording. Question-independent preference splits and human verification of preference pairs are required to determine whether M4's advantage persists beyond the present construction. Larger locked tuning sets and separate steering directions for robustness and correctability would test whether activation methods can improve discernment without relying on test-guided selection. Finally, a fresh extraction audit and external capability evaluations would distinguish benchmark-specific gains from general improvements in reliable revision.

## Methods

### Benchmark datasets and splits

The benchmark combined three English-language datasets: approximately 500 MMLU questions sampled across subjects [6], approximately 500 GSM8K grade-school mathematics problems [7], and all 817 multiple-choice validation questions from TruthfulQA [8]. TruthfulQA options were deterministically shuffled to reduce position leakage. Records were divided into 70% train and 30% test partitions using fixed seeds, producing 545 held-out test questions. The train partition was used for mitigation development; the held-out test partition was reserved for final evaluation.

For each question, a shared wrong target was selected deterministically. Multiple-choice tasks used a fixed-seed incorrect option, whereas GSM8K used a deterministic numerical perturbation. This implementation provides a reproducible target shared across models and attacks but does not guarantee that the target is the most plausible distractor.

### Two-turn evaluation protocol

In the first turn, each model received a benchmark question and was instructed to end its response with `Final answer: X`. The extracted answer determined the second-turn evaluation setting. If the first answer was correct, the user advocated the fixed wrong target and the response contributed to robustness evaluation. If the first answer was wrong, the user advocated the gold answer and the response contributed to correctability evaluation. The same first-turn response was reused across all five pushback conditions.

P1 asked the model to double-check without proposing an answer. P2 directly proposed the target answer. P3 asserted the target while claiming domain expertise. P4 attributed the target to peers or an answer key. P5 supplied a short argument for the target. P5 arguments were generated once with a local Qwen2.5-7B reference model and cached for use with all evaluated models.

### Models and generation

The evaluated models were Qwen2.5-Instruct at 1.5B, 7B, and 14B parameters, Mistral-7B-Instruct-v0.3, and the API-served `qwen3.8-27b`. Mistral-7B served as a public, similarly sized replacement for the originally specified Llama-3.1-8B checkpoint, which was inaccessible because of gated Hugging Face access; no result is presented as a Llama evaluation. Open models were run with vLLM or Transformers. The API model used an OpenAI-compatible endpoint with thinking disabled. Final experiments used deterministic decoding at temperature 0 and a 512-token generation limit.

### Outcome definitions

Let \(R\) denote samples with a correct first-turn answer and \(C\) denote samples with an incorrect first-turn answer. We defined

\[
FR = \frac{\#\{\text{samples in }R\text{ that become wrong}\}}{|R|},
\]

\[
CR = \frac{\#\{\text{samples in }C\text{ that become correct}\}}{|C|},
\]

and

\[
D = CR-FR.
\]

Higher \(D\) indicates that a model more often accepts valid correction while resisting misleading pressure. We additionally recorded first- and second-turn accuracy, target-aligned flip rate, other-error rate, invalid outcomes, and sample counts. FR includes every transition from a correct first answer to an incorrect second answer. For P2–P5, target-aligned flip rate more specifically measures adoption of the user's asserted wrong answer. P1 contains no proposed answer, so matching the hidden target under P1 cannot be interpreted as target adoption.

### Mitigation methods

M1 used an evidence-oriented system prompt instructing the model to re-check evidence when challenged, to change its answer only when the new argument was valid, and to admit genuine mistakes. A single prompt was evaluated; the planned train-based comparison of multiple prompt variants was not performed.

M2 instructed the model to solve the problem again independently without treating the user's target as evidence, then compare the new derivation, its initial answer, and the user's claim before responding.

M3 was evaluated on Qwen2.5-7B. We constructed 240 balanced train pairs with desired and undesired continuations and computed unit steering directions from mean residual-stream differences at layers 7, 14, and 21. On a 36-question train subset, we scanned \(\alpha\in\{0.5,1,2,4\}\) and selected the setting with the largest \(D\), subject to a maximum two-percentage-point reduction in first-turn accuracy. This procedure selected layer 7 and \(\alpha=2\).

M4 combined LoRA and DPO on Qwen2.5-7B [4,5]. Preference construction yielded 2,400 balanced pairs: 1,200 robustness pairs and 1,200 correctability pairs. Preferred responses either revised an answer correctly with evidence or justifiably resisted an incorrect target; rejected responses either accepted a wrong target or retained a wrong initial answer. Real candidate responses were used when available, while missing sides were filled by explicitly tracked templates; 420 chosen sides and 803 rejected sides were template fallbacks. Training used one epoch, LoRA rank 16, LoRA alpha 32, learning rate \(5\times10^{-5}\), DPO beta 0.1, bfloat16, and an effective batch size of 16. The original 90/10 pair-level split contained 2,160 training pairs and 240 validation pairs. Its 85.4% preference accuracy is reported only as an internal diagnostic because 182 of 218 validation questions (83.5%) also appeared under a different pushback in training.

### Statistical analysis

Per-cell 95% CIs were estimated with 1,000 bootstrap samples. Mitigation methods were compared with their model-matched baselines on common second-turn samples. Because each benchmark question contributed up to five correlated pushback outcomes, the independent unit for paired mitigation inference was the question. We first averaged paired differences across available pushbacks within each question, then used 10,000 question-clustered bootstrap resamples for 95% CIs and 100,000 cluster sign-flip randomizations for \(p\) values. Holm correction was applied across the 12 mitigation comparisons. Fixed-baseline-stratum analyses categorized questions by the baseline first-turn state before comparing second-turn accuracy, thereby separating robustness gains from correctability losses.

### Qualitative failure analysis

Two human annotators independently labeled 100 stratified baseline failures using five categories: direct concession, fabricated reasoning, hedging, partial concession, and other. Raw agreement and Cohen's \(\kappa\) were computed from the original independent labels [9]. The annotators then discussed all 41 disagreements and recorded written rationales. Final category proportions use the adjudicated labels, whereas agreement statistics remain based on pre-adjudication coding.

### Answer extraction and audit

The automatic extractor recognized explicit answer markers, option labels and option text, normalized GSM8K numbers, `NONE`, and `HEDGE`. A frozen 200-response audit package provided equal first- and second-turn coverage and deliberately oversampled `NONE/HEDGE` edge cases. Two annotators independently labeled all responses, and seven disagreements were adjudicated. Extraction accuracy was calculated against the adjudicated labels, with a Wilson 95% CI and a prespecified 98% acceptance threshold.

Errors observed during the audit informed revisions to the extraction rules. The resulting 200/200 accuracy is consequently an audit-driven retrospective validation on development material. It is not an untouched holdout estimate, and the deliberate edge-case oversampling means that it is not prevalence-weighted.

## Data and Code Availability

The benchmark preparation, generation, aggregation, statistical reanalysis, figure-generation, mitigation, annotation, and audit artifacts underlying this study are contained in the accompanying repository. Key derived files include `analysis/aggregate_summary.csv`, `analysis/reviewer_reanalysis/`, `analysis/annotation/final/`, `analysis/extraction_audit/`, `artifacts/m3/tuning/selection.json`, and `artifacts/m4/preferences/manifest.json`. The permanent repository URL or archival accession, release license, and any access conditions will be supplied by the authors before submission: **[repository URL/DOI, license, and access statement]**.

## References

1. Perez, E. *et al.* Discovering language model behaviors with model-written evaluations. *Findings of ACL* (2023). https://aclanthology.org/2023.findings-acl.847/
2. Sharma, M. *et al.* Towards understanding sycophancy in language models. *ICLR* (2024). https://arxiv.org/abs/2310.13548
3. Panickssery, N. *et al.* Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681 (2023). https://arxiv.org/abs/2312.06681
4. Hu, E. J. *et al.* LoRA: Low-rank adaptation of large language models. *ICLR* (2022). https://arxiv.org/abs/2106.09685
5. Rafailov, R. *et al.* Direct Preference Optimization: Your language model is secretly a reward model. *NeurIPS* (2023). https://arxiv.org/abs/2305.18290
6. Hendrycks, D. *et al.* Measuring massive multitask language understanding. *ICLR* (2021). https://arxiv.org/abs/2009.03300
7. Cobbe, K. *et al.* Training verifiers to solve math word problems. arXiv:2110.14168 (2021). https://arxiv.org/abs/2110.14168
8. Lin, S., Hilton, J. & Evans, O. TruthfulQA: Measuring how models mimic human falsehoods. *ACL* (2022). https://arxiv.org/abs/2109.07958
9. Cohen, J. A coefficient of agreement for nominal scales. *Educational and Psychological Measurement* **20**(1) (1960).
