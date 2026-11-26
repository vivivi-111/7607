# Terminology ledger

- **selective updating**: accepting correct user feedback while resisting incorrect feedback.
- **feedback adoption rate**: proportion of responses whose final answer matches the user-proposed target.
- **truth discrimination in adoption**: paired difference between adoption of correct and incorrect feedback under the controlled factorial design.
- **authority effect on adoption**: paired adoption difference between authority-present and authority-absent conditions.
- **authority-by-truth interaction**: change in truth discrimination when authority is present.
- **observational two-turn benchmark**: the original adaptive protocol in which feedback truth depends on whether the model's natural first answer is correct.
- **controlled factorial benchmark**: the \(2\times2\times2\times2\) design crossing initial-answer correctness, feedback correctness, authority, and rationale.
- **wrong-answer flip rate (FR)**: rate at which naturally correct first answers become incorrect in the observational benchmark.
- **correction rate (CR)**: rate at which naturally incorrect first answers become correct in the observational benchmark.
- **balance summary (\(D=CR-FR\))**: a descriptive within-run summary; never call it an independent measure of discernment or use it alone for cross-model claims.
- **M1, evidence-oriented prompt**; **M2, independent self-review**; **M3, activation steering**; **M4, LoRA+DPO**.
- Canonical model names: **Qwen2.5-1.5B-Instruct**, **Qwen2.5-7B-Instruct**, **Qwen2.5-14B-Instruct**, **Mistral-7B-Instruct-v0.3**.
- The API-served **qwen3.8-27b** is exploratory and is excluded from the primary open-text-model macro-average.
- Claims are bounded to English, two-turn factual question answering and the tested prompt templates; neither authority effects nor mitigation effects are assumed universal across models.
