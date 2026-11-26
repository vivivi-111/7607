# Terminology ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| sycophancy | agreement with unsupported user claims at the expense of factual accuracy | Use lowercase except at sentence start |
| user pushback | a second-turn message challenging the model's first answer | Use `pushback`, not `attack`, unless discussing adversarial intent |
| wrong-answer flip rate (FR) | proportion of initially correct answers that become wrong | Lower is better |
| correction rate (CR) | proportion of initially wrong answers that become correct | Higher is better |
| discernment (D) | `D = CR − FR` | Higher is better |
| target-aligned flip | second-turn answer matches the target asserted by the user | Use for P2–P5; qualify P1 because no target is exposed |
| P1 simple challenge | asks the model to double-check without proposing an answer | `P1` after first definition |
| P2 alternative answer | directly proposes the target answer | `P2` after first definition |
| P3 authority | asserts expertise and proposes the target answer | `P3` after first definition |
| P4 social pressure | attributes the target answer to peers or an answer key | `P4` after first definition |
| P5 reasoned persuasion | supplies an argument for the target answer | `P5` after first definition |
| M1 evidence-oriented system prompt | prompts evidence re-evaluation while permitting valid correction | `M1` after first definition |
| M2 independent self-review | independently resolves the question before comparing claims | `M2` after first definition |
| M3 activation steering | contrastive residual-stream direction added at inference | `M3` after first definition |
| M4 LoRA+DPO | low-rank adaptation with direct preference optimization | `M4` after first definition |
| Qwen2.5-1.5B / Qwen2.5-7B / Qwen2.5-14B | local open-weight instruction models | Do not shorten to ambiguous `Qwen-7B` in prose |
| Mistral-7B-Instruct-v0.3 | public replacement for inaccessible Llama-3.1-8B | Never describe as a Llama result |
| qwen3.8-27B | API-served model identifier used in the experiment | Preserve identifier exactly |
| question-clustered bootstrap | bootstrap with benchmark question as the independent unit | Use for paired mitigation effects |
| cluster sign-flip test | paired randomization test over question-level effects | Report Holm-adjusted p-values |
