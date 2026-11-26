# Reproducibility guide

This document separates what can be reproduced from the public repository from
what requires model weights or generated responses.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,annotation]"
```

Install one inference backend:

```bash
pip install -e ".[api]"
# or
pip install -e ".[vllm]"
```

M3/M4 training additionally requires:

```bash
pip install -e ".[train]"
```

All commands below are run from the repository root with `PYTHONPATH=src:.`.

## 2. Data preparation

```bash
sycophancy prepare --output data/processed/benchmark.jsonl
python scripts/build_capability_holdout.py \
  --benchmark data/processed/benchmark.jsonl \
  --output data/processed/capability_holdout.jsonl
```

The benchmark builder uses fixed seeds. MMLU is sampled by subject, TruthfulQA
options are deterministically shuffled, and records are assigned to fixed
train/test partitions.

P5 requires a justification cache:

```bash
python scripts/generate_justifications.py \
  --backend vllm \
  --model /path/to/reference-model \
  --input data/processed/benchmark.jsonl \
  --output data/processed/p5_justifications_test.json
```

## 3. Adaptive two-turn benchmark

```bash
sycophancy run \
  --backend vllm \
  --model /path/to/model \
  --input data/processed/benchmark.jsonl \
  --split test \
  --justification-cache data/processed/p5_justifications_test.json \
  --output-dir results/example

sycophancy metrics \
  --input results/example/dialogues.jsonl \
  --output analysis/example.csv \
  --bootstrap 1000
```

The first turn is shared across P1–P5. FR and CR have different first-turn
strata and are retained as operational stress-test quantities. `D = CR - FR`
is a balance summary, not an independent discernment measure.

## 4. Controlled factorial experiment

```bash
python scripts/run_factorial_feedback.py \
  --backend vllm \
  --model /path/to/model \
  --model-id example-model \
  --input data/processed/benchmark.jsonl \
  --split test \
  --justification-cache data/processed/p5_justifications_test.json \
  --output-dir results/factorial_example

python analysis/factorial_feedback_analysis.py \
  --input results/factorial_example/factorial_dialogues.jsonl \
  --output-dir analysis/factorial/example
```

Each eligible question enters all 16 cells. The primary contrast is
correct-minus-incorrect feedback adoption paired within question. Confidence
intervals use question-clustered bootstrap resampling; tests use question-level
sign flips with Holm correction.

Portable multi-model and prompt-mitigation launchers are available at:

- `scripts/run_factorial_baselines.sh`
- `scripts/run_factorial_prompt_mitigations.sh`

Set `MODEL_ROOT`, `MODEL`, `PYTHON`, `GPU_ID`, and memory settings through
environment variables as needed.

## 5. M4 three-seed revalidation

The end-to-end launcher is:

```bash
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct \
PYTHON="$(command -v python)" \
bash scripts/run_m4_multiseed_revalidation.sh
```

It trains seeds 42, 43, and 44, evaluates adaptive and factorial behavior, and
runs the 1,000-question capability holdout. Preference pairs are
question-disjoint and contain real model responses only.

The released aggregate tables are under `analysis/m4_revalidation/`. Model
weights and adapters are not included.

## 6. Extraction and human audits

Build a frozen extraction-audit packet:

```bash
python analysis/build_extraction_audit_packets.py \
  --aggregate analysis/aggregate_summary.csv \
  --output-dir analysis/extraction_audit_new
```

After two independent annotators return their workbooks:

```bash
python -m analysis.score_extraction_audit \
  annotator_A.xlsx annotator_B.xlsx \
  --mapping analysis/extraction_audit_new/private_mapping.csv \
  --output-dir analysis/extraction_audit_new/scores
```

If disagreements exist, fill the generated `adjudication.xlsx` and rerun with
`--adjudication`.

The current locked audit produced 193/200 exact human agreements
(`κ = 0.962`). The frozen extractor matched 179/200 adjudicated responses
(89.5%, Wilson 95% CI 84.5–93.0%), failing the prespecified 98% threshold.
Because the packet uses quotas on extractor-predicted status and samples the
observational/mitigation response pool, this is a diagnostic audit value rather
than a population-weighted accuracy or a direct audit of factorial outputs.

## 7. Manuscript

The LaTeX source, style file, and publication figures are maintained outside
this code repository. Analysis outputs used by the manuscript remain under
`analysis/`.

## 8. Reproducibility levels

- **Runs from scratch:** requires source datasets, model weights, GPU resources,
  and the P5 cache.
- **Re-analysis:** uses released aggregate CSV files and analysis scripts.
- **Manuscript rebuild:** requires the separately stored LaTeX source and figures.
- **Exact raw-output replay:** not provided in the GitHub package because raw
  generations and model artifacts are large and may carry provider-specific
  restrictions.

The final manuscript explicitly treats automated behavioral effect sizes as
provisional until a corrected extractor passes a new untouched audit.
