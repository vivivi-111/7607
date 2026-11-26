# Selective Updating under User Pushback

Code and analysis for evaluating whether language models selectively accept
correct feedback while resisting incorrect feedback.

The repository contains two complementary protocols:

- an adaptive two-turn stress test that exposes operational failures but
  couples feedback truth to first-turn correctness;
- a fully crossed `2 × 2 × 2 × 2` experiment that independently varies the
  supplied initial answer, feedback truth, authority, and rationale.

It also includes four mitigation families: prompting (M1), independent
self-review (M2), activation steering (M3), and LoRA+DPO preference
optimization (M4).

## Scientific status

The crossed design fixes the logical confound in the adaptive protocol.
However, the frozen answer extractor matched only 179/200 adjudicated responses
in a checksum-locked quota audit (89.5%, Wilson 95% CI 84.5–93.0%), below the
prespecified 98% threshold. Numerical FR, CR, adoption, interaction, and
mitigation estimates should therefore be treated as provisional until a
revised extractor passes a new untouched audit. See the manuscript and
`docs/REPRODUCIBILITY.md` for the complete boundary.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,annotation]"

# Select only what you need:
pip install -e ".[api]"
pip install -e ".[vllm]"
pip install -e ".[train]"
```

## Quick start

Prepare the benchmark:

```bash
sycophancy prepare --output data/processed/benchmark.jsonl
```

Run the adaptive protocol:

```bash
sycophancy run \
  --backend vllm \
  --model /path/to/model \
  --input data/processed/benchmark.jsonl \
  --justification-cache data/processed/p5_justifications_test.json \
  --output-dir results/example
```

Run the controlled factorial protocol:

```bash
PYTHONPATH=src python scripts/run_factorial_feedback.py \
  --backend vllm \
  --model /path/to/model \
  --model-id example-model \
  --input data/processed/benchmark.jsonl \
  --split test \
  --justification-cache data/processed/p5_justifications_test.json \
  --output-dir results/factorial_example
```

Clustered analyses and the three-seed M4 workflow are documented in
`docs/REPRODUCIBILITY.md`. The shell launchers take paths through environment
variables and do not require the original compute environment.

## Tests

Tests use in-memory fixtures and do not download models or datasets:

```bash
PYTHONPATH=src:. pytest -q
```

## Repository layout

- `src/sycophancy/`: benchmark preparation, model backends, extraction, metrics
- `scripts/`: experiment, mitigation, training, and presentation entry points
- `analysis/`: statistical analyses and frozen aggregate outputs
- `paper/neurips2026/`: manuscript source and reproducible figure scripts
- `prompts/`: pushback templates
- `tests/`: unit tests
- `docs/`: reproducibility and annotation documentation

The final manuscript, slides, speaker script, and QA records are distributed
as a separate release bundle rather than duplicated in the GitHub code
snapshot.

Raw generations, model weights, local caches, and API credentials are excluded
from version control. Aggregate analysis tables and figure source data are
included.

## Reproducibility and licenses

See:

- `docs/REPRODUCIBILITY.md`
- `LICENSE`
- `ASSET_LICENSES.md`

The software is released under the MIT License. Source datasets and model
weights remain subject to their respective licenses.
