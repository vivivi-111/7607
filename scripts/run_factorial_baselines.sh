#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON="${PYTHON:-$(command -v python)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to the directory containing model checkpoints}"
LOCAL_ROOT="${LOCAL_ROOT:-${TMPDIR:-/tmp}/sycophancy_factorial_models}"
MODEL_NAMES="${MODEL_NAMES:-Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Mistral-7B-Instruct-v0.3}"
EXPECTED_FACTORIAL_ROWS="${EXPECTED_FACTORIAL_ROWS:-8608}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

cd "${ROOT}"
mkdir -p "${LOCAL_ROOT}"
for model_name in ${MODEL_NAMES}; do
  shared_model="${MODEL_ROOT}/${model_name}"
  local_model="${LOCAL_ROOT}/${model_name}"
  slug="$(printf '%s' "${model_name}" | tr '[:upper:]' '[:lower:]')"
  output="results/factorial_${slug}"

  if [[ ! -f "${local_model}/config.json" ]]; then
    rm -rf "${local_model}.staging"
    mkdir -p "${local_model}.staging"
    tar -C "${shared_model}" \
      --exclude=".cache" \
      --exclude=".git" \
      --exclude="consolidated.safetensors" \
      -cf - . | tar -C "${local_model}.staging" -xf -
    mv "${local_model}.staging" "${local_model}"
  fi
  completed_rows=0
  if [[ -f "${output}/factorial_dialogues.jsonl" ]]; then
    completed_rows="$(wc -l < "${output}/factorial_dialogues.jsonl")"
  fi
  if (( completed_rows < EXPECTED_FACTORIAL_ROWS )); then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${PYTHON}" scripts/run_factorial_feedback.py \
      --backend vllm \
      --model "${local_model}" \
      --model-id "${model_name}" \
      --input data/processed/benchmark.jsonl \
      --split test \
      --justification-cache data/processed/p5_justifications_test.json \
      --output-dir "${output}" \
      --max-tokens 1024 \
      --generation-batch-size "${GENERATION_BATCH_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  fi
done
