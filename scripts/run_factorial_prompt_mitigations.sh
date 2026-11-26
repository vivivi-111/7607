#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON="${PYTHON:-$(command -v python)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL="${MODEL:?Set MODEL to the Qwen2.5-7B-Instruct checkpoint}"
EXPECTED_FACTORIAL_ROWS="${EXPECTED_FACTORIAL_ROWS:-8608}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

cd "${ROOT}"
for mitigation in M1 M2; do
  output="results/factorial_qwen2.5-7b-${mitigation}"
  completed_rows=0
  if [[ -f "${output}/factorial_dialogues.jsonl" ]]; then
    completed_rows="$(wc -l < "${output}/factorial_dialogues.jsonl")"
  fi
  if (( completed_rows < EXPECTED_FACTORIAL_ROWS )); then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${PYTHON}" scripts/run_factorial_feedback.py \
      --backend vllm \
      --model "${MODEL}" \
      --model-id "Qwen2.5-7B-Instruct" \
      --input data/processed/benchmark.jsonl \
      --split test \
      --mitigation "${mitigation}" \
      --justification-cache data/processed/p5_justifications_test.json \
      --output-dir "${output}" \
      --max-tokens 1024 \
      --generation-batch-size "${GENERATION_BATCH_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  fi
done
