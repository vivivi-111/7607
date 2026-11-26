#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON="${PYTHON:-$(command -v python)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the Qwen2.5-7B-Instruct checkpoint}"
LOCAL_ROOT="${LOCAL_ROOT:-${TMPDIR:-/tmp}/sycophancy_m4_revalidation}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
SEEDS="${SEEDS:-42 43 44}"
EXPECTED_FACTORIAL_ROWS="${EXPECTED_FACTORIAL_ROWS:-8608}"
PERSIST_MERGED="${PERSIST_MERGED:-0}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-1024}"
DPO_MAX_PROMPT_LENGTH="${DPO_MAX_PROMPT_LENGTH:-800}"

cd "${ROOT}"
mkdir -p "${LOCAL_ROOT}"
BASE_LOCAL="${LOCAL_ROOT}/Qwen2.5-7B-Instruct"
if [[ ! -f "${BASE_LOCAL}/config.json" ]]; then
  rm -rf "${BASE_LOCAL}.staging"
  mkdir -p "${BASE_LOCAL}.staging"
  tar -C "${BASE_MODEL}" --exclude=".cache" --exclude=".git" \
    -cf - . | tar -C "${BASE_LOCAL}.staging" -xf -
  mv "${BASE_LOCAL}.staging" "${BASE_LOCAL}"
fi

base_capability="results/capability_qwen2.5-7b-base.jsonl"
base_capability_rows=0
if [[ -f "${base_capability}" ]]; then
  base_capability_rows="$(wc -l < "${base_capability}")"
fi
if (( base_capability_rows < 1000 )); then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "${PYTHON}" scripts/run_capability_eval.py \
    --input data/processed/capability_holdout.jsonl \
    --model "${BASE_LOCAL}" \
    --model-id "Qwen2.5-7B-Instruct" \
    --output "${base_capability}" \
    --max-tokens 1024 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
fi

for seed in ${SEEDS}; do
  run_name="grouped-real-seed${seed}"
  adapter_dir="artifacts/m4/dpo_${run_name}"
  merged_local="${LOCAL_ROOT}/Qwen2.5-7B-DPO-M4-${run_name}"

  if [[ ! -f "${adapter_dir}/training_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
      "${PYTHON}" scripts/train_dpo.py \
      --model "${BASE_LOCAL}" \
      --train artifacts/m4/preferences_grouped_real_only/train.jsonl \
      --validation artifacts/m4/preferences_grouped_real_only/validation.jsonl \
      --output-dir "${adapter_dir}" \
      --epochs 1 \
      --learning-rate 5e-5 \
      --beta 0.1 \
      --batch-size 1 \
      --gradient-accumulation 16 \
      --max-length "${DPO_MAX_LENGTH}" \
      --max-prompt-length "${DPO_MAX_PROMPT_LENGTH}" \
      --seed "${seed}"
  fi

  if [[ ! -f "${merged_local}/config.json" ]]; then
    rm -rf "${merged_local}.staging"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
      "${PYTHON}" scripts/merge_lora.py \
      --model "${BASE_LOCAL}" \
      --adapter "${adapter_dir}/best_adapter" \
      --output "${merged_local}.staging"
    mv "${merged_local}.staging" "${merged_local}"
  fi
  # Adapters and training summaries are already persistent. Saving each
  # deterministic merged checkpoint is optional and costs roughly 15 GB/seed.
  if [[ "${PERSIST_MERGED}" == "1" && -z "${MERGED_MODEL_DIR}" ]]; then
    echo "PERSIST_MERGED=1 requires MERGED_MODEL_DIR" >&2
    exit 2
  fi
  merged_persistent="${MERGED_MODEL_DIR}/Qwen2.5-7B-DPO-M4-${run_name}"
  if [[ "${PERSIST_MERGED}" == "1" && ! -f "${merged_persistent}/config.json" ]]; then
    rm -rf "${merged_persistent}.staging"
    mkdir -p "${MERGED_MODEL_DIR}"
    cp -a "${merged_local}" "${merged_persistent}.staging"
    mv "${merged_persistent}.staging" "${merged_persistent}"
  fi

  capability_output="results/capability_qwen2.5-7b-dpo-m4-${run_name}.jsonl"
  capability_rows=0
  if [[ -f "${capability_output}" ]]; then
    capability_rows="$(wc -l < "${capability_output}")"
  fi
  if (( capability_rows < 1000 )); then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${PYTHON}" scripts/run_capability_eval.py \
      --input data/processed/capability_holdout.jsonl \
      --model "${merged_local}" \
      --model-id "Qwen2.5-7B-DPO-M4-${run_name}" \
      --output "${capability_output}" \
      --max-tokens 1024 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  fi

  benchmark_output="results/qwen2.5-7b-dpo-m4-${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE}" \
    "${PYTHON}" -m sycophancy.cli run \
    --backend vllm \
    --model "${merged_local}" \
    --model-id "Qwen2.5-7B-DPO-M4-${run_name}" \
    --input data/processed/benchmark.jsonl \
    --split test \
    --mitigation M4 \
    --justification-cache data/processed/p5_justifications_test.json \
    --output-dir "${benchmark_output}" \
    --max-tokens 512 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

  factorial_output="results/factorial_qwen2.5-7b-dpo-m4-${run_name}"
  completed_rows=0
  if [[ -f "${factorial_output}/factorial_dialogues.jsonl" ]]; then
    completed_rows="$(wc -l < "${factorial_output}/factorial_dialogues.jsonl")"
  fi
  if (( completed_rows < EXPECTED_FACTORIAL_ROWS )); then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${PYTHON}" scripts/run_factorial_feedback.py \
      --backend vllm \
      --model "${merged_local}" \
      --model-id "Qwen2.5-7B-DPO-M4-${run_name}" \
      --input data/processed/benchmark.jsonl \
      --split test \
      --mitigation M4 \
      --justification-cache data/processed/p5_justifications_test.json \
      --output-dir "${factorial_output}" \
      --max-tokens 1024 \
      --generation-batch-size "${GENERATION_BATCH_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  fi
done
