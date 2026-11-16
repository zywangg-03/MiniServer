#!/usr/bin/env bash
# 本地版本：在已有 GPU 环境执行 Phase 1。也由集群入口复用，以保持实验条件一致。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
MINISERVE_PYTHON="${MINISERVE_PYTHON:-$VENV_DIR/bin/python}"

if [[ ! -x "$MINISERVE_PYTHON" ]]; then
    printf 'Python not found: %s\nSet VENV_DIR or MINISERVE_PYTHON to the prepared environment.\n' "$MINISERVE_PYTHON" >&2
    exit 1
fi

cd -- "$PROJECT_ROOT"
# cuda:0 表示进程可见的第一张卡；不要覆盖调度系统设置的 CUDA_VISIBLE_DEVICES。
COMMON_ARGS=(--device "${MINISERVE_DEVICE:-cuda:0}" --dtype "${MINISERVE_DTYPE:-float16}")
COMMON_ARGS+=("$@")
MEASUREMENT_ARGS=(
    --warmup "${MINISERVE_WARMUP:-1}"
    --repetitions "${MINISERVE_REPETITIONS:-5}"
    --max-new-tokens "${MINISERVE_MAX_NEW_TOKENS:-64}"
)

# mktemp 为重复运行生成独立目录，避免覆盖已有原始结果。
RESULTS_ROOT="${MINISERVE_RESULTS_DIR:-$PROJECT_ROOT/results}"
mkdir -p -- "$RESULTS_ROOT"
RUN_DIR="$(mktemp -d "$RESULTS_ROOT/phase1-${SLURM_JOB_ID:-${MINISERVE_RUN_LABEL:-local}}-XXXXXX")"
printf 'Phase 1 results: %s\n' "$RUN_DIR"

"$MINISERVE_PYTHON" -m benchmarks.verify "${COMMON_ARGS[@]}" --max-new-tokens "${MINISERVE_MAX_NEW_TOKENS:-64}" --output "$RUN_DIR/correctness.json"
"$MINISERVE_PYTHON" -m benchmarks.hf_baseline "${COMMON_ARGS[@]}" "${MEASUREMENT_ARGS[@]}" --output "$RUN_DIR/hf_baseline.json"
"$MINISERVE_PYTHON" -m benchmarks.manual_generate "${COMMON_ARGS[@]}" "${MEASUREMENT_ARGS[@]}" --output "$RUN_DIR/manual_generate.json"
printf 'Completed. Results: %s\n' "$RUN_DIR"
