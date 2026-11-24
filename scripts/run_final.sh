#!/usr/bin/env bash
# Local/SSH/Slurm: run INSIDE an allocated GPU session. One process, one GPU.
# Existing HF_HOME is preserved; cache can live on your approved scratch path.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${MINISERVE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MINISERVE_TEST_DEVICE="${MINISERVE_DEVICE:-cuda:0}"
export MINISERVE_TEST_DTYPE="${MINISERVE_DTYPE:-float16}"
REPETITIONS=5
OUTPUT_TOKENS=64
CONCURRENCY=(1 4 8)
PATTERNS=(burst staggered)
if [[ "${1:-}" == '--quick' ]]; then
    REPETITIONS=1
    OUTPUT_TOKENS=16
    CONCURRENCY=(1 4)
    PATTERNS=(burst)
    shift
fi
if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/run_final.sh [--quick]\n' >&2
    exit 2
fi
MODEL_ARGS=()
if [[ -n "${MINISERVE_MODEL_PATH:-}" ]]; then MODEL_ARGS+=(--model "$MINISERVE_MODEL_PATH"); fi
if [[ -n "${MINISERVE_MODEL_REVISION:-}" ]]; then MODEL_ARGS+=(--revision "$MINISERVE_MODEL_REVISION"); fi
COMMON=(--device "$MINISERVE_TEST_DEVICE" --dtype "$MINISERVE_TEST_DTYPE" --local-files-only "${MODEL_ARGS[@]}")
mkdir -p results
RUN_DIR="$(mktemp -d "$PROJECT_ROOT/results/final-${SLURM_JOB_ID:-local}-XXXXXX")"
printf 'Final validation artifacts: %s\n' "$RUN_DIR"
"$PYTHON" -m pip freeze > "$RUN_DIR/environment.txt"
# Tests include actual pretrained variable-length batched decode; fail fast.
MINISERVE_RUN_MODEL_TESTS=1 "$PYTHON" -m pytest -q --junitxml="$RUN_DIR/tests.xml"
"$PYTHON" -m benchmarks.serving_eval "${COMMON[@]}" \
    --input-lengths 128 256 --max-new-tokens "$OUTPUT_TOKENS" \
    --concurrency "${CONCURRENCY[@]}" --patterns "${PATTERNS[@]}" \
    --kv-budgets-mb 8 48 --warmup 1 --repetitions "$REPETITIONS" \
    --output "$RUN_DIR/serving_eval.json"
"$PYTHON" -m benchmarks.compare "$RUN_DIR/serving_eval.json" --output "$RUN_DIR/comparison.json"
# Pick a free loopback port. Keep server reachable only within this compute node.
PORT="$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
"$PYTHON" -m miniserve "${COMMON[@]}" --port "$PORT" > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
"$PYTHON" - "$PORT" "$SERVER_PID" <<'PY'
import os, sys, time, urllib.request
port, pid = int(sys.argv[1]), int(sys.argv[2])
for _ in range(120):
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as r:
            if r.status == 200:
                break
    except OSError:
        pass
    time.sleep(1)
else:
    raise RuntimeError('Server failed readiness; inspect server.log')
PY
"$PYTHON" -m benchmarks.load_test --url "http://127.0.0.1:$PORT" \
    --concurrency 8 --max-new-tokens "$OUTPUT_TOKENS" --repetitions "$REPETITIONS" \
    --output "$RUN_DIR/http_load.json"
printf 'Finished. Review comparison.csv, http_load.json and tests.xml in %s\n' "$RUN_DIR"
