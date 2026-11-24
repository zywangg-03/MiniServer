# Final GPU validation: one-session workflow

This guide assumes the existing remote project `/home/zywangg/miniserver/`, its Linux `.venv`, and the Qwen model cache from Phase 1. Your previous successful run used PyTorch 2.6.0+cu124 on a V100. Retain that installation while adding the HTTP dependencies.

## 1. Update source

From your Mac:

```bash
rsync -av --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='.bootstrap/' \
  --exclude='.python/' --exclude='.uv-cache/' --exclude='.cache/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.egg-info/' \
  --exclude='.env*' --exclude='.DS_Store' --exclude='results/' \
  /Users/zywangg/Desktop/miniserver/ \
  zywangg@greatlakes-xfer.arc-ts.umich.edu:/home/zywangg/miniserver/
```

The existing remote virtual environment, cached model, and experiment results are preserved. If the remote project is a Git checkout instead, pull the updated branch there.

## 2. Prepare the existing environment

In a server terminal:

```bash
cd /home/zywangg/miniserver
module load python3.11-anaconda
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -e . --no-deps
python -m pip check
export HF_HOME="$PWD/.cache/huggingface"
python -m pytest -q
```

If your Phase 1 cache lives elsewhere, set `HF_HOME` to that existing location instead. Avoid downloading a second copy when home storage is nearly full. Use `home-quota -u zywangg` and `du -xhd1 /home/zywangg | sort -h` to inspect usage. The final script runs offline and will fail clearly if the cache is missing.

Expected default result: **84 passed, 8 skipped**. The eight real-model acceptance cases include the new unequal-length batched decode test. That test verifies exact greedy IDs against individual HF generation, not just plausible text. With real-model tests enabled by the final script, a successful run reports **92 passed**.

## 3. Allocate the GPU

Skip this if already inside an allocated GPU session. Your account inspection showed that the `gpu` partition denies `lsa1` and `lsa3`; `lsa2` is associated with your user and was absent from that deny list. Request from a login terminal, subject to current account limits:

```bash
salloc --account=lsa2 --partition=gpu --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --gres=gpu:1 --time=01:00:00
```

After the compute-node prompt appears:

```bash
cd /home/zywangg/miniserver
module load python3.11-anaconda
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

The CPU-only Open OnDemand desktop on `standard` is not a GPU session. Do not override the allocation's `CUDA_VISIBLE_DEVICES`.

## 4. Run the short validation, then the full experiment

```bash
bash scripts/run_final.sh --quick
```

This runs all tests with the actual Qwen checkpoint, then a short engine experiment and a real HTTP/SSE load test. Stop and retain the report if any stage fails. In particular, GPU batched output mismatches must be resolved before using throughput numbers.

Once the short run succeeds:

```bash
bash scripts/run_final.sh
```

The full run uses 1/4/8 requests per workload, burst and 50-ms staggered arrivals, alternating 128/256-token prompts, up to 64 outputs, 8/48 MiB logical KV budgets, one warm-up and five measured repetitions. This replaces the larger original experiment matrix. It tests two questions: shared decode behavior, and the queueing effect of admission limits.

The script prints a unique `results/final-<jobid>-XXXXXX/` path containing:

| Artifact | What to inspect |
| --- | --- |
| `tests.xml` | No failures or skips for the real-Qwen acceptance run |
| `serving_eval.json` | `passed: true`; all request outputs match HF; real batch sizes >1 in concurrent runs |
| `comparison.json`, `comparison.csv` | Paired throughput ratios, variation across repetitions, latency and peak allocated VRAM |
| `http_load.json` | `passed: true`; zero failed requests; client TTFT/TPOT/latency |
| `environment.txt` | Actual installed versions |
| `server.log` | Server startup, request errors, shutdown |

The script starts and stops its own loopback server. It needs no public endpoint or SSH tunnel. Budget effects are demonstrated in `serving_eval.json`; the HTTP smoke test uses the default 48 MiB budget.

If you prefer a batch job, submit from the project root:

```bash
export HF_HOME="$PWD/.cache/huggingface"
sbatch --account=lsa2 --partition=gpu scripts/slurm_final.sbatch --quick
# After inspecting the quick run:
sbatch --account=lsa2 --partition=gpu scripts/slurm_final.sbatch
```

This template loads `python3.11-anaconda`. If your remote `.venv` was created from a different module, change that line to match. For a different environment, export `MINISERVE_PYTHON=/absolute/path/to/python` before launching either script.

## 5. Run the API manually if desired

Inside the GPU allocation:

```bash
python -m miniserve --local-files-only --port 8000
```

In another terminal on that same compute node:

```bash
curl -N http://127.0.0.1:8000/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请解释 continuous batching。","max_new_tokens":32,"temperature":0}'
python -m benchmarks.load_test --url http://127.0.0.1:8000 \
  --concurrency 8 --repetitions 5 --output results/http_manual.json
```

The service supports raw prompts with `"use_chat_template": false`. Only greedy decoding is implemented. `done.text` is the complete authoritative response; individual text deltas can be empty for incomplete UTF-8 tokens.

## 6. Retrieve the artifacts

From your Mac:

```bash
rsync -av zywangg@greatlakes-xfer.arc-ts.umich.edu:/home/zywangg/miniserver/results/ \
  /Users/zywangg/Desktop/miniserver/results/remote/
```

Retain both favorable and unfavorable measurements. The padded-cache implementation may be slower than sequential generation; explaining where copies, padding, prefill interference, or queueing dominate is a valid systems result. Only the Phase 1 V100 baselines have been measured on the GPU so far.
