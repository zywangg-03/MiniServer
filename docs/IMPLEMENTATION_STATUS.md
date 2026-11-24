# Shortened scope and evidence boundaries

## Research question

How does iteration-level batching change throughput and latency, and how does a conservative logical KV reservation budget affect queueing? The core contribution is implementing and validating these mechanisms together, with reproducible measurements.

## Implemented

- A dedicated inference thread owns serving forwards and request execution state.
- FIFO waiting queue and running set; one prefill is admitted per iteration.
- Actual shared decode forwards accept `[batch_size, 1]` inputs. Membership changes at iteration boundaries.
- Full-attention KV caches are left-padded to the longest history. `position_ids` use true lengths while physical `cache_position` uses padded length. Attention masks hide padded history. Unpacking clones compact histories so slices do not retain the whole batch allocation.
- Per-request maximum-length logical KV reservations, immediate rejection if individually oversized, waiting when capacity is temporarily unavailable, and idempotent release on all terminal paths.
- Bounded waiting/output queues, bounded timing history and membership trace; cancellation and shutdown cleanup; sleeping idle worker.
- JSON and SSE API, input validation, disconnect cancellation, health and metrics endpoints.
- HF sequential control, exact-token workload generation, burst/staggered arrivals, warm-up/repetitions, correctness checks before and during measurement, JSON/CSV aggregation, and a separate client-observed SSE load tool.

## Deliberately deferred

Static-batch baseline, alternative scheduling policies, chunked prefill, paged/static KV allocation, prefix caching, CPU offload, quantization, multi-GPU/NCCL, production authentication, and UI work. The project does not claim to outperform mature serving systems.

## Acceptance status

The Phase 1 V100 data supplied by the user passes six correctness comparisons and all fifteen benchmark-pair token-ID comparisons. The raw files in `evidence/phase1_v100` are unchanged copies of that run. HF and manual sequential throughput are 30.5674 and 31.1452 tokens/s, respectively; these do not measure batching.

New functionality is tested locally using real tiny Qwen2 forwards, including cache packing, joins/exits, EOS, reservation accounting, FIFO blocking, timeouts, cancellation, model errors, concurrent HTTP clients and benchmark reporting. A saved synthetic checkpoint also exercises the complete final script. These CPU smoke timings are not GPU performance results.

Real-Qwen variable-length batched decode, CUDA memory behavior, and full serving performance still require the final GPU script. An implementation milestone can be complete while its GPU evaluation remains pending; reports and README must preserve that distinction.

## Measurement interpretation

- In-process TTFT starts after CPU tokenization at engine submission. HTTP TTFT starts before sending the HTTP request and includes transport, tokenization, queueing and response buffering.
- TPOT averages intervals after the first token. For one-token outputs it is undefined/null.
- HF sequential TTFT/TPOT are null because `generate()` returns the full output at once. Compare HF against the engine on workload throughput and completion latency; use client metrics to characterize streaming separately.
- Evaluation throughput divides successfully completed output tokens by entire workload wall time, including scheduled arrival gaps. Every failure/rejection remains in the raw report.
- Engine admission queue time begins at submission; HF simulated queue time begins at each predetermined arrival time. Host enqueue overhead is included in overall workload wall time but these per-request queue clocks are not perfectly identical.
- Comparison CSV latency columns average per-run percentiles; they are not pooled percentiles across all requests.
- The 8/48 MiB budgets count compact caches only. For default Qwen FP16, a 256+64-token request reserves 3.75 MiB. Thus 8 MiB constrains concurrency while 48 MiB normally accommodates the default batch. Neither limit is a physical VRAM cap.
- CUDA allocations include weights, activations, original caches, padded batch copies and unpacked copies. Every decode can incur copying overhead. CUDA output synchronization is included in timed generation.
- FIFO may block shorter requests behind a larger head request. Long unchunked prefills pause decode work. These are intended limitations of the reduced policy, not hidden optimizations.

## Results to discuss after the final GPU run

1. Confirm real batch sizes above one and changing membership in decode traces.
2. Compare throughput and completion latency at 1, 4 and 8 requests, including regressions.
3. Compare queueing and P95 TTFT at 8 vs 48 MiB budgets.
4. Contrast in-process metrics with client-observed SSE metrics without treating the workloads as identical.
5. Report GPU/software versions, all raw runs, variability, and the padding/copying limitation.
