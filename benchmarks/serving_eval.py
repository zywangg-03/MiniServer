"""Compare HF sequential execution and real continuous batching in one process.

This is a controlled engine experiment, not an HTTP latency benchmark. Inputs
are exact-length token workloads; every measured output is checked against HF.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from benchmarks.common import DEFAULT_PROMPTS, config_from_args, make_parser, percentile, positive_int, synchronize, write_report
from miniserve.backends.hf_backend import HFBackend
from miniserve.config import ServingConfig
from miniserve.engine import InferenceEngine


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def make_workload(backend: HFBackend, lengths: list[int], count: int) -> list[torch.Tensor]:
    """Repeat natural-text token IDs to exact lengths; no claims of natural prose."""
    seeds = [backend.prepare_prompt(text, use_chat_template=False)[0].cpu() for text in DEFAULT_PROMPTS]
    inputs = []
    for i in range(count):
        length = lengths[i % len(lengths)]
        seed = seeds[i % len(seeds)]
        inputs.append(seed.repeat((length + len(seed) - 1) // len(seed))[:length].unsqueeze(0))
    return inputs


def cuda_peaks(backend: HFBackend) -> dict:
    if backend.device.type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(backend.device),
        "reserved_bytes": torch.cuda.max_memory_reserved(backend.device),
    }


def summarize(records: list[dict], elapsed: float) -> dict:
    successful = [r for r in records if r.get("status") == "finished"]
    tokens = sum(len(r.get("output_ids", [])) for r in successful)
    summary = {
        "submitted_requests": len(records), "completed_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "total_output_tokens": tokens, "wall_seconds": elapsed,
        "output_tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
        "all_outputs_match": all(r.get("matches_hf") is True for r in records),
    }
    for key in ("queue_seconds", "ttft_seconds", "tpot_seconds", "latency_seconds"):
        values = [r["metrics"][key] for r in successful if r.get("metrics", {}).get(key) is not None]
        summary[f"p50_{key}"] = percentile(values, 0.5) if values else None
        summary[f"p95_{key}"] = percentile(values, 0.95) if values else None
    return summary


def run_case(backend: HFBackend, inputs: list[torch.Tensor], expected: list[list[int]], *,
             max_new_tokens: int, mode: str, pattern: str, arrival_interval: float,
             serving_config: ServingConfig, timeout: float = 300.0) -> dict:
    """Timed interval includes scheduled arrival gaps, queueing, and completion."""
    engine = InferenceEngine(backend, serving_config) if mode == "continuous" else None
    if engine is not None:
        engine.start()
    synchronize(backend.device)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    records: list[dict] = []
    pending = []
    snapshot = None
    try:
        for i, ids in enumerate(inputs):
            offset = i * arrival_interval if pattern == "staggered" else 0.0
            target = started + offset
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            base = {"request_index": i, "input_ids": ids[0].tolist(), "input_tokens": ids.shape[1], "scheduled_arrival_offset_seconds": offset}
            if engine is not None:
                try:
                    handle = engine.submit_ids(ids, max_new_tokens)
                    pending.append((base, handle))
                except Exception as error:
                    records.append({**base, "status": "rejected", "error": str(error), "output_ids": [], "matches_hf": False, "metrics": {}})
            else:
                admission = time.perf_counter()
                output = backend.generate_hf_ids(ids, max_new_tokens)
                synchronize(backend.device)
                finished = time.perf_counter()
                records.append({**base, "status": "finished", "output_ids": output, "matches_hf": output == expected[i], "metrics": {
                    "arrival_time": target, "prefill_start_time": admission, "finish_time": finished,
                    "queue_seconds": admission - target, "latency_seconds": finished - target,
                    "ttft_seconds": None, "tpot_seconds": None,
                }})
        for base, handle in pending:
            remaining = max(0.0, started + timeout - time.perf_counter())
            if not handle.done.wait(remaining):
                engine.cancel(handle.request_id)
                records.append({**base, "status": "timeout", "output_ids": [], "matches_hf": False, "metrics": {}})
            else:
                result = handle.result()
                records.append({**base, **result, "matches_hf": result.get("output_ids") == expected[base["request_index"]]})
        synchronize(backend.device)
        elapsed = time.perf_counter() - started
        if engine is not None:
            snapshot = engine.snapshot()
        return {"mode": mode, "pattern": pattern, "requests": len(inputs),
                "input_lengths": [ids.shape[1] for ids in inputs], "serving_config": asdict(serving_config),
                "summary": summarize(records, elapsed), "cuda_peak_memory": cuda_peaks(backend),
                "engine": snapshot, "records": sorted(records, key=lambda r: r["request_index"])}
    finally:
        if engine is not None:
            engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("Controlled HF sequential versus continuous-batching experiment.", "serving_eval.json", benchmark=True)
    parser.add_argument("--input-lengths", nargs="+", type=positive_int, default=[128, 256])
    parser.add_argument("--concurrency", nargs="+", type=positive_int, default=[1, 4, 8])
    parser.add_argument("--patterns", nargs="+", choices=["burst", "staggered"], default=["burst", "staggered"])
    parser.add_argument("--kv-budgets-mb", nargs="+", type=positive_float, default=[8, 48])
    parser.add_argument("--max-batch-size", type=positive_int, default=8)
    parser.add_argument("--arrival-interval", type=positive_float, default=0.05)
    parser.add_argument("--timeout", type=positive_float, default=300)
    args = parser.parse_args(argv)
    if args.prompt or args.raw_prompt:
        parser.error("This experiment constructs exact-length raw token workloads; --prompt and --raw-prompt do not apply.")
    config = config_from_args(args)
    backend = HFBackend.from_config(config)
    if max(args.input_lengths) + args.max_new_tokens > backend.context_limit:
        parser.error("input length + max-new-tokens exceeds the model context limit")
    inputs = make_workload(backend, args.input_lengths, max(args.concurrency))
    expected = [backend.generate_hf_ids(ids, args.max_new_tokens) for ids in inputs]
    report = {
        "schema_version": 1, "kind": "serving_evaluation", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": backend.environment_info(), "model_config": asdict(config),
        "workload": {"input_lengths": args.input_lengths, "max_new_tokens": args.max_new_tokens,
                     "concurrency": args.concurrency, "patterns": args.patterns,
                     "arrival_interval_seconds": args.arrival_interval,
                     "warmup": args.warmup, "repetitions": args.repetitions,
                     "description": "Exact-length repeated natural-text token IDs; EOS follows model generation_config."},
        "measurement": {
            "scope": "in-process engine; not HTTP/client latency",
            "throughput": "Completed output tokens / entire workload wall time, including scheduled arrival gaps and waiting.",
            "sequential_arrivals": "Queue wait is measured against the predetermined arrival schedule.",
            "hf_ttft_tpot": "null: HF generate() returns all output IDs at completion; no fabricated token times.",
            "memory": "CUDA allocator peaks include weights, padded cache copies and temporary allocations. Logical KV budgets are not physical VRAM limits.",
        }, "preflight": [], "runs": [], "passed": True,
    }
    for budget in args.kv_budgets_mb:
        limits = ServingConfig(max_batch_size=args.max_batch_size, max_queue_size=max(64, max(args.concurrency)),
                               kv_budget_mb=budget, max_prompt_tokens=max(args.input_lengths),
                               max_new_tokens=args.max_new_tokens, queue_timeout_seconds=args.timeout)
        preflight = run_case(backend, inputs, expected, max_new_tokens=args.max_new_tokens,
                             mode="continuous", pattern="burst", arrival_interval=args.arrival_interval,
                             serving_config=limits, timeout=args.timeout)
        report["preflight"].append(preflight)
        if not preflight["summary"]["all_outputs_match"]:
            report["passed"] = False
            write_report(args.output, report)
            print(f"Correctness preflight failed; inspect {args.output}")
            return 1
        for count in args.concurrency:
            for pattern in args.patterns:
                # A repeated HF control under each budget keeps the paired report simple.
                # Budget is inactive for HF; never interpret this as an HF memory limit.
                for repetition in range(-args.warmup, args.repetitions):
                    # Alternate order to reduce a fixed execution-order bias.
                    modes = ["hf_sequential", "continuous"] if repetition % 2 == 0 else ["continuous", "hf_sequential"]
                    for mode in modes:
                        run = run_case(backend, inputs[:count], expected[:count], max_new_tokens=args.max_new_tokens,
                                       mode=mode, pattern=pattern, arrival_interval=args.arrival_interval,
                                       serving_config=limits, timeout=args.timeout)
                        if repetition >= 0:
                            run["repetition"] = repetition
                            report["runs"].append(run)
                        if not run["summary"]["all_outputs_match"]:
                            report["passed"] = False
                            if repetition < 0:
                                report.setdefault("failed_warmups", []).append(run)
                            write_report(args.output, report)
                            print(f"Output mismatch or request failure; inspect {args.output}")
                            return 1
                        if repetition >= 0:
                            write_report(args.output, report)
                            print(f"{mode} n={count} {pattern} KV={budget:g} MiB rep={repetition}: {run['summary']['output_tokens_per_second']:.2f} tokens/s")
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
