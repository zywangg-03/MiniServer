"""Shared CLI and measurement code for local and remote single-GPU runs.

Run this module's clients on the GPU server itself after SSH login. Device IDs
are process-local, so existing CUDA_VISIBLE_DEVICES settings remain intact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

import torch

from miniserve.backends.hf_backend import HFBackend
from miniserve.config import ModelConfig


DEFAULT_PROMPTS = (
    "Explain what a key-value cache does in a language model.",
    "请用两句话解释什么是自回归语言模型。",
    "A server receives requests with different prompt lengths. Explain why "
    "caching previous attention keys and values can reduce repeated work, and "
    "describe one memory cost of keeping those caches during generation.",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def make_parser(description: str, output_name: str, *, benchmark: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    defaults = ModelConfig()
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=defaults.dtype)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default=defaults.attn_implementation)
    parser.add_argument("--seed", type=nonnegative_int, default=defaults.seed)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true", help="Skip the tokenizer chat template.")
    parser.add_argument("--prompt", action="append", help="Repeat to supply multiple prompts; defaults to three English/Chinese prompts.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results") / output_name)
    if benchmark:
        parser.add_argument("--warmup", type=nonnegative_int, default=1, help="Warm-up passes over every prompt (excluded from measurements).")
        parser.add_argument("--repetitions", type=positive_int, default=5)
    return parser


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        use_chat_template=not args.raw_prompt,
        local_files_only=args.local_files_only,
    )


def prepare_inputs(backend: HFBackend, args: argparse.Namespace) -> list[tuple[str, torch.Tensor]]:
    return [
        (prompt, backend.prepare_prompt(prompt, use_chat_template=not args.raw_prompt))
        for prompt in (args.prompt if args.prompt is not None else DEFAULT_PROMPTS)
    ]


def metadata(backend: HFBackend, config: ModelConfig, args: argparse.Namespace, kind: str) -> dict[str, Any]:
    template = getattr(backend.tokenizer, "chat_template", None)
    template_payload = str(template).encode("utf-8")
    return {
        "schema_version": 1,
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "environment": backend.environment_info(),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "eos_token_id": backend.model.generation_config.eos_token_id,
            "use_chat_template": not args.raw_prompt,
            "chat_template_sha256": hashlib.sha256(template_payload).hexdigest() if template is not None else None,
        },
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Replace a report atomically, keeping partial JSON out of result files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_generation(backend: HFBackend, input_ids: torch.Tensor, max_new_tokens: int, *, manual: bool) -> dict[str, Any]:
    """Measure generation wall time including CUDA completion and IDs to host.

    Model load, prompt tokenization, output text decoding, and warm-up are outside
    this interval. Memory peaks include resident model weights, not just KV cache.
    This is an in-process generation measurement, not client latency or TTFT.
    """
    generate = backend.generate_ids if manual else backend.generate_hf_ids
    synchronize(backend.device)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    output_ids = generate(input_ids, max_new_tokens=max_new_tokens)
    synchronize(backend.device)
    elapsed = time.perf_counter() - started
    return {
        "output_ids": output_ids,
        "output_tokens": len(output_ids),
        "generation_seconds": elapsed,
        "generated_tokens_per_second": len(output_ids) / elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(backend.device) if backend.device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(backend.device) if backend.device.type == "cuda" else None,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def run_benchmark(args: argparse.Namespace, *, manual: bool) -> dict[str, Any]:
    config = config_from_args(args)
    backend = HFBackend.from_config(config)
    inputs = prepare_inputs(backend, args)
    generate = backend.generate_ids if manual else backend.generate_hf_ids
    for _ in range(args.warmup):
        for _, input_ids in inputs:
            generate(input_ids, max_new_tokens=args.max_new_tokens)
    synchronize(backend.device)

    records = []
    for repetition in range(args.repetitions):
        for prompt_index, (prompt, input_ids) in enumerate(inputs):
            record = {
                "repetition": repetition,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "input_ids": input_ids[0].tolist(),
                "input_tokens": input_ids.shape[1],
                **measure_generation(backend, input_ids, args.max_new_tokens, manual=manual),
            }
            record["text"] = backend.tokenizer.decode(record["output_ids"], skip_special_tokens=True)
            records.append(record)

    times = [record["generation_seconds"] for record in records]
    total_time = sum(times)
    total_tokens = sum(record["output_tokens"] for record in records)
    report = metadata(backend, config, args, "manual_sequential" if manual else "hf_sequential")
    report.update({
        "measurement": {
            "warmup_passes": args.warmup,
            "repetitions": args.repetitions,
            "timing": "Synchronized in-process generation wall time; includes output IDs to host; excludes load, tokenization, text decoding and warm-up.",
            "memory": "Per-request CUDA allocator peak including resident weights; reserved memory can include reusable allocations from prior requests.",
            "throughput_denominator": "Sum of measured request generation seconds; excludes between-request reporting overhead.",
            "client_ttft_seconds": None,
            "client_tpot_seconds": None,
        },
        "summary": {
            "requests": len(records),
            "total_output_tokens": total_tokens,
            "total_generation_seconds": total_time,
            "sequential_generated_tokens_per_second": total_tokens / total_time,
            "p50_generation_seconds": statistics.median(times),
            "p95_generation_seconds": percentile(times, 0.95),
            "cuda_peak_allocated_bytes": max((r["cuda_peak_allocated_bytes"] for r in records if r["cuda_peak_allocated_bytes"] is not None), default=None),
            "cuda_peak_reserved_bytes": max((r["cuda_peak_reserved_bytes"] for r in records if r["cuda_peak_reserved_bytes"] is not None), default=None),
        },
        "records": records,
    })
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return report
