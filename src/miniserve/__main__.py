"""Run one server process on a local GPU or inside an allocated cluster job."""

import argparse

import uvicorn

from miniserve.api import create_app
from miniserve.config import ModelConfig, ServingConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MiniServe single-GPU continuous batching server")
    defaults = ModelConfig()
    serving_defaults = ServingConfig()
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--revision")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=defaults.dtype)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default=defaults.attn_implementation)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, default=serving_defaults.max_batch_size)
    parser.add_argument("--max-queue-size", type=int, default=serving_defaults.max_queue_size)
    parser.add_argument("--kv-budget-mb", type=float, default=serving_defaults.kv_budget_mb)
    parser.add_argument("--queue-timeout-seconds", type=float, default=serving_defaults.queue_timeout_seconds)
    parser.add_argument("--max-new-tokens", type=int, default=serving_defaults.max_new_tokens)
    parser.add_argument("--max-prompt-tokens", type=int, default=serving_defaults.max_prompt_tokens)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        model_config = ModelConfig(
            model_name=args.model, revision=args.revision, device=args.device,
            dtype=args.dtype, attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only, seed=args.seed,
        )
        serving_config = ServingConfig(
            max_batch_size=args.max_batch_size, max_queue_size=args.max_queue_size,
            kv_budget_mb=args.kv_budget_mb, queue_timeout_seconds=args.queue_timeout_seconds,
            max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens,
        )
    except ValueError as error:
        parser.error(str(error))
    # Multiple uvicorn workers would duplicate GPU weights and split scheduling.
    uvicorn.run(create_app(model_config=model_config, serving_config=serving_config),
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
