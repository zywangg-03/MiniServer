"""Sequential Hugging Face generate() baseline; use on local or SSH GPUs."""

from benchmarks.common import make_parser, run_benchmark


def main(argv: list[str] | None = None) -> int:
    args = make_parser("Measure the HF greedy single-request baseline.", "hf_baseline.json", benchmark=True).parse_args(argv)
    run_benchmark(args, manual=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
