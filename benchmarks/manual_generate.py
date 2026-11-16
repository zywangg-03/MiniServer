"""Measure MiniServe's manual prefill/decode loop using the same workload."""

from benchmarks.common import make_parser, run_benchmark


def main(argv: list[str] | None = None) -> int:
    args = make_parser("Measure manual greedy prefill/decode generation.", "manual_generate.json", benchmark=True).parse_args(argv)
    run_benchmark(args, manual=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
