"""Check real model token IDs before collecting performance measurements."""

from __future__ import annotations

import json

from benchmarks.common import config_from_args, make_parser, metadata, prepare_inputs, write_report
from miniserve.backends.hf_backend import HFBackend


def first_difference(expected: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    return min(len(expected), len(actual)) if len(expected) != len(actual) else None


def main(argv: list[str] | None = None) -> int:
    args = make_parser("Compare exact greedy IDs against HF generate().", "correctness.json", benchmark=False).parse_args(argv)
    config = config_from_args(args)
    backend = HFBackend.from_config(config)
    records = []
    for prompt_index, (prompt, input_ids) in enumerate(prepare_inputs(backend, args)):
        for limit in sorted({1, args.max_new_tokens}):
            expected = backend.generate_hf_ids(input_ids, max_new_tokens=limit)
            actual = backend.generate_ids(input_ids, max_new_tokens=limit)
            mismatch = first_difference(expected, actual)
            records.append({
                "prompt_index": prompt_index,
                "prompt": prompt,
                "input_ids": input_ids[0].tolist(),
                "input_tokens": input_ids.shape[1],
                "max_new_tokens": limit,
                "hf_output_ids": expected,
                "manual_output_ids": actual,
                "matches": mismatch is None,
                "first_difference_index": mismatch,
                "hf_text": backend.tokenizer.decode(expected, skip_special_tokens=True),
                "manual_text": backend.tokenizer.decode(actual, skip_special_tokens=True),
            })
    passed = all(record["matches"] for record in records)
    report = metadata(backend, config, args, "greedy_correctness")
    report.update({"passed": passed, "comparisons": len(records), "records": records})
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "passed": passed, "comparisons": len(records)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
