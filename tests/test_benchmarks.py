"""Validate reported data and exit status without loading model weights."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import common, verify
from benchmarks.hf_baseline import main as hf_main
from benchmarks.manual_generate import main as manual_main


class RecordingBackend:
    def __init__(self, *, mismatch=False):
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 3]))
        self.tokenizer = SimpleNamespace(chat_template="{{ messages }}", decode=lambda ids, **_: " ".join(map(str, ids)))
        self.calls = []
        self.mismatch = mismatch

    def environment_info(self):
        return {"device": "cpu", "test_backend": True}

    def prepare_prompt(self, prompt, *, use_chat_template=True):
        self.calls.append(("prompt", prompt, use_chat_template))
        return torch.tensor([[10, len(prompt)]])

    def generate_hf_ids(self, input_ids, *, max_new_tokens):
        self.calls.append(("hf", input_ids.tolist(), max_new_tokens))
        return [4] * max_new_tokens

    def generate_ids(self, input_ids, *, max_new_tokens):
        self.calls.append(("manual", input_ids.tolist(), max_new_tokens))
        return [5 if self.mismatch else 4] * max_new_tokens


@pytest.mark.parametrize("entrypoint,method", [(hf_main, "hf"), (manual_main, "manual")])
def test_benchmark_warmup_excluded_and_raw_records_preserved(tmp_path, monkeypatch, entrypoint, method):
    backend = RecordingBackend()
    monkeypatch.setattr(common.HFBackend, "from_config", lambda config: backend)
    # Two measured requests each take 0.5 seconds; warm-up must not be counted.
    ticks = iter([10.0, 10.5, 20.0, 20.5])
    monkeypatch.setattr(common.time, "perf_counter", lambda: next(ticks))
    output = tmp_path / "nested" / "report.json"
    assert entrypoint([
        "--device", "cpu", "--dtype", "float32", "--prompt", "中文",
        "--raw-prompt", "--warmup", "3", "--repetitions", "2",
        "--max-new-tokens", "2", "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text())
    assert len([call for call in backend.calls if call[0] == method]) == 5
    assert report["summary"]["requests"] == 2
    assert report["summary"]["total_output_tokens"] == 4
    assert report["summary"]["total_generation_seconds"] == 1.0
    assert report["summary"]["sequential_generated_tokens_per_second"] == 4.0
    assert report["summary"]["cuda_peak_allocated_bytes"] is None
    assert report["records"][0]["input_ids"] == [10, 2]
    assert report["records"][0]["output_ids"] == [4, 4]
    assert report["generation"]["use_chat_template"] is False
    assert report["measurement"]["client_ttft_seconds"] is None
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("mismatch,exit_code", [(False, 0), (True, 1)])
def test_verify_writes_report_on_match_or_mismatch(tmp_path, monkeypatch, mismatch, exit_code):
    backend = RecordingBackend(mismatch=mismatch)
    monkeypatch.setattr(verify.HFBackend, "from_config", lambda config: backend)
    output = tmp_path / "verification.json"
    assert verify.main([
        "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "3",
        "--output", str(output),
    ]) == exit_code
    report = json.loads(output.read_text())
    assert report["passed"] is not mismatch
    assert report["comparisons"] == 6  # Three default prompts, two lengths each.
    assert {record["max_new_tokens"] for record in report["records"]} == {1, 3}
    assert all(record["first_difference_index"] == (0 if mismatch else None) for record in report["records"])


@pytest.mark.parametrize("expected,actual,index", [([1, 2], [1, 2], None), ([1, 2], [1], 1), ([1], [1, 2], 1), ([1, 2], [1, 3], 1)])
def test_first_difference_detects_length_and_token_mismatch(expected, actual, index):
    assert verify.first_difference(expected, actual) == index


@pytest.mark.parametrize("option,value", [("--warmup", "-1"), ("--repetitions", "0"), ("--max-new-tokens", "0"), ("--seed", "-1")])
def test_invalid_cli_counts_rejected_before_model_load(monkeypatch, option, value):
    def unexpected_load(config):
        pytest.fail("invalid CLI arguments must fail before loading weights")

    monkeypatch.setattr(common.HFBackend, "from_config", unexpected_load)
    with pytest.raises(SystemExit) as error:
        hf_main([option, value])
    assert error.value.code == 2
