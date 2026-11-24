import asyncio
import json
from types import SimpleNamespace

import httpx
import torch

from benchmarks.compare import summarize as compare
from benchmarks.load_test import one_request
from benchmarks.serving_eval import run_case
from miniserve.config import ServingConfig


def test_engine_experiment_compares_same_tokens_and_counts_failures(tiny_backend):
    inputs = [torch.tensor([[3, 4]]), torch.tensor([[5, 6, 7, 8]])]
    expected = [tiny_backend.generate_hf_ids(x, 8) for x in inputs]
    runs = []
    for mode in ('hf_sequential', 'continuous'):
        run = run_case(tiny_backend, inputs, expected, max_new_tokens=8, mode=mode, pattern='burst', arrival_interval=0.001, serving_config=ServingConfig())
        assert run['summary']['completed_requests'] == 2
        assert run['summary']['all_outputs_match']
        assert run['summary']['output_tokens_per_second'] == sum(len(x) for x in expected) / run['summary']['wall_seconds']
        assert run['cuda_peak_memory']['allocated_bytes'] is None
        runs.append(run)
    summary = compare({'kind': 'serving_evaluation', 'passed': True, 'environment': {}, 'runs': runs})
    assert len(summary['rows']) == 2
    for row in summary['rows']:
        assert row['throughput_stdev'] is None
        if row['mode'] == 'hf_sequential':
            assert row['throughput_ratio_to_hf'] == 1
            assert row['mean_run_p95_ttft_seconds'] is None
    failed = run_case(tiny_backend, inputs, expected, max_new_tokens=8, mode='continuous', pattern='burst', arrival_interval=0.001, serving_config=ServingConfig(kv_budget_mb=0.0001))
    assert failed['summary']['failed_requests'] == 2
    assert not failed['summary']['all_outputs_match']
    assert failed['summary']['output_tokens_per_second'] == 0


def test_sse_client_counts_empty_text_tokens_and_rejects_incomplete_streams():
    async def run(events, status=200):
        payload = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events)
        async def handler(request):
            return httpx.Response(status, text=payload)
        async with httpx.AsyncClient(base_url='http://test', transport=httpx.MockTransport(handler)) as client:
            return await one_request(client, 0, 'hello', 2)
    events = [{'type': 'token', 'token_id': 4, 'text': ''}, {'type': 'token', 'token_id': 5, 'text': '中文'}, {'type': 'done', 'status': 'finished', 'output_ids': [4, 5]}]
    record = asyncio.run(run(events))
    assert record['status'] == 'finished'
    assert record['output_ids'] == [4, 5]
    assert len(record['token_arrival_offsets_seconds']) == 2
    assert record['tpot_seconds'] is not None
    assert asyncio.run(run(events[:-1]))['status'] == 'error'
    assert asyncio.run(run([], status=429))['http_status'] == 429
