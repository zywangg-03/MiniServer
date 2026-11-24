"""Lifecycle, true batching and memory reservations with real tiny Qwen forwards."""
import threading
import time

import pytest
import torch

from miniserve.config import ServingConfig
from miniserve.engine import InferenceEngine, RejectedRequest
from miniserve.kv_manager import KVManager, model_kv_bytes_per_token
from miniserve.request import RequestStatus


def wait(request):
    assert request.done.wait(10), 'worker did not complete'
    return request.result()


def test_dynamic_batch_membership_and_exact_outputs(tiny_backend):
    inputs = [torch.tensor([[3, 4]]), torch.tensor([[5, 7, 8, 9, 10]]), torch.tensor([[8, 9, 6]])]
    limits = [15, 8, 10]
    expected = [tiny_backend.generate_hf_ids(x, n, eos_token_ids=[]) for x, n in zip(inputs, limits)]
    engine = InferenceEngine(tiny_backend)
    entered, release = threading.Event(), threading.Event()
    original = tiny_backend.prefill
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    tiny_backend.prefill = blocked
    engine.start()
    try:
        a = engine.submit_ids(inputs[0], limits[0], [])
        assert entered.wait(5)
        b = engine.submit_ids(inputs[1], limits[1], [])
        c = engine.submit_ids(inputs[2], limits[2], [])
        release.set()
        results = [wait(r) for r in (a, b, c)]
        assert [r['output_ids'] for r in results] == expected
        snapshot = engine.snapshot()
        members = [b['request_ids'] for b in snapshot['decode_batches']]
        assert [a.request_id] in members
        assert [a.request_id, b.request_id] in members
        assert [a.request_id, b.request_id, c.request_id] in members
        assert any(a.request_id in m and b.request_id not in m for m in members[3:])
        assert snapshot['kv_reserved_bytes'] == 0
        assert snapshot['waiting'] == snapshot['running'] == 0
        assert all(r.past_key_values is None for r in (a, b, c))
        for r in (a, b, c):
            events = list(r.output_queue.queue)
            assert len(events) == len(r.generated_ids) + 1
            assert events[-1]['type'] == 'done'
            metrics = r.result()['metrics']
            assert metrics['latency_seconds'] >= metrics['ttft_seconds'] >= metrics['queue_seconds'] >= 0
    finally:
        release.set()
        engine.close()


def test_kv_reservation_fifo_and_release(tiny_backend):
    # Each request reserves 20 tokens: budget permits only one active request.
    bytes_per_token = model_kv_bytes_per_token(tiny_backend.model)
    engine = InferenceEngine(tiny_backend, ServingConfig(kv_budget_mb=20 * bytes_per_token / 2**20))
    engine.start()
    try:
        requests = [engine.submit_ids(torch.tensor([[3, 4]]), 18, []) for _ in range(3)]
        assert all(wait(r)['status'] == 'finished' for r in requests)
        assert requests[1].prefill_start_time >= requests[0].finish_time
        assert requests[2].prefill_start_time >= requests[1].finish_time
        assert max(engine.snapshot()['decode_batch_sizes']) == 1
        assert engine.snapshot()['kv_peak_reserved_bytes'] == 20 * bytes_per_token
        assert engine.snapshot()['kv_reserved_bytes'] == 0
        with pytest.raises(RejectedRequest, match='request_exceeds_kv_budget'):
            engine.submit_ids(torch.tensor([[3, 4, 5]]), 18)
    finally:
        engine.close()


def test_queue_full_cancel_and_timeout(tiny_backend):
    engine = InferenceEngine(tiny_backend, ServingConfig(max_batch_size=1, max_queue_size=1, queue_timeout_seconds=0.03))
    entered, release = threading.Event(), threading.Event()
    original = tiny_backend.prefill
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(5)
        return original(*a, **kw)
    tiny_backend.prefill = blocked
    engine.start()
    try:
        active = engine.submit_ids(torch.tensor([[3, 4]]), 12, [])
        assert entered.wait(5)
        queued = engine.submit_ids(torch.tensor([[3, 4]]), 12, [])
        with pytest.raises(RejectedRequest) as error:
            engine.submit_ids(torch.tensor([[3, 4]]), 12)
        assert error.value.status_code == 429
        engine.cancel(active.request_id)
        time.sleep(0.04)
        release.set()
        assert wait(active)['status'] == 'cancelled'
        assert wait(queued)['finish_reason'] == 'queue_timeout'
        assert engine.snapshot()['kv_reserved_bytes'] == 0
        assert active.past_key_values is None
    finally:
        release.set()
        engine.close()


def test_cancel_waiting_and_shutdown(tiny_backend):
    engine = InferenceEngine(tiny_backend, ServingConfig(max_batch_size=1))
    entered, release = threading.Event(), threading.Event()
    original = tiny_backend.prefill
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(5)
        return original(*a, **kw)
    tiny_backend.prefill = blocked
    engine.start()
    a = engine.submit_ids(torch.tensor([[3]]), 20, [])
    assert entered.wait(5)
    b = engine.submit_ids(torch.tensor([[4]]), 20, [])
    engine.cancel(b.request_id)
    engine.cancel(a.request_id)
    release.set()
    assert wait(b)['status'] == 'cancelled'
    engine.close()
    assert wait(a)['status'] == 'cancelled'
    assert engine.snapshot()['kv_reserved_bytes'] == 0
    assert not engine.snapshot()['worker_alive']
    with pytest.raises(RejectedRequest) as error:
        engine.submit_ids(torch.tensor([[3]]), 1)
    assert error.value.status_code == 503


def test_model_error_releases_resources_and_worker_survives(tiny_backend):
    def fail(*a, **kw):
        raise RuntimeError('synthetic forward failure')
    original = tiny_backend.prefill
    tiny_backend.prefill = fail
    engine = InferenceEngine(tiny_backend)
    engine.start()
    try:
        assert wait(engine.submit_ids(torch.tensor([[3]]), 2))['status'] == 'error'
        assert engine.snapshot()['kv_reserved_bytes'] == 0
        tiny_backend.prefill = original
        assert wait(engine.submit_ids(torch.tensor([[3]]), 2))['status'] == 'finished'
    finally:
        engine.close()


def test_budget_release_is_idempotent_and_duplicate_reserve_rejected():
    manager = KVManager(100, 10)
    assert manager.reserve('a', 60)
    assert not manager.reserve('b', 50)
    with pytest.raises(ValueError):
        manager.reserve('a', 1)
    manager.release('a')
    manager.release('a')
    assert manager.reserved_bytes == 0
    assert manager.reserve('b', 100)


@pytest.mark.parametrize('kwargs', [{'max_batch_size': 0}, {'max_queue_size': True}, {'kv_budget_mb': float('nan')}, {'queue_timeout_seconds': 0}])
def test_serving_config_validation(kwargs):
    with pytest.raises(ValueError):
        ServingConfig(**kwargs)
