import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from fastapi.testclient import TestClient
import pytest
import torch

from miniserve.api import create_app, _stream_events, _wait_for_result
from miniserve.engine import InferenceEngine
from miniserve.request import RequestState


class Tokenizer:
    def apply_chat_template(self, messages, **kw):
        return torch.tensor([[3 + ord(x) % 70 for x in messages[0]['content']]])
    def __call__(self, text, **kw):
        return {'input_ids': self.apply_chat_template([{'content': text}])}
    def decode(self, ids, **kw):
        return ''.join(chr(65 + x % 26) for x in ids)


def test_four_streams_are_isolated_and_routes_work(tiny_backend):
    tiny_backend.tokenizer = Tokenizer()
    prompts = ['one', 'two longer', '中文测试', 'four']
    expected = {p: tiny_backend.generate_hf_ids(tiny_backend.tokenizer(p)['input_ids'], 8) for p in prompts}
    engine = InferenceEngine(tiny_backend)
    with TestClient(create_app(engine=engine)) as client:
        assert client.get('/health').json()['ready']
        def request(prompt):
            response = client.post('/v1/generate/stream', json={'prompt': prompt, 'max_new_tokens': 8, 'temperature': 0})
            assert response.status_code == 200
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            assert events[-1]['type'] == 'done'
            assert [e['token_id'] for e in events if e['type'] == 'token'] == expected[prompt]
            assert len({e['request_id'] for e in events}) == 1
            return events[-1]['request_id']
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert len(set(pool.map(request, prompts))) == 4
        response = client.post('/v1/generate', json={'prompt': 'one', 'max_new_tokens': 8})
        assert response.json()['output_ids'] == expected['one']
        assert client.get('/metrics').json()['kv_reserved_bytes'] == 0
        for body in ({'prompt': ''}, {'prompt': 'x', 'temperature': 0.5}, {'prompt': 'x', 'max_new_tokens': True}):
            assert client.post('/v1/generate', json=body).status_code == 422
        assert client.post('/v1/generate', json={'prompt': 'x', 'max_new_tokens': 500}).status_code == 400
        assert client.post('/v1/generate', json={'prompt': 'x' * 128, 'max_new_tokens': 2}).status_code == 413
    assert not engine.snapshot()['worker_alive']


def test_stream_disconnect_requests_cancellation():
    class Engine:
        cancelled = []
        def cancel(self, request_id):
            self.cancelled.append(request_id)
    class Disconnected:
        async def is_disconnected(self):
            return True
    state = RequestState(torch.tensor([[3]]), 8, set())
    engine = Engine()
    async def consume():
        return [event async for event in _stream_events(engine, state, Disconnected())]
    assert asyncio.run(consume()) == []
    assert engine.cancelled == [state.request_id]


def test_cancelled_stream_task_cleans_up():
    class Engine:
        cancelled = []
        def cancel(self, request_id):
            self.cancelled.append(request_id)
    class Connected:
        async def is_disconnected(self):
            return False
    state = RequestState(torch.tensor([[3]]), 8, set())
    engine = Engine()
    async def run():
        stream = _stream_events(engine, state, Connected())
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.aclose()
    asyncio.run(run())
    assert engine.cancelled == [state.request_id]


def test_incomplete_unicode_is_buffered_until_valid(tiny_backend):
    class ByteTokenizer(Tokenizer):
        def decode(self, ids, **kw):
            return ['\ufffd', '中', '中文'][len(ids) - 1]
    tiny_backend.tokenizer = ByteTokenizer()
    engine = InferenceEngine(tiny_backend)
    engine.start()
    try:
        request = engine.submit_ids(torch.tensor([[3, 4]]), 3, [])
        assert request.done.wait(5)
        events = list(request.output_queue.queue)
        assert [event['text'] for event in events[:-1]] == ['', '中', '文']
        assert events[-1]['text'] == '中文'
    finally:
        engine.close()
