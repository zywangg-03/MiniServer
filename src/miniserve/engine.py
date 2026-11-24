"""One inference thread, iteration-level membership changes, bounded resources.

Local GPU and Great Lakes jobs run this same engine. HTTP threads tokenize on
CPU and enqueue; all serving forwards and CUDA timing belong to this worker.
"""
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import asdict
import logging
import threading
import time

import torch

from miniserve.config import ServingConfig
from miniserve.kv_manager import KVManager, model_kv_bytes_per_token
from miniserve.metrics import latency_summary
from miniserve.request import RequestState, RequestStatus
from miniserve.scheduler import Scheduler

logger = logging.getLogger(__name__)


class RejectedRequest(ValueError):
    def __init__(self, reason: str, status_code: int = 400):
        super().__init__(reason)
        self.reason, self.status_code = reason, status_code


class InferenceEngine:
    def __init__(self, backend, config: ServingConfig | None = None):
        self.backend = backend
        self.config = config or ServingConfig()
        self.environment = backend.environment_info()
        self.kv = KVManager(int(self.config.kv_budget_mb * 2**20), model_kv_bytes_per_token(backend.model))
        self.scheduler = Scheduler(self.config, self.kv)
        self._condition = threading.Condition()
        self._tokenizer_lock = threading.Lock()
        self._requests: dict[str, RequestState] = {}
        self._thread = None
        self._ready = threading.Event()
        self._stopping = False
        self._accepting = False
        self._fatal_error = None
        self._counts = Counter()
        self._rejections = Counter()
        self._completed = deque(maxlen=1024)
        self._decode_batches = deque(maxlen=256)
        self._batch_histogram = Counter()
        self._gpu_memory = {'allocated_bytes': None, 'peak_allocated_bytes': None, 'peak_reserved_bytes': None}

    def start(self):
        with self._condition:
            if self._thread is not None:
                if self._thread.is_alive() and not self._stopping:
                    return
                raise RuntimeError('Engine cannot be restarted')
            self._accepting = True
            self._thread = threading.Thread(target=self._run, name='miniserve-inference', daemon=True)
            self._thread.start()
        if not self._ready.wait(30):
            self.close()
            raise RuntimeError('Inference worker initialization timed out')
        if self._fatal_error:
            raise RuntimeError(self._fatal_error)

    def close(self):
        with self._condition:
            self._accepting = False
            self._stopping = True
            for request in self._requests.values():
                request.cancel_requested.set()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise RuntimeError('Inference worker did not stop within 30 seconds')

    def _reject(self, reason, code):
        with self._condition:
            self._rejections[reason] += 1
        raise RejectedRequest(reason, code)

    def submit(self, prompt: str, max_new_tokens: int, use_chat_template: bool = True):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 65536:
            self._reject('invalid_prompt', 400)
        tokenizer = self.backend.tokenizer
        if tokenizer is None:
            self._reject('tokenizer_unavailable', 503)
        # Do not call backend.prepare_prompt here: it transfers tensors to GPU.
        with self._tokenizer_lock:
            if use_chat_template:
                ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
            else:
                ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
        return self.submit_ids(ids, max_new_tokens)

    def submit_ids(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_ids: Sequence[int] | None = None):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= self.config.max_new_tokens:
            self._reject('invalid_output_length', 400)
        if not isinstance(input_ids, torch.Tensor) or input_ids.device.type != 'cpu' or input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            self._reject('expected_nonempty_cpu_ids_shape_1_L', 400)
        n = input_ids.shape[1]
        if n > self.config.max_prompt_tokens or n + max_new_tokens > self.backend.context_limit:
            self._reject('context_length_exceeded', 413)
        if bool(((input_ids < 0) | (input_ids >= self.backend.model.config.vocab_size)).any()):
            self._reject('invalid_token_id', 400)
        eos = set(self.backend._eos_ids(eos_token_ids))
        if self.kv.required(n, max_new_tokens) > self.kv.budget_bytes:
            self._reject('request_exceeds_kv_budget', 413)
        request = RequestState(input_ids.clone(), max_new_tokens, eos)
        with self._condition:
            if not self._accepting:
                self._reject('engine_unavailable', 503)
            try:
                self.scheduler.add_request(request)
            except OverflowError:
                self._reject('waiting_queue_full', 429)
            self._requests[request.request_id] = request
            self._counts['accepted'] += 1
            self._condition.notify()
        return request

    def cancel(self, request_id):
        with self._condition:
            if request_id in self._requests:
                self._requests[request_id].cancel_requested.set()
                self._condition.notify()

    def _finish(self, request, status, reason, error=None):
        with self._condition:
            if request.done.is_set():
                return
            request.status = status
            request.finish_reason = reason
            request.error = error
            request.finish_time = time.perf_counter()
            request.past_key_values = None
            request.next_token_id = None
            self.scheduler.finish_request(request.request_id)
            self._requests.pop(request.request_id, None)
            self._counts[status.value] += 1
            self._completed.append({'request_id': request.request_id, 'status': status.value, 'finish_reason': reason, **request.metrics()})
            request.done.set()
            request.output_queue.put_nowait({'type': 'error' if status == RequestStatus.ERROR else 'done', **request.result()})

    def _emit(self, request, token_id):
        if request.cancel_requested.is_set():
            self._finish(request, RequestStatus.CANCELLED, 'cancelled')
            return
        now = time.perf_counter()
        request.generated_ids.append(token_id)
        request.next_token_id = token_id
        request.last_token_time = now
        if request.first_token_time is None:
            request.first_token_time = now
        text_delta = ''
        if self.backend.tokenizer is not None:
            with self._tokenizer_lock:
                request.text = self.backend.tokenizer.decode(request.generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # Byte-level tokens may end in an incomplete UTF-8 sequence. Buffer
            # replacement characters; done.text is always the authoritative text.
            stable = request.text.rstrip('\ufffd')
            if stable.startswith(request.emitted_text):
                text_delta = stable[len(request.emitted_text):]
                request.emitted_text = stable
        request.output_queue.put_nowait({'type': 'token', 'request_id': request.request_id, 'token_id': token_id, 'text': text_delta})
        with self._condition:
            self._counts['generated_tokens'] += 1
        if token_id in request.eos_token_ids:
            self._finish(request, RequestStatus.FINISHED, 'eos')
        elif len(request.generated_ids) == request.max_new_tokens:
            self._finish(request, RequestStatus.FINISHED, 'length')
        else:
            request.status = RequestStatus.DECODING

    def _sync(self):
        if self.backend.device.type == 'cuda':
            torch.cuda.synchronize(self.backend.device)

    def _run(self):
        try:
            with torch.inference_mode():
                if self.backend.device.type == 'cuda':
                    torch.cuda.set_device(self.backend.device)
                    torch.cuda.reset_peak_memory_stats(self.backend.device)
                self._ready.set()
                while True:
                    with self._condition:
                        while not self._requests and not self._stopping:
                            self._condition.wait()  # No idle CPU spin.
                        now = time.perf_counter()
                        for request in list(self._requests.values()):
                            if request.cancel_requested.is_set() or self._stopping:
                                self._finish(request, RequestStatus.CANCELLED, 'cancelled')
                            elif request.status == RequestStatus.WAITING and now - request.arrival_time >= self.config.queue_timeout_seconds:
                                self._finish(request, RequestStatus.ERROR, 'queue_timeout', 'Queue timeout')
                        if self._stopping:
                            break
                        request = self.scheduler.schedule()
                        if request is not None:
                            request.prefill_start_time = time.perf_counter()
                    if request is not None:
                        try:
                            logits, cache = self.backend.prefill(request.prompt_ids, logits_to_keep=1)
                            token_id = logits[:, -1].argmax(-1).item()
                            request.past_key_values = cache
                            del logits, cache
                            self._emit(request, token_id)
                        except Exception as exc:
                            logger.exception('Prefill failed')
                            self._finish(request, RequestStatus.ERROR, 'model_error', str(exc))
                    with self._condition:
                        batch = list(self.scheduler.running.values())
                    if batch:
                        try:
                            next_tokens = torch.tensor([[r.next_token_id] for r in batch], dtype=torch.long, device=self.backend.device)
                            started = time.perf_counter()
                            logits, caches = self.backend.decode_batch(next_tokens, [r.past_key_values for r in batch])
                            tokens = logits[:, -1].argmax(-1).tolist()  # Waits for CUDA completion.
                            elapsed = time.perf_counter() - started
                            with self._condition:
                                self._decode_batches.append({'request_ids': [r.request_id for r in batch], 'seconds': elapsed})
                                self._batch_histogram[len(batch)] += 1
                                self._counts['decode_forwards'] += 1
                            for r, cache, token in zip(batch, caches, tokens, strict=True):
                                r.past_key_values = cache
                                self._emit(r, token)
                            del logits, caches, cache
                        except Exception as exc:
                            logger.exception('Batched decode failed')
                            for r in batch:
                                self._finish(r, RequestStatus.ERROR, 'model_error', str(exc))
                    if self.backend.device.type == 'cuda':
                        with self._condition:
                            self._gpu_memory = {
                                'allocated_bytes': torch.cuda.memory_allocated(self.backend.device),
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.backend.device),
                                'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.backend.device),
                            }
        except Exception as exc:
            logger.exception('Inference worker failed')
            with self._condition:
                self._fatal_error = str(exc)
                self._accepting = False
                for r in list(self._requests.values()):
                    self._finish(r, RequestStatus.ERROR, 'worker_error', str(exc))
        finally:
            self._ready.set()
            with self._condition:
                self._accepting = False

    def snapshot(self):
        with self._condition:
            return {
                'worker_alive': self._thread is not None and self._thread.is_alive(),
                'accepting': self._accepting and not self._stopping,
                'fatal_error': self._fatal_error,
                'environment': self.environment,
                'serving_config': asdict(self.config),
                'waiting': len(self.scheduler.waiting), 'running': len(self.scheduler.running),
                'counters': dict(self._counts), 'rejections': dict(self._rejections),
                'kv_budget_bytes': self.kv.budget_bytes,
                'kv_reserved_bytes': self.kv.reserved_bytes,
                'kv_peak_reserved_bytes': self.kv.peak_reserved_bytes,
                'kv_bytes_per_token': self.kv.bytes_per_token,
                'kv_budget_scope': 'logical compact caches only; excludes padded copies, weights and activations',
                'decode_batch_sizes': [len(b['request_ids']) for b in self._decode_batches],
                'decode_batches': list(self._decode_batches),
                'decode_batch_histogram': dict(self._batch_histogram),
                'recent_requests': list(self._completed),
                'server_latency_window': latency_summary(self._completed),
                'gpu_memory': dict(self._gpu_memory),
            }
