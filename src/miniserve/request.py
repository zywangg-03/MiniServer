"""Bounded, per-request state. Only the inference worker changes execution state."""
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import time
import uuid
from typing import Any


class RequestStatus(str, Enum):
    WAITING = 'waiting'
    PREFILL = 'prefill'
    DECODING = 'decoding'
    FINISHED = 'finished'
    CANCELLED = 'cancelled'
    ERROR = 'error'


@dataclass
class RequestState:
    prompt_ids: Any
    max_new_tokens: int
    eos_token_ids: set[int]
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    arrival_time: float = field(default_factory=time.perf_counter)
    status: RequestStatus = RequestStatus.WAITING
    generated_ids: list[int] = field(default_factory=list)
    past_key_values: Any = None
    next_token_id: int | None = None
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    finish_reason: str | None = None
    text: str = ''
    emitted_text: str = ''
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    output_queue: queue.Queue = field(init=False)

    def __post_init__(self):
        # One event per token plus exactly one terminal event, even if never read.
        self.output_queue = queue.Queue(maxsize=self.max_new_tokens + 1)

    def metrics(self):
        return {
            'arrival_time': self.arrival_time,
            'prefill_start_time': self.prefill_start_time,
            'first_token_time': self.first_token_time,
            'finish_time': self.finish_time,
            'queue_seconds': None if self.prefill_start_time is None else self.prefill_start_time - self.arrival_time,
            'ttft_seconds': None if self.first_token_time is None else self.first_token_time - self.arrival_time,
            'tpot_seconds': (self.last_token_time - self.first_token_time) / (len(self.generated_ids) - 1) if len(self.generated_ids) > 1 else None,
            'latency_seconds': None if self.finish_time is None else self.finish_time - self.arrival_time,
        }

    def result(self):
        if not self.done.is_set():
            raise RuntimeError('Request is still running')
        return {
            'request_id': self.request_id, 'status': self.status.value,
            'finish_reason': self.finish_reason, 'text': self.text,
            'output_ids': list(self.generated_ids), 'output_tokens': len(self.generated_ids),
            'metrics': self.metrics(), 'error': self.error,
        }
