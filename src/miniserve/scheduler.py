"""FIFO, no preemption. Called under the engine condition lock."""
from collections import deque
from miniserve.request import RequestState, RequestStatus


class Scheduler:
    def __init__(self, config, kv_manager):
        self.config = config
        self.kv = kv_manager
        self.waiting: deque[RequestState] = deque()
        self.running: dict[str, RequestState] = {}

    def add_request(self, request):
        if len(self.waiting) >= self.config.max_queue_size:
            raise OverflowError('waiting_queue_full')
        self.waiting.append(request)

    def schedule(self):
        # At most one prefill per iteration bounds prefill interference with decode.
        if not self.waiting or len(self.running) >= self.config.max_batch_size:
            return None
        request = self.waiting[0]
        amount = self.kv.required(request.prompt_ids.shape[1], request.max_new_tokens)
        if not self.kv.reserve(request.request_id, amount):
            return None  # Strict FIFO: no bypass of a large head request.
        self.waiting.popleft()
        self.running[request.request_id] = request
        request.status = RequestStatus.PREFILL
        return request

    def finish_request(self, request_id):
        self.running.pop(request_id, None)
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)
        self.kv.release(request_id)
