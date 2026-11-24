"""Logical compact KV reservations; excludes padding, copying and activations."""


class KVManager:
    def __init__(self, budget_bytes: int, bytes_per_token: int):
        if budget_bytes <= 0 or bytes_per_token <= 0:
            raise ValueError('KV sizes must be positive')
        self.budget_bytes = budget_bytes
        self.bytes_per_token = bytes_per_token
        self.reservations: dict[str, int] = {}
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0

    def required(self, prompt_tokens: int, output_tokens: int):
        return self.bytes_per_token * (prompt_tokens + output_tokens)

    def reserve(self, request_id: str, amount: int) -> bool:
        if request_id in self.reservations:
            raise ValueError('Request already reserved')
        if amount <= 0:
            raise ValueError('Reservation must be positive')
        if self.reserved_bytes + amount > self.budget_bytes:
            return False
        self.reservations[request_id] = amount
        self.reserved_bytes += amount
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        return True

    def release(self, request_id: str):
        self.reserved_bytes -= self.reservations.pop(request_id, 0)


def model_kv_bytes_per_token(model):
    c = model.config
    head_dim = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
    return 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * next(model.parameters()).element_size()
