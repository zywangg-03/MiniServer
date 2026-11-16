"""Phase 1 correctness: greedy IDs, cache boundaries, EOS, and input limits."""

from __future__ import annotations

import pytest
import torch
from transformers import GenerationConfig

from miniserve.backends.hf_backend import HFBackend


def hf_reference(
    backend: HFBackend,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list[int] | None = None,
) -> list[int]:
    """Independent oracle, with all non-greedy checkpoint defaults cleared."""
    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_ids or None,
        pad_token_id=0,
        bos_token_id=1,
    )
    with torch.inference_mode():
        output = backend.model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            generation_config=config,
            use_model_defaults=False,
            # Even use_model_defaults=False allows special-token fallback;
            # the explicit kwarg is applied after that fallback in 4.57.1.
            eos_token_id=eos_token_ids or None,
        )
    return output[0, prompt_ids.shape[1] :].tolist()


@pytest.mark.parametrize("prompt", [[4], [5, 8, 13], list(range(3, 35))])
@pytest.mark.parametrize("max_new_tokens", [1, 6])
def test_manual_greedy_matches_independent_hf(tiny_backend, prompt, max_new_tokens):
    prompt_ids = torch.tensor([prompt], dtype=torch.long)
    expected = hf_reference(tiny_backend, prompt_ids, max_new_tokens)
    assert tiny_backend.generate_ids(prompt_ids, max_new_tokens, eos_token_ids=[]) == expected
    assert tiny_backend.generate_hf_ids(prompt_ids, max_new_tokens, eos_token_ids=[]) == expected
    assert len(expected) == max_new_tokens


def test_prefill_and_decode_cache_boundary(tiny_backend):
    prompt = torch.tensor([[6, 13, 22, 7]], dtype=torch.long)
    logits, cache = tiny_backend.prefill(prompt)
    assert logits.shape == (1, 4, 97)
    assert cache.get_seq_length() == 4
    assert not logits.requires_grad
    assert not tiny_backend.model.training

    first_token = logits[:, -1, :].argmax(dim=-1).reshape(1, 1)
    # Selecting/emitting the first token does not write it to the KV cache.
    assert cache.get_seq_length() == 4
    next_logits, next_cache = tiny_backend.decode(first_token, cache)
    assert next_cache.get_seq_length() == 5
    assert next_logits.shape == (1, 1, 97)
    assert not next_logits.requires_grad

    whole_sequence = torch.cat([prompt, first_token], dim=1)
    with torch.inference_mode():
        full_logits = tiny_backend.model(
            input_ids=whole_sequence,
            attention_mask=torch.ones_like(whole_sequence),
            use_cache=False,
        ).logits
    torch.testing.assert_close(next_logits[:, -1], full_logits[:, -1], rtol=1e-4, atol=1e-5)


def test_interleaved_requests_keep_independent_caches(tiny_backend):
    prompt_a = torch.tensor([[4, 8, 15]], dtype=torch.long)
    prompt_b = torch.tensor([[7, 6, 5, 4, 3, 9]], dtype=torch.long)
    reference_a = hf_reference(tiny_backend, prompt_a, 2)
    reference_b = hf_reference(tiny_backend, prompt_b, 2)

    logits_a, cache_a = tiny_backend.prefill(prompt_a)
    logits_b, cache_b = tiny_backend.prefill(prompt_b)
    first_a = logits_a[:, -1].argmax(dim=-1).reshape(1, 1)
    first_b = logits_b[:, -1].argmax(dim=-1).reshape(1, 1)
    second_a, cache_a = tiny_backend.decode(first_a, cache_a)
    assert cache_a.get_seq_length() == 4
    assert cache_b.get_seq_length() == 6
    second_b, cache_b = tiny_backend.decode(first_b, cache_b)
    assert [first_a.item(), second_a[:, -1].argmax().item()] == reference_a
    assert [first_b.item(), second_b[:, -1].argmax().item()] == reference_b
    assert cache_a.get_seq_length() == 4
    assert cache_b.get_seq_length() == 7


@pytest.mark.parametrize("method", ["generate_ids", "generate_hf_ids"])
def test_eos_first_token_is_included_and_stops(tiny_backend, method):
    prompt = torch.tensor([[3, 7, 11]], dtype=torch.long)
    first = hf_reference(tiny_backend, prompt, 1)[0]
    result = getattr(tiny_backend, method)(prompt, 8, eos_token_ids=[first])
    assert result == [first]


@pytest.mark.parametrize("method", ["generate_ids", "generate_hf_ids"])
def test_eos_after_first_token_stops_at_first_occurrence(tiny_backend, method):
    prompt = torch.tensor([[3, 7, 11]], dtype=torch.long)
    tokens = hf_reference(tiny_backend, prompt, 12)
    later = next((i for i, token in enumerate(tokens) if token != tokens[0]), None)
    assert later is not None, "Tiny model fixture must exercise a later, distinct EOS."
    eos = tokens[later]
    expected = tokens[: later + 1]
    assert getattr(tiny_backend, method)(prompt, 12, eos_token_ids=[eos]) == expected


@pytest.mark.parametrize("method", ["generate_ids", "generate_hf_ids"])
def test_generation_config_multiple_eos_ids_are_respected(tiny_backend, method):
    prompt = torch.tensor([[3, 7, 11]], dtype=torch.long)
    first = hf_reference(tiny_backend, prompt, 1)[0]
    different = (first + 1) % tiny_backend.model.config.vocab_size
    tiny_backend.model.config.eos_token_id = different
    tiny_backend.model.generation_config.eos_token_id = [different, first]
    assert getattr(tiny_backend, method)(prompt, 8) == [first]
    assert len(getattr(tiny_backend, method)(prompt, 8, eos_token_ids=[])) == 8


def test_checkpoint_sampling_and_repetition_settings_are_neutralized(tiny_backend):
    prompt = torch.tensor([[3, 7, 11]], dtype=torch.long)
    expected = hf_reference(tiny_backend, prompt, 8)
    checkpoint_config = tiny_backend.model.generation_config
    checkpoint_config.do_sample = True
    checkpoint_config.temperature = 0.7
    checkpoint_config.top_p = 0.8
    checkpoint_config.top_k = 20
    checkpoint_config.repetition_penalty = 1.8
    checkpoint_config.forced_eos_token_id = (expected[-1] + 1) % 97
    assert tiny_backend.generate_hf_ids(prompt, 8, eos_token_ids=[]) == expected
    assert tiny_backend.generate_ids(prompt, 8, eos_token_ids=[]) == expected
    assert checkpoint_config.do_sample is True  # Do not mutate checkpoint metadata.


@pytest.mark.parametrize("max_new_tokens", [1, 5])
def test_no_forward_after_final_token(tiny_backend, max_new_tokens):
    input_lengths = []

    def record_forward(_model, _args, kwargs):
        input_lengths.append(kwargs["input_ids"].shape[1])

    handle = tiny_backend.model.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        tokens = tiny_backend.generate_ids(torch.tensor([[3, 7, 11]]), max_new_tokens, eos_token_ids=[])
    finally:
        handle.remove()
    assert len(tokens) == max_new_tokens
    assert input_lengths == [3] + [1] * (max_new_tokens - 1)


def test_no_decode_after_first_token_eos(tiny_backend):
    prompt = torch.tensor([[3, 7, 11]])
    first = hf_reference(tiny_backend, prompt, 1)[0]
    input_lengths = []

    def record_forward(_model, _args, kwargs):
        input_lengths.append(kwargs["input_ids"].shape[1])

    handle = tiny_backend.model.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        assert tiny_backend.generate_ids(prompt, 8, eos_token_ids=[first]) == [first]
    finally:
        handle.remove()
    assert input_lengths == [3]


def test_prepare_prompt_uses_single_template_and_raw_path(tiny_backend):
    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(("chat", messages, kwargs))
            return torch.tensor([[1, 3, 4, 5]])

        def __call__(self, text, **kwargs):
            self.calls.append(("raw", text, kwargs))
            return {"input_ids": torch.tensor([[3, 4]])}

    tokenizer = RecordingTokenizer()
    tiny_backend.tokenizer = tokenizer
    text = "请解释 KV cache。"
    assert tiny_backend.prepare_prompt(text).tolist() == [[1, 3, 4, 5]]
    assert tiny_backend.prepare_prompt(text, use_chat_template=False).tolist() == [[3, 4]]
    assert tokenizer.calls == [
        (
            "chat",
            [{"role": "user", "content": text}],
            {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"},
        ),
        ("raw", text, {"add_special_tokens": False, "return_tensors": "pt"}),
    ]
    for invalid in ("", " \n ", None):
        with pytest.raises(ValueError):
            tiny_backend.prepare_prompt(invalid)


@pytest.mark.parametrize("eos", [[-1], [97], [True], [1.5]])
def test_invalid_eos_ids_rejected(tiny_backend, eos):
    with pytest.raises(ValueError):
        tiny_backend.generate_ids(torch.tensor([[3, 7]]), 2, eos_token_ids=eos)


@pytest.mark.parametrize("prompt", [torch.tensor([[-1]]), torch.tensor([[97]]), torch.tensor([[3.0]])])
def test_invalid_token_ids_rejected(tiny_backend, prompt):
    with pytest.raises(ValueError):
        tiny_backend.generate_ids(prompt, 2)


@pytest.mark.parametrize("method", ["generate_ids", "generate_hf_ids"])
@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_generation_length_rejected(tiny_backend, method, limit):
    with pytest.raises((ValueError, TypeError)):
        getattr(tiny_backend, method)(torch.tensor([[4, 5]]), limit)


@pytest.mark.parametrize(
    "prompt",
    [torch.empty((1, 0), dtype=torch.long), torch.tensor([[4], [5]]), torch.tensor([4, 5])],
)
def test_invalid_prompt_shape_rejected(tiny_backend, prompt):
    with pytest.raises((ValueError, TypeError)):
        tiny_backend.generate_ids(prompt, 2)


@pytest.mark.parametrize("method", ["generate_ids", "generate_hf_ids"])
def test_context_overflow_rejected(tiny_backend, method):
    prompt = torch.full((1, 127), 4, dtype=torch.long)
    with pytest.raises(ValueError):
        getattr(tiny_backend, method)(prompt, 2)


@pytest.mark.model
@pytest.mark.parametrize(
    "prompt",
    [
        "What is the capital of France? Answer briefly.",
        "请用一句话解释什么是 KV cache。",
        "Explain how prefill and autoregressive decoding differ, including why the first output token is not yet in the cache.",
    ],
)
@pytest.mark.parametrize("max_new_tokens", [1, 16])
def test_pretrained_qwen_greedy_ids(pretrained_backend, prompt, max_new_tokens):
    prompt_ids = pretrained_backend.prepare_prompt(prompt)
    expected = pretrained_backend.generate_hf_ids(prompt_ids, max_new_tokens)
    assert pretrained_backend.generate_ids(prompt_ids, max_new_tokens) == expected


@pytest.mark.model
def test_pretrained_raw_prompt(pretrained_backend):
    ids = pretrained_backend.prepare_prompt("The capital of France is", use_chat_template=False)
    assert pretrained_backend.generate_ids(ids, 8) == pretrained_backend.generate_hf_ids(ids, 8)
