"""Actual shared forwards must preserve per-request greedy generation."""

import pytest
import torch
from transformers import DynamicCache

from miniserve.cache_adapter import pack_kv_caches


def prefill_requests(backend, prompts):
    caches, generated = {}, {}
    for name, prompt in prompts.items():
        logits, caches[name] = backend.prefill(prompt, logits_to_keep=1)
        generated[name] = [int(logits[0, -1].argmax())]
    return caches, generated


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_unequal_histories_share_forward_and_match_hf(tiny_backend, attention):
    tiny_backend.model.set_attn_implementation(attention)
    prompts = {
        "short": torch.tensor([[4, 7]]),
        "long": torch.tensor([list(range(3, 30))]),
        "middle": torch.tensor([[9, 8, 6, 7, 5]]),
    }
    references = {name: tiny_backend.generate_hf_ids(ids, 8, eos_token_ids=[]) for name, ids in prompts.items()}
    caches, generated = prefill_requests(tiny_backend, prompts)
    shapes = []
    hook = tiny_backend.model.register_forward_pre_hook(
        lambda model, args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    try:
        for _ in range(7):
            tokens = torch.tensor([[generated[name][-1]] for name in prompts])
            logits, next_caches = tiny_backend.decode_batch(tokens, list(caches.values()))
            for row, name in enumerate(prompts):
                generated[name].append(int(logits[row, -1].argmax()))
                caches[name] = next_caches[row]
    finally:
        hook.remove()
    assert shapes == [(3, 1)] * 7  # One shared forward each iteration, never a per-request loop.
    assert generated == references
    for name, prompt in prompts.items():
        assert caches[name].get_seq_length() == prompt.shape[1] + 7


def test_join_exit_and_reorder_preserve_cache_identity(tiny_backend):
    prompts = {
        "a": torch.tensor([[3, 4, 5]]),
        "b": torch.tensor([[4, 5, 8, 9, 7, 6]]),
        "c": torch.tensor([[8]]),
        "d": torch.tensor([list(range(4, 24))]),
    }
    references = {name: tiny_backend.generate_hf_ids(ids, 8, eos_token_ids=[]) for name, ids in prompts.items()}
    caches, generated = prefill_requests(tiny_backend, {name: prompts[name] for name in ("a", "b")})
    memberships = [("a", "b"), ("c", "b", "a"), ("a", "c"), ("d", "c", "a"), ("d",)]
    for names in memberships:
        for name in names:
            if name not in generated:
                added_cache, added_tokens = prefill_requests(tiny_backend, {name: prompts[name]})
                caches.update(added_cache)
                generated.update(added_tokens)
        logits, next_caches = tiny_backend.decode_batch(
            torch.tensor([[generated[name][-1]] for name in names]),
            [caches[name] for name in names],
        )
        for row, name in enumerate(names):
            generated[name].append(int(logits[row, -1].argmax()))
            caches[name] = next_caches[row]
            assert generated[name] == references[name][:len(generated[name])]
            assert caches[name].get_seq_length() == prompts[name].shape[1] + len(generated[name]) - 1


def test_padding_positions_and_compact_storage(tiny_backend):
    prompts = {"a": torch.tensor([[3, 4]]), "b": torch.tensor([[4, 5, 6, 7, 8]])}
    caches, generated = prefill_requests(tiny_backend, prompts)
    packed = pack_kv_caches(list(caches.values()))
    assert packed.attention_mask.tolist() == [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    assert packed.position_ids.tolist() == [[2], [5]]
    assert packed.cache_position.tolist() == [5]
    snapshots = [cache.layers[0].keys.clone() for cache in caches.values()]
    _, results = tiny_backend.decode_batch(torch.tensor([[generated[n][-1]] for n in prompts]), list(caches.values()))
    for source, expected in zip(caches.values(), snapshots):
        torch.testing.assert_close(source.layers[0].keys, expected)
    for result, length in zip(results, (3, 6)):
        assert result.get_seq_length() == length
        for layer in result.layers:
            for tensor in (layer.keys, layer.values):
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                assert tensor.storage_offset() == 0
    assert results[0].layers[0].keys.data_ptr() != results[1].layers[0].keys.data_ptr()


def test_invalid_batches_fail_before_forward(tiny_backend):
    _, cache = tiny_backend.prefill(torch.tensor([[3, 4]]))
    for tokens, caches in (
        (torch.tensor([[3]]), []),
        (torch.tensor([[3], [4]]), [cache]),
        (torch.tensor([[3.0]]), [cache]),
        (torch.tensor([[97]]), [cache]),
        (torch.tensor([[3]]), [DynamicCache()]),
    ):
        with pytest.raises(ValueError):
            tiny_backend.decode_batch(tokens, caches)


@pytest.mark.model
def test_pretrained_variable_length_batched_greedy(pretrained_backend):
    backend = pretrained_backend
    prompts = {
        "english": backend.prepare_prompt("What is a KV cache?"),
        "chinese": backend.prepare_prompt("请用两句话解释自回归生成与注意力缓存之间的关系。"),
        "long": backend.prepare_prompt("Explain how an inference server handles users whose prompts have different lengths, and why batching can help."),
    }
    references = {name: backend.generate_hf_ids(ids, 12, eos_token_ids=[]) for name, ids in prompts.items()}
    caches, generated = prefill_requests(backend, prompts)
    for _ in range(11):
        tokens = torch.tensor([[generated[name][-1]] for name in prompts], device=backend.device)
        logits, next_caches = backend.decode_batch(tokens, list(caches.values()))
        for row, name in enumerate(prompts):
            generated[name].append(int(logits[row, -1].argmax()))
            caches[name] = next_caches[row]
    assert generated == references
