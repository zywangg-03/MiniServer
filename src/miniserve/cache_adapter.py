"""Left-padded full-attention KV batching for Transformers 4.57.1.

Each request retains only its real history. A decode iteration copies those
histories into one temporary padded batch, then copies valid rows back out.
This is intentionally a correctness-first implementation, not PagedAttention:
padding and these extra copies consume memory beyond logical KV reservations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache


@dataclass
class PackedCache:
    cache: DynamicCache
    lengths: tuple[int, ...]
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor


def pack_kv_caches(caches: Sequence[Cache]) -> PackedCache:
    """Pack unpadded single-request caches for exactly one shared decode step.

    RoPE positions use each request's true history length. The physical cache
    append position uses the common padded length. The attention mask includes
    the incoming token and masks out every left-padding position.
    """
    if not caches or any(not isinstance(cache, DynamicCache) for cache in caches):
        raise ValueError("Expected one or more full-attention DynamicCache instances")
    lengths = tuple(int(cache.get_seq_length()) for cache in caches)
    if any(length < 1 for length in lengths):
        raise ValueError("Cannot batch an empty cache")
    layer_count = len(caches[0].layers)
    if any(len(cache.layers) != layer_count for cache in caches):
        raise ValueError("All caches must have the same number of layers")
    maximum = max(lengths)
    batch_size = len(caches)
    reference = caches[0].layers[0].keys
    if reference.ndim != 4:
        raise ValueError("KV tensors must have shape [1, heads, length, head_dim]")
    device, dtype = reference.device, reference.dtype
    heads, head_dim = reference.shape[1], reference.shape[3]
    packed = DynamicCache()
    for layer_index in range(layer_count):
        keys = reference.new_zeros((batch_size, heads, maximum, head_dim))
        values = torch.zeros_like(keys)
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            layer = cache.layers[layer_index]
            for tensor in (layer.keys, layer.values):
                if tensor.shape != (1, heads, length, head_dim):
                    raise ValueError("All cache layers must have compatible unpadded single-request shapes")
                if tensor.device != device or tensor.dtype != dtype:
                    raise ValueError("All caches must share device and dtype")
            keys[row, :, maximum - length:, :] = layer.keys[0]
            values[row, :, maximum - length:, :] = layer.values[0]
        packed.update(keys, values, layer_index)
    real_lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    offsets = maximum - real_lengths
    columns = torch.arange(maximum + 1, device=device)
    attention_mask = (columns.unsqueeze(0) >= offsets.unsqueeze(1)).long()
    return PackedCache(
        cache=packed,
        lengths=lengths,
        attention_mask=attention_mask,
        position_ids=real_lengths.unsqueeze(1),
        cache_position=torch.tensor([maximum], device=device, dtype=torch.long),
    )


def unpack_kv_caches(cache: Cache, lengths: Sequence[int]) -> list[DynamicCache]:
    """Keep true history plus the appended token, without retaining batch storage.

    ``lengths`` are the original per-request lengths, before the decode forward.
    clone() matters: storing a slice would keep the entire padded batch alive.
    """
    if not isinstance(cache, DynamicCache) or not lengths or any(length < 1 for length in lengths):
        raise ValueError("Expected a nonempty decoded DynamicCache and positive history lengths")
    maximum = max(lengths)
    batch_size = len(lengths)
    if cache.get_seq_length() != maximum + 1:
        raise ValueError("Unpack expects exactly one token appended to the padded cache")
    results = [DynamicCache() for _ in lengths]
    for layer_index, layer in enumerate(cache.layers):
        if layer.keys.shape[0] != batch_size or layer.keys.shape[-2] != maximum + 1:
            raise ValueError("Decoded cache shape does not match original batch lengths")
        for row, length in enumerate(lengths):
            start = maximum - length
            keys = layer.keys[row:row + 1, :, start:, :].clone()
            values = layer.values[row:row + 1, :, start:, :].clone()
            results[row].update(keys, values, layer_index)
    return results
