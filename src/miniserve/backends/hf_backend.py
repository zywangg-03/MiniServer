"""HF execution, shared by local and remote single-GPU runs.

Single-request generation owns a fresh DynamicCache; decode_batch additionally
supports genuine shared forwards for different-length request histories. An
output token enters its cache only on the next decode call.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.cache_utils import Cache

from miniserve.config import ModelConfig
from miniserve.cache_adapter import pack_kv_caches, unpack_kv_caches


class HFBackend:
    """Manual prefill/decode and a neutral HF greedy reference implementation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config: ModelConfig | None = None
        self.device = next(model.parameters()).device
        self.context_limit = int(model.config.max_position_embeddings)

    @classmethod
    def from_config(cls, config: ModelConfig) -> HFBackend:
        device = torch.device(config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Run on the GPU server with a CUDA PyTorch "
                    "build, or explicitly choose --device cpu --dtype float32 for "
                    "correctness checks."
                )
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {index} is not visible to this process")
            # Respect CUDA_VISIBLE_DEVICES on both SSH servers and scheduled jobs.
            torch.cuda.set_device(index)
            if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("Selected GPU does not support bfloat16")
        torch.manual_seed(config.seed)
        load_options = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, **load_options)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=getattr(torch, config.dtype),
            attn_implementation=config.attn_implementation,
            **load_options,
        ).to(device)
        if model.config.model_type != "qwen2":
            raise ValueError("Phase 1 supports Qwen2/Qwen2.5 models (model_type=qwen2)")
        if getattr(model.config, "use_sliding_window", False):
            raise ValueError("Phase 1 requires full attention, without sliding-window cache")
        backend = cls(model, tokenizer)
        backend.config = config
        return backend

    def prepare_prompt(self, prompt: str, use_chat_template: bool = True) -> torch.Tensor:
        """Use exactly one user message and the tokenizer's pinned chat template."""
        if self.tokenizer is None:
            raise RuntimeError("A tokenizer is required for text prompts")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if use_chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
        return self._validate_input(input_ids)

    def _validate_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Phase 1 expects one unpadded nonempty prompt: shape [1, L]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        if input_ids.shape[1] > self.context_limit:
            raise ValueError("Prompt exceeds model context length")
        if bool(((input_ids < 0) | (input_ids >= self.model.config.vocab_size)).any()):
            raise ValueError("input_ids contain token IDs outside the model vocabulary")
        return input_ids.to(self.device)

    def _validate_generation(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        input_ids = self._validate_input(input_ids)
        if input_ids.shape[1] + max_new_tokens > self.context_limit:
            raise ValueError("prompt length + max_new_tokens exceeds model context length")
        return input_ids

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, *, logits_to_keep: int = 0) -> tuple[torch.Tensor, Cache]:
        """Cache the prompt and return its logits; no output token is cached yet."""
        input_ids = self._validate_input(input_ids)
        positions = torch.arange(input_ids.shape[1], device=self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=DynamicCache(config=self.model.config),
            use_cache=True,
            return_dict=True,
            logits_to_keep=logits_to_keep,
        )
        return outputs.logits, outputs.past_key_values

    @torch.inference_mode()
    def decode(self, next_token: torch.Tensor, past_key_values: Cache) -> tuple[torch.Tensor, Cache]:
        """Append one previously emitted token using full history attention.

        The cache is mutated in place by Transformers. The caller owns it and
        must not share it between independent requests.
        """
        if next_token.shape != (1, 1) or next_token.dtype != torch.long:
            raise ValueError("next_token must be a torch.long tensor with shape [1, 1]")
        if not isinstance(past_key_values, Cache):
            raise ValueError("past_key_values must be a Transformers Cache")
        past_length = past_key_values.get_seq_length()
        if past_length < 1 or past_length >= self.context_limit:
            raise ValueError("Cache must be nonempty and leave room for the next token")
        position = torch.tensor([past_length], device=self.device, dtype=torch.long)
        outputs = self.model(
            input_ids=next_token.to(self.device),
            attention_mask=torch.ones((1, past_length + 1), device=self.device, dtype=torch.long),
            position_ids=position.unsqueeze(0),
            cache_position=position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, outputs.past_key_values

    def _eos_ids(self, eos_token_ids: Sequence[int] | None) -> list[int]:
        if eos_token_ids is None:
            raw = self.model.generation_config.eos_token_id
            eos_token_ids = [] if raw is None else ([raw] if isinstance(raw, int) else raw)
        values = list(eos_token_ids)
        if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < self.model.config.vocab_size for x in values):
            raise ValueError("EOS IDs must be valid integer vocabulary IDs")
        return values

    @torch.inference_mode()
    def decode_batch(
        self, next_tokens: torch.Tensor, caches: list[Cache]
    ) -> tuple[torch.Tensor, list[Cache]]:
        """Run ONE forward for the entire active decode batch.

        Inputs are each request's previously emitted token. Outputs are logits
        predicting its next output token and compact caches containing the input
        token. Multiple-request inputs remain unmodified. A single-request call
        uses decode() directly and therefore updates that cache in place.
        """
        if not caches or not isinstance(next_tokens, torch.Tensor):
            raise ValueError("A nonempty cache list and next-token tensor are required")
        if next_tokens.shape != (len(caches), 1) or next_tokens.dtype != torch.long:
            raise ValueError("next_tokens must be torch.long with shape [batch_size, 1]")
        if any(not isinstance(cache, Cache) for cache in caches):
            raise ValueError("Every request must have a Transformers Cache")
        if any(not 0 < cache.get_seq_length() < self.context_limit for cache in caches):
            raise ValueError("Each cache must be nonempty and leave room for one token")
        if bool(((next_tokens < 0) | (next_tokens >= self.model.config.vocab_size)).any()):
            raise ValueError("next_tokens contain IDs outside the vocabulary")
        if len(caches) == 1:
            logits, cache = self.decode(next_tokens, caches[0])
            return logits, [cache]
        packed = pack_kv_caches(caches)
        if packed.attention_mask.device != self.device:
            raise ValueError("Caches must be on the model device")
        output = self.model(
            input_ids=next_tokens.to(self.device),
            past_key_values=packed.cache,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            cache_position=packed.cache_position,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        return output.logits, unpack_kv_caches(output.past_key_values, packed.lengths)

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """Return new token IDs, including EOS when emitted; no logits processors."""
        input_ids = self._validate_generation(input_ids, max_new_tokens)
        eos_ids = set(self._eos_ids(eos_token_ids))
        # Like HF generate(), project only the last prompt position to vocabulary
        # logits. The cache still includes every prompt token.
        logits, cache = self.prefill(input_ids, logits_to_keep=1)
        generated: list[int] = []
        for step in range(max_new_tokens):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id in eos_ids or step + 1 == max_new_tokens:
                break
            # Release prefill logits before decode, which only needs next_token.
            del logits
            logits, cache = self.decode(next_token, cache)
        return generated

    @torch.inference_mode()
    def generate_hf_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """HF reference with a fresh configuration, never model sampling defaults.

        Qwen's published generation_config has repetition_penalty=1.1. Merely
        setting do_sample=False would therefore NOT match raw-logit argmax.
        """
        input_ids = self._validate_generation(input_ids, max_new_tokens)
        eos_ids = self._eos_ids(eos_token_ids)
        original = self.model.generation_config
        pad_id = original.pad_token_id
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else 0
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            bos_token_id=original.bos_token_id,
            eos_token_id=eos_ids or None,
            pad_token_id=pad_id,
            use_cache=True,
            # An explicit dynamic cache prevents persistent/static cache reuse.
            cache_implementation="dynamic",
        )
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config,
            use_model_defaults=False,
            # Prevent fallback to a model EOS when the caller explicitly disables it.
            eos_token_id=eos_ids or None,
        )
        return output[0, input_ids.shape[1]:].tolist()

    def environment_info(self) -> dict[str, Any]:
        """Record actual hardware, software and resolved model revision per run."""
        template = getattr(self.tokenizer, 'chat_template', None)
        info: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "model_name": self.model.config._name_or_path,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "attention_implementation": self.model.config._attn_implementation,
            "context_limit": self.context_limit,
            "eos_token_ids": self._eos_ids(None),
            "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest() if template is not None else None,
            "requested_config": asdict(self.config) if self.config is not None else None,
            "gpu": None,
        }
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            info["gpu"] = {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(self.device)),
            }
        return info
