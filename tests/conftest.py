"""Offline tests use a small real Qwen2 model; model downloads are opt-in."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from miniserve.backends.hf_backend import HFBackend
from miniserve.config import ModelConfig


@pytest.fixture
def tiny_backend() -> HFBackend:
    # No pretrained weights or tokenizer are downloaded by this fixture.
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    return HFBackend(Qwen2ForCausalLM(config))


@pytest.fixture(scope="session")
def pretrained_backend() -> HFBackend:
    if os.environ.get("MINISERVE_RUN_MODEL_TESTS") != "1":
        pytest.skip("Set MINISERVE_RUN_MODEL_TESTS=1 to test pretrained Qwen weights.")
    kwargs = {
        "device": os.environ.get("MINISERVE_TEST_DEVICE", "cuda:0"),
        "dtype": os.environ.get("MINISERVE_TEST_DTYPE", "float16"),
        "local_files_only": os.environ.get("HF_HUB_OFFLINE") == "1",
    }
    if os.environ.get("MINISERVE_MODEL_PATH"):
        kwargs["model_name"] = os.environ["MINISERVE_MODEL_PATH"]
    if os.environ.get("MINISERVE_MODEL_REVISION"):
        kwargs["revision"] = os.environ["MINISERVE_MODEL_REVISION"]
    # An explicitly requested integration run must fail if CUDA/loading fails.
    return HFBackend.from_config(ModelConfig(**kwargs))
