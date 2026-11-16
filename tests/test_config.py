"""The same model settings serve local and SSH GPU execution."""

import pytest

from miniserve.config import ModelConfig


def test_default_model_and_gpu_settings():
    config = ModelConfig()
    assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.device == "cuda:0"
    assert config.dtype == "float16"
    assert config.attn_implementation == "sdpa"
    assert config.use_chat_template is True
    assert config.local_files_only is False


def test_cpu_correctness_mode_is_explicit():
    config = ModelConfig(device="cpu", dtype="float32", seed=7, local_files_only=True)
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.seed == 7
    assert config.local_files_only is True


def test_remote_model_path_and_revision_are_configurable():
    config = ModelConfig(model_name="/scratch/models/qwen", revision=None)
    assert config.model_name == "/scratch/models/qwen"
    assert config.revision is None

    pinned = ModelConfig(revision="0123456789abcdef0123456789abcdef01234567")
    assert pinned.revision == "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": " "},
        {"dtype": "int8"},
        {"device": "mps"},
        {"device": "cuda:-1"},
        {"device": "cpu", "dtype": "float16"},
        {"attn_implementation": "flash_attention_2"},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_unsupported_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
