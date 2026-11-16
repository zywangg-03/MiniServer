"""Shared settings for local GPUs and SSH-accessible school GPU servers.

Both deployments use one explicitly selected device. On clusters, cuda:0 means
the first GPU visible through the administrator's CUDA_VISIBLE_DEVICES setting.
We never override that setting or silently fall back from CUDA to CPU.
"""

from dataclasses import dataclass


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Published model commit from the official repository, not the moving main ref.
# Run reports also save Hugging Face's fully resolved commit hash.
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    attn_implementation: str = "sdpa"
    seed: int = 0
    use_chat_template: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.model_name == DEFAULT_MODEL and self.revision is None:
            object.__setattr__(self, "revision", DEFAULT_REVISION)
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.attn_implementation not in {"eager", "sdpa"}:
            raise ValueError("attn_implementation must be eager or sdpa")
        if self.device != "cpu" and self.device != "cuda" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU correctness runs require dtype=float32")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

