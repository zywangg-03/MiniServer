# Phase 1：学校 SSH GPU 服务器手动验证

这份流程验证基础推理，不启动 HTTP 服务。推理进程仅使用一张 GPU；服务器有多张卡时选择其中一张即可。本地单卡和远端单卡使用同一份代码。

## 1. 同步源码

在本地终端进入项目目录，替换本地路径、学校账号、主机和远端路径后执行：

```bash
cd /path/to/local/miniserve
ssh USER@GPU_SERVER 'mkdir -p /path/to/miniserve'
rsync -av \
  --exclude='.venv*' --exclude='.bootstrap/' --exclude='.python/' \
  --exclude='.uv-cache/' --exclude='.cache/' --exclude='.git/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='*.egg-info/' --exclude='results/' --exclude='.DS_Store' \
  ./ USER@GPU_SERVER:/path/to/miniserve/
ssh USER@GPU_SERVER
```

不要上传 macOS 的 Python 环境；以下命令都在远端执行。若学校提供了已装好 CUDA PyTorch 的环境，可以激活该环境后跳过虚拟环境创建步骤，并在运行脚本时设置 `MINISERVE_PYTHON`。

## 2. 准备远端 Python 与 CUDA

```bash
cd /path/to/miniserve
nvidia-smi
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择适配远端 Linux、NVIDIA 驱动的 CUDA wheel，执行页面给出的安装命令。这里不固定 CUDA wheel：不同学校服务器的驱动版本可能不同。之后执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python - <<'PY'
import torch
import transformers
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "当前环境无法使用 CUDA，请先修复 GPU 环境"
print("Visible GPUs:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("GPU 0 memory GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
assert transformers.__version__ == "4.57.1"
PY
```

预期 CUDA available 为 `True`。`cuda:0` 是当前进程可见的第一张卡；如果管理员设置了 `CUDA_VISIBLE_DEVICES`，不要直接把物理卡号当作进程内编号。

## 3. 离线测试与下载模型

```bash
python -m pytest -q
```

预期离线测试全部通过，7 项正式模型测试被跳过；这些跳过将在下一步单独执行。离线测试只使用随机初始化的微型 Qwen2，不下载正式权重。

把缓存设置到有足够空间的项目目录或学校数据盘。默认模型权重约 1 GB，环境、缓存和输出还需要额外空间。

```bash
export HF_HOME="$PWD/.cache/huggingface"
python - <<'PY'
from huggingface_hub import snapshot_download
from miniserve.config import ModelConfig
c = ModelConfig()
print("Downloading:", c.model_name, c.revision)
snapshot_download(
    c.model_name,
    revision=c.revision,
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
PY
```

默认 revision 固定为 `7ae557604adf67be50417f59c2c2f167def9a775`。下载完成后可离线运行：

```bash
export HF_HUB_OFFLINE=1
MINISERVE_RUN_MODEL_TESTS=1 \
MINISERVE_TEST_DEVICE=cuda:0 \
MINISERVE_TEST_DTYPE=float16 \
python -m pytest -q -m model
```

预期 **7 passed**。测试包含英文/中文、不同长度的 chat prompt，以及 raw prompt；比较的是输出 token IDs，而非仅比较文本。显式开启正式模型测试后，CUDA 不可用或加载失败会报错，不会当作通过。

## 4. 一次运行完整 Phase 1 验证与测量

```bash
bash scripts/run_cluster.sh --local-files-only
```

如果不使用项目 `.venv`，可以指定已激活环境的解释器：

```bash
MINISERVE_PYTHON="$(command -v python)" bash scripts/run_cluster.sh --local-files-only
```

如需选择其他可见 GPU，使用 `MINISERVE_DEVICE=cuda:1`。脚本依次运行正确性比较、HF 串行 baseline、手动 prefill/decode，每项都加载同一版本模型。默认 3 个 prompt、最多 64 个输出 tokens、每个 prompt 1 次 warm-up、5 次正式重复；遇到 EOS 可以提前结束。

脚本输出 `Phase 1 results: ...`，结果在独立的 `results/phase1-remote-XXXXXX/` 中，包含：

| 文件 | 检查内容 |
| --- | --- |
| `correctness.json` | `passed: true`，6 个比较全部 `matches: true` |
| `hf_baseline.json` | `kind: hf_sequential`，15 条正式请求记录 |
| `manual_generate.json` | `kind: manual_sequential`，15 条正式请求记录 |

若只想先检查是否能够跑通，缩短输出及重复次数：

```bash
MINISERVE_MAX_NEW_TOKENS=8 MINISERVE_REPETITIONS=1 \
bash scripts/run_cluster.sh --local-files-only
```

这个快速检查不能代替默认 5 次重复的正式测量。脚本在正确性不一致时会停止；先查看 `correctness.json` 的 `first_difference_index` 和两组 token IDs。

## 5. 检查原始结果

替换下方目录为脚本实际打印的路径：

```bash
python - results/phase1-remote-XXXXXX <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
check = json.loads((root / "correctness.json").read_text())
assert check["passed"], "存在 token ID 不一致，请先排查正确性"
print("Correctness comparisons:", check["comparisons"])
reports = {}
for name in ("hf_baseline", "manual_generate"):
    report = json.loads((root / f"{name}.json").read_text())
    reports[name] = report
    env, summary = report["environment"], report["summary"]
    assert env["device"].startswith("cuda")
    assert env["dtype"] == "torch.float16"
    assert env["gpu"] is not None
    assert summary["cuda_peak_allocated_bytes"] is not None
    print(name)
    print("  GPU:", env["gpu"]["name"])
    print("  Revision:", env["model_revision"])
    print("  Requests:", summary["requests"])
    print("  Output tokens:", summary["total_output_tokens"])
    print("  Tokens/s:", summary["sequential_generated_tokens_per_second"])
    print("  P95 generation seconds:", summary["p95_generation_seconds"])
    print("  Peak allocated MiB:", summary["cuda_peak_allocated_bytes"] / 2**20)

hf, manual = reports["hf_baseline"], reports["manual_generate"]
for key in ("model_revision", "dtype", "device", "gpu", "attention_implementation", "chat_template_sha256"):
    assert hf["environment"][key] == manual["environment"][key], key
for left, right in zip(hf["records"], manual["records"], strict=True):
    assert left["input_ids"] == right["input_ids"]
    assert left["output_ids"] == right["output_ids"]
print("两份基线的输入/输出 token IDs 一致。")
PY
```

`generation_seconds` 使用 CUDA 同步，包含推理和把输出 IDs 取回 CPU 的时间，不含模型加载、分词和文本解码。吞吐量的分母是正式请求推理时间之和。显存峰值包含模型权重，reserved memory 可能含复用的分配；这些都不是仅 KV cache 的大小。当前没有 HTTP 客户端 TTFT/TPOT，也没有 continuous batching 性能数据。

保存实际依赖版本便于复现：

```bash
python -m pip freeze > results/phase1-remote-XXXXXX/environment.txt
```

回到本地终端取回结果：

```bash
mkdir -p results/remote
rsync -av USER@GPU_SERVER:/path/to/miniserve/results/ ./results/remote/
```

## 6. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `CUDA is unavailable` | 检查是否登录到 GPU 服务器、`nvidia-smi` 是否正常、是否激活了安装 CUDA PyTorch 的同一环境 |
| 模型下载失败/离线缓存缺失 | 在允许联网的环境先完成相同 revision 的下载；检查 `HF_HOME` 与运行时一致，准备缓存时不要设置 `HF_HUB_OFFLINE=1` |
| `Python not found` | 指定 `VENV_DIR`，或设置 `MINISERVE_PYTHON` 为远端解释器的绝对路径 |
| `CUDA out of memory` | 检查该 GPU 是否有其他任务；选用空闲且允许使用的卡，或用更短 prompt/输出先定位问题；记录改过的条件 |
| token IDs 不一致 | 保留 `correctness.json`、依赖版本和 GPU 信息；用 `--attn-implementation eager` 复查。不要把正确性失败当作可用 benchmark |
| `ModuleNotFoundError: miniserve` | 在项目根目录用同一个 Python 运行 `python -m pip install -e . --no-deps` |

单 GPU 的 correctness 完成后，再按项目计划进入 HTTP/SSE、调度、continuous batching 与 KV admission 阶段。
