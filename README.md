# MiniServe

MiniServe 是一个逐步实现的 LLM inference serving 项目。当前按[项目计划第 9 节](MiniServe_Implementation_Blueprint_CN.md#9-第一次编码-session从这里开始)实现 **Phase 1：基础推理与正确性验证**：Hugging Face `generate()` baseline、自定义 prefill/decode 循环，以及逐 token 对照测试。

本地 NVIDIA GPU 和学校远程 GPU 服务器使用同一个推理后端，分别提供启动脚本。学校环境按直接 SSH 登录 GPU 服务器准备，另附可选 Slurm 模板。当前每个进程在单服务器上使用一张 GPU，未实现多 GPU 通信。HTTP/SSE、请求调度、continuous batching、KV admission 和并发性能实验属于后续阶段。

学校 GPU 的逐步安装、验证命令、预期结果和排错说明见[远程 GPU 手动验证指南](docs/GPU_VALIDATION_CN.md)。

## 环境安装

推荐 Python 3.11，项目支持 Python 3.11–3.13。首先在**执行推理的机器**上创建环境，并按照 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)安装与操作系统、NVIDIA 驱动和 CUDA 兼容的 PyTorch（项目要求 `torch>=2.6`）。本地 macOS 环境可使用 CPU 做开发验证；正式 GPU 实验使用 CUDA。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 此处执行 PyTorch 官方安装页针对目标机器给出的 pip install 命令。
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

Transformers 固定为 `4.57.1`，pytest 固定为 `8.4.2`。PyTorch 的 CUDA wheel 由目标环境选择，运行记录保存实际版本。默认模型 `Qwen/Qwen2.5-0.5B-Instruct` 固定到其官方提交 `7ae5576`，报告同时保存完整解析后的 revision。首次加载需要联网下载权重；后续可使用 `--local-files-only` 读取缓存。

## 本地单 GPU

在已安装依赖的项目目录运行：

```bash
bash scripts/run_local.sh
```

脚本依次执行 token IDs 正确性比较、HF baseline 和手动生成基线，默认 FP16、greedy decoding、最多 64 个输出 tokens、1 次 warm-up、5 次重复。结果放入独立的 `results/phase1-local-*/` 目录，包含 `correctness.json`、`hf_baseline.json` 和 `manual_generate.json`。正确性检查失败时停止后续测量。

也可以分别运行，方便只重试某一步：

```bash
python -m benchmarks.verify --device cuda:0 --dtype float16 --output results/correctness.json
python -m benchmarks.hf_baseline --device cuda:0 --dtype float16 --warmup 1 --repetitions 5 --max-new-tokens 64 --output results/hf_baseline.json
python -m benchmarks.manual_generate --device cuda:0 --dtype float16 --warmup 1 --repetitions 5 --max-new-tokens 64 --output results/manual_generate.json
```

`cuda:0` 指进程可见的第一张 GPU。两种脚本均保留已有的 `CUDA_VISIBLE_DEVICES`，因此可以使用调度器分配的 GPU。公共参数 `--model`、`--revision`、`--local-files-only`、`--raw-prompt` 可以直接追加给脚本；默认使用模型的 chat template，`--raw-prompt` 用于比较纯文本输入。选择其他模型时应同时指定对应 revision。

```bash
# CPU 开发验证：明确使用 FP32，结果不能与 GPU FP16 性能混用。
MINISERVE_DEVICE=cpu MINISERVE_DTYPE=float32 bash scripts/run_local.sh

# 调整重复次数、输出长度和虚拟环境路径。
VENV_DIR=/path/to/venv MINISERVE_REPETITIONS=5 MINISERVE_MAX_NEW_TOKENS=64 bash scripts/run_local.sh --local-files-only
```

## 学校远程 GPU 服务器：直接 SSH

在本地项目根目录同步源码；将 `USER@GPU_SERVER` 和 `/path/to/miniserve` 替换为学校提供的 SSH 账号、服务器地址和远端项目目录。同步命令排除虚拟环境、模型缓存和已有实验结果；**不要把本地 macOS `.venv` 上传到 Linux 服务器**。

```bash
# 以下命令在本地执行。
ssh USER@GPU_SERVER 'mkdir -p /path/to/miniserve'
rsync -av --exclude='.venv*' --exclude='venv/' --exclude='.bootstrap/' --exclude='.python/' --exclude='.uv-cache/' --exclude='.cache/' --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.egg-info/' --exclude='results/' --exclude='.DS_Store' ./ USER@GPU_SERVER:/path/to/miniserve/
ssh USER@GPU_SERVER
```

登录后，在**远端 Linux 服务器**重新安装环境，选择匹配远端 NVIDIA 驱动和 CUDA 的 PyTorch wheel：

```bash
# 以下命令在远端执行。
cd /path/to/miniserve
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
# 在此执行 PyTorch 官方安装页针对远端 Linux / CUDA 给出的安装命令。
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

export HF_HOME="$PWD/.cache/huggingface"
bash scripts/run_cluster.sh
```

`run_cluster.sh` 不依赖 Slurm，默认使用远端 `.venv/bin/python` 和 `cuda:0`，结果保存到 `results/phase1-remote-*/`。可通过 `VENV_DIR` 指定其他远端环境，通过 `HF_HOME` 把模型缓存放到学校允许使用的数据盘。若服务器暴露多张 GPU，可用 `MINISERVE_DEVICE=cuda:1 bash scripts/run_cluster.sh` 选择获准使用的卡；程序仍只使用这一张卡，脚本不会改写已有的 `CUDA_VISIBLE_DEVICES`。

首次运行需要下载默认模型。若希望先准备缓存再离线执行，在允许联网的机器上将指定 revision 下载到远端可读取的同一个 `HF_HOME`：

```bash
python -c 'from huggingface_hub import snapshot_download; from miniserve.config import ModelConfig; c = ModelConfig(); snapshot_download(c.model_name, revision=c.revision)'
export HF_HUB_OFFLINE=1
bash scripts/run_cluster.sh --local-files-only
```

`HF_HOME` 控制 Hugging Face 缓存目录，`HF_HUB_OFFLINE=1` 禁止 Hub HTTP 请求；必须先完成所需模型 revision 的下载。[Hugging Face 环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

运行完成后，可在**本地**取回原始结果：

```bash
mkdir -p results/remote
rsync -av USER@GPU_SERVER:/path/to/miniserve/results/ ./results/remote/
```

## 可选：Slurm 集群

仅当运行环境使用 Slurm 时使用此入口。项目、远端 Python 环境和模型缓存须放在计算节点可见的存储中。从项目根目录提交模板，按学校要求替换账户和分区；不需要这些参数时可省略：

```bash
export HF_HOME=/path/to/shared/scratch/huggingface
export VENV_DIR=/path/to/cluster/venv
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm_phase1.sbatch

# 缓存已完整下载且计算节点不能联网时：
HF_HUB_OFFLINE=1 sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm_phase1.sbatch --local-files-only
```

模板申请单节点、单任务、1 张 GPU、4 CPU cores、16 GiB 主机内存和 1 小时运行时间，具体资源可用 `sbatch` 参数覆盖。学校需要特定 GPU 型号或 module 时，按本校文档修改模板。作业日志为 `miniserve-<job_id>.out`，JSON 结果保存在 `results/phase1-<job_id>-*/`。Slurm 不会自动复制项目与模型文件，路径必须能被计算节点访问。[Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)和 [GPU 资源配置说明](https://slurm.schedmd.com/gres.html)解释了资源申请及 GPU 可见性的规则。

## 测试与结果边界

2026-11-16 本地验证：Python 3.11.18、PyTorch 2.15.0、Transformers 4.57.1，macOS ARM64 / CPU，**63 passed、7 skipped**。已验证离线微型 Qwen2 的推理正确性及本地 checkpoint → CLI → JSON 完整流程；7 项跳过项是需要正式权重的测试。测试报告位于 `results/local-tests.xml`，本次本地依赖记录位于 `results/local-environment.txt`；这些本机产物不提交到 Git。

```bash
# 离线测试：随机初始化的微型模型，无需下载正式权重。
python -m pytest -q

# 正式 Qwen 模型对照：在远端 GPU 服务器或已分配的计算节点执行。
MINISERVE_RUN_MODEL_TESTS=1 MINISERVE_TEST_DEVICE=cuda:0 MINISERVE_TEST_DTYPE=float16 python -m pytest -q -m model
```

离线测试用于验证生成循环、KV cache 长度与 EOS/长度边界；正式模型测试与 `benchmarks.verify` 用于验证中英文、不同输入长度下手动生成与 `generate()` 的 token IDs 是否一致。微型模型通过测试不能替代真实 Qwen + CUDA 验收。

基线 JSON 保留实际生成的 token IDs、耗时、输出 token 数、CUDA 峰值显存及软硬件环境，便于复查和复现实验。CUDA 计时需要同步执行结果；CPU 结果不应填写 GPU 显存数字。当前未发布学校集群或本地 NVIDIA GPU 的实测性能数据，也未宣称相对 HF 的加速。

比较性能时，应保持模型 revision、硬件、dtype、prompt、chat template、EOS 规则、输出限制和 warm-up/repetitions 相同。两种生成路径均禁用采样、重复惩罚等 logits processors，显式关闭 HF 的 checkpoint 默认配置回填，并使用 generation config 的完整 EOS 列表。手动 prefill 在生成时只计算最后一个位置的 vocabulary logits，与 HF 的行为一致。Phase 1 是单请求推理测量，还没有客户端 TTFT、并发 throughput 或排队时间；这些指标将在后续 serving 阶段采集。

实现细节以 [Hugging Face 4.57.1 KV cache 文档](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)和 [Qwen 模型说明](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)为参考。完整后续里程碑见[项目计划](MiniServe_Implementation_Blueprint_CN.md)。
