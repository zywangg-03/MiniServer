#!/usr/bin/env bash
# 远程服务器版本：SSH 登录学校 GPU 服务器后执行，不依赖 Slurm 环境变量。
# 单服务器、单 GPU 推理；没有多 GPU 并行。本脚本不建立 SSH 连接或申请资源。
# 可选：Slurm 计算节点也可通过 slurm_phase1.sbatch 调用。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT

# 可以显式设置 HF_HOME，使用学校提供的数据盘或 scratch 缓存。
# 默认项目内缓存便于使用同一个目录预下载模型；不假设学校的 scratch 路径。
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MINISERVE_RUN_LABEL="${MINISERVE_RUN_LABEL:-remote}"
mkdir -p -- "$HF_HOME"

# 保留 GPU 分配和可见性；同一个后端运行在本地卡或远程分配到的单张卡上。
exec bash "$PROJECT_ROOT/scripts/run_local.sh" "$@"
