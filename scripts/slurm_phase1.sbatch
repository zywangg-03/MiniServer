#!/usr/bin/env bash
#SBATCH --job-name=miniserve-phase1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=miniserve-%j.out

# 学校远程集群模板：单节点、单 GPU；没有 NCCL、torchrun 或多卡并行。
# 从项目根目录提交；账户、分区和所需 module 由学校环境决定。
# 可用 sbatch --account=... --partition=... --time=... 覆盖资源参数。
# 某些学校要求指定 GPU 类型，请按其文档覆盖 --gres=gpu:<type>:1。
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the project directory.}"
export PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

# 如果学校要求 module load，请在此处添加其规定的 Python / CUDA 模块。
# 不覆盖 Slurm 管理的 CUDA_VISIBLE_DEVICES；cuda:0 是分配后进程看到的第一张卡。
cd -- "$PROJECT_ROOT"
srun --ntasks=1 bash "$PROJECT_ROOT/scripts/run_cluster.sh" "$@"
