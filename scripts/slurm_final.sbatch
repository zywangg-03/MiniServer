#!/usr/bin/env bash
#SBATCH --job-name=miniserve-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=miniserve-final-%j.out
# Great Lakes: sbatch --account=lsa2 --partition=gpu scripts/slurm_final.sbatch --quick
# Account/partition access must be valid for the submitting user.
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd "$SLURM_SUBMIT_DIR"
# Match the Python module used to CREATE the remote .venv. Adjust if needed.
module load python3.11-anaconda
srun --ntasks=1 bash scripts/run_final.sh "$@"
