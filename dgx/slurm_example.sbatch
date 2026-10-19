#!/usr/bin/env bash
# Example Slurm job (only if the DGX is behind a scheduler). Edit partition / account / paths.
#   sbatch dgx/slurm_example.sbatch 10_generate.sh        # or 30_train.sh, 40_eval.sh ...
#SBATCH --job-name=realac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.log
set -euo pipefail
STEP="${1:-10_generate.sh}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# container route:
#   srun singularity exec --nv -B "$ROOT:/work" realac.sif bash /work/dgx/$STEP
# conda route:
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate realac
bash "$ROOT/dgx/$STEP"
