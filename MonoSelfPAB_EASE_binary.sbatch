#!/bin/bash
#SBATCH -J ease_selfPAB
#SBATCH -p lrz-dgx-1-p100x8          # adjust to your GPU partition
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                     # 64G is usually enough; raise if needed
#SBATCH -t 20:00:00
#SBATCH -o /dss/dsshome1/06/ge38qav/Pretrained_Models/slurm/%x_%j.out
#SBATCH -e /dss/dsshome1/06/ge38qav/Pretrained_Models/slurm/%x_%j.err

set -euo pipefail
echo "Node: $(hostname) | GPU(s): ${CUDA_VISIBLE_DEVICES:-N/A} | Start: $(date)"

# ---------- Paths (edit these to your layout) ----------
CODE="$HOME/Pretrained_Models/SelfPAB"           # cloned repo
VENV_DIR="$HOME/.venvs/harnet"                              # venv location in HOME
OUTDIR="$HOME/Pretrained_Models/experiments_selfPAB/exp_5_binary" 

DATA="$HOME/DATA/raw/labeled"                                      # folder with your X.npy/Y.npy/pid.npy

# Keep BLAS threads in check
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HYDRA_FULL_ERROR=1
export PATH="$HOME/.local/bin:$PATH"   # user-site scripts (pip, virtualenv, etc.)

# --- Activate venv ---
source "$VENV_DIR/bin/activate"
python -V || { echo 'Python in venv not working'; exit 1; }
nvidia-smi || true

ulimit -n 4096  # increase if you run into "Too many open files" errors

# ---------------- run the downstream evaluation ----------------
echo "Launching loo_cross_validation.py ..."
srun python loo_cross_validation.py \
  -p params/MonoSelfPAB_downstream_experiments/MonoSelfPAB_downstream_EASE/config_binary.yml \
  -d "$DATA"

echo "Done: $(date)"
