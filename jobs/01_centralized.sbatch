#!/bin/bash
# =====================================================================
# 01_centralized.sbatch
# Run centralized CIFAR-10 training (no federation) as the accuracy
# ceiling baseline.  Run this first before any FL jobs.
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/01_centralized.sbatch
# =====================================================================

#SBATCH --job-name=fl_centralized
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/blue/cis6931/sujeongjo/logs/centralized_%j.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/centralized_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

# ── Paths ──────────────────────────────────────────────────────────
BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
DATA=$BASE/data
RESULTS=$BASE/results/centralized
LOGS=$BASE/logs

mkdir -p "$LOGS" "$RESULTS"

echo "============================================"
echo " Centralized Baseline"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURMD_NODENAME"
echo " Start : $(date)"
echo "============================================"

# ── Environment ────────────────────────────────────────────────────
module purge
module load pytorch/2.8.0
# Remove any user-installed torch/torchvision that shadows the module version
pip uninstall -q -y torch torchvision 2>/dev/null || true
pip install --user --quiet nvflare

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SRC:$PYTHONPATH"

echo "Python : $(which python) ($(python --version))"
python -c "import torch, torchvision; print('Imports OK | CUDA:', torch.cuda.is_available())"

echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Run ────────────────────────────────────────────────────────────
echo ""
echo "--- Starting centralized training ---"
python "$SRC/train_centralized.py" \
    --data_dir "$DATA" \
    --out_dir  "$RESULTS" \
    --epochs   100 \
    --batch_size 128 \
    --lr       0.1 \
    --seed     42

EXIT=$?
echo ""
echo "============================================"
echo " Done : $(date)  |  Exit: $EXIT"
echo " Results in: $RESULTS"
echo "============================================"
exit $EXIT
