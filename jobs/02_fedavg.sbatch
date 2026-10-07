#!/bin/bash
# =====================================================================
# 02_fedavg.sbatch
# Runs FedAvg under IID and non-IID partitions for 2, 5, 10 clients.
# Uses SLURM array: array index encodes (partition x num_clients).
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/02_fedavg.sbatch
# Array indices:
#   0 = iid    / 2  clients
#   1 = iid    / 5  clients
#   2 = iid    / 10 clients
#   3 = noniid / 2  clients
#   4 = noniid / 5  clients
#   5 = noniid / 10 clients
# =====================================================================

#SBATCH --job-name=fl_fedavg
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/blue/cis6931/sujeongjo/logs/fedavg_%A_%a.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/fedavg_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

# ── Map array index -> (partition, num_clients) ────────────────────
PARTITIONS=(iid iid iid noniid noniid noniid)
CLIENTS=(2 5 10 2 5 10)

PARTITION=${PARTITIONS[$SLURM_ARRAY_TASK_ID]}
NUM_CLIENTS=${CLIENTS[$SLURM_ARRAY_TASK_ID]}

# ── Paths ──────────────────────────────────────────────────────────
BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
DATA=$BASE/data
RESULTS=$BASE/results/fedavg_${PARTITION}_${NUM_CLIENTS}clients
LOGS=$BASE/logs

mkdir -p "$LOGS" "$RESULTS"

echo "============================================"
echo " FedAvg  |  partition=$PARTITION  clients=$NUM_CLIENTS"
echo " Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo " Node: $SLURMD_NODENAME  Start: $(date)"
echo "============================================"

# ── Environment ────────────────────────────────────────────────────
module purge
module load pytorch/2.8.0
# Remove any user-installed torch/torchvision that shadows the module version
pip uninstall -q -y torch torchvision 2>/dev/null || true
pip install --user --quiet nvflare

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SRC:$PYTHONPATH"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# ── Run ────────────────────────────────────────────────────────────
echo ""
echo "--- Starting FedAvg simulation ---"
python "$SRC/run_simulation.py" \
    --algo        fedavg \
    --partition   "$PARTITION" \
    --num_clients "$NUM_CLIENTS" \
    --rounds      100 \
    --local_epochs 5 \
    --batch_size  64 \
    --lr          0.01 \
    --alpha       0.5 \
    --data_dir    "$DATA" \
    --out_dir     "$RESULTS" \
    --seed        42

EXIT=$?
echo ""
echo "============================================"
echo " Done: $(date)  |  Exit: $EXIT"
echo " Results: $RESULTS"
echo "============================================"
exit $EXIT
