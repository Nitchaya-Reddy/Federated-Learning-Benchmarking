#!/bin/bash
# =====================================================================
# 07_pruning.sbatch
# Unstructured magnitude pruning experiments.
# Tests FedAvg and FedNova with prune_ratio ∈ {0.0,0.5,0.7,0.9,0.95}.
#
# Array indices (20 total):
#   algo x prune_ratio x partition
#   0-4:  prune_fedavg  iid     ratios=[0.0,0.5,0.7,0.9,0.95]
#   5-9:  prune_fednova iid     ratios=[0.0,0.5,0.7,0.9,0.95]
#  10-14: prune_fedavg  noniid  ratios=[0.0,0.5,0.7,0.9,0.95]
#  15-19: prune_fednova noniid  ratios=[0.0,0.5,0.7,0.9,0.95]
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/07_pruning.sbatch
# =====================================================================

#SBATCH --job-name=fl_prune
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=/blue/cis6931/sujeongjo/logs/prune_%A_%a.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/prune_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

# ── Map array index ────────────────────────────────────────────────
ALGOS=(prune_fedavg prune_fedavg prune_fedavg prune_fedavg prune_fedavg
       prune_fednova prune_fednova prune_fednova prune_fednova prune_fednova
       prune_fedavg prune_fedavg prune_fedavg prune_fedavg prune_fedavg
       prune_fednova prune_fednova prune_fednova prune_fednova prune_fednova)

RATIOS=(0.0 0.5 0.7 0.9 0.95  0.0 0.5 0.7 0.9 0.95
        0.0 0.5 0.7 0.9 0.95  0.0 0.5 0.7 0.9 0.95)

PARTS=(iid    iid    iid    iid    iid
       iid    iid    iid    iid    iid
       noniid noniid noniid noniid noniid
       noniid noniid noniid noniid noniid)

ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
RATIO=${RATIOS[$SLURM_ARRAY_TASK_ID]}
PARTITION=${PARTS[$SLURM_ARRAY_TASK_ID]}
NUM_CLIENTS=10

# Convert ratio to integer percent for directory name
RATIO_PCT=$(python3 -c "print(int(float('${RATIO}') * 100))")

BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
DATA=$BASE/data
RESULTS=$BASE/results/${ALGO}_${PARTITION}_${NUM_CLIENTS}clients_${RATIO_PCT}pct
LOGS=$BASE/logs

mkdir -p "$LOGS" "$RESULTS"

echo "============================================"
echo " Magnitude Pruning FL"
echo " algo=$ALGO  prune_ratio=$RATIO  partition=$PARTITION  clients=$NUM_CLIENTS"
echo " Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo " Node: $SLURMD_NODENAME  Start: $(date)"
echo "============================================"

module purge
module load pytorch/2.8.0
pip uninstall -q -y torch torchvision 2>/dev/null || true
pip install --user --quiet nvflare

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SRC:$PYTHONPATH"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo ""
echo "--- Starting pruning FL simulation ---"
python "$SRC/run_simulation.py" \
    --algo        "$ALGO" \
    --partition   "$PARTITION" \
    --num_clients "$NUM_CLIENTS" \
    --rounds      100 \
    --local_epochs 5 \
    --batch_size  64 \
    --lr          0.01 \
    --alpha       0.5 \
    --prune_ratio "$RATIO" \
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
