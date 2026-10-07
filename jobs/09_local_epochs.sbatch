#!/bin/bash
# =====================================================================
# 08_local_epochs.sbatch
# Local epoch sensitivity experiments.
# Tests FedAvg and FedNova with local_epochs ∈ {1,3,5,10,20}.
# Uses non-IID partition (most sensitive to local epoch count).
# Note: local_epochs=5 is the default experiment; re-running it here
#       produces a tagged output dir (*_5ep) for cleaner comparison.
#
# Array indices (10 total):
#   0-4:  fedavg  noniid  epochs=[1,3,5,10,20]
#   5-9:  fednova noniid  epochs=[1,3,5,10,20]
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/08_local_epochs.sbatch
# =====================================================================

#SBATCH --job-name=fl_epochs
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/blue/cis6931/sujeongjo/logs/epochs_%A_%a.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/epochs_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

# ── Map array index ────────────────────────────────────────────────
ALGOS=(fedavg fedavg fedavg fedavg fedavg
       fednova fednova fednova fednova fednova)

EPOCHS=(1 3 5 10 20  1 3 5 10 20)

ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
LOCAL_EPOCHS=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
PARTITION=noniid
NUM_CLIENTS=10

BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
DATA=$BASE/data
RESULTS=$BASE/results/${ALGO}_${PARTITION}_${NUM_CLIENTS}clients_${LOCAL_EPOCHS}ep
LOGS=$BASE/logs

mkdir -p "$LOGS" "$RESULTS"

echo "============================================"
echo " Local Epoch Sensitivity"
echo " algo=$ALGO  local_epochs=$LOCAL_EPOCHS  partition=$PARTITION  clients=$NUM_CLIENTS"
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
echo "--- Starting local epoch sensitivity experiment ---"
python "$SRC/run_simulation.py" \
    --algo        "$ALGO" \
    --partition   "$PARTITION" \
    --num_clients "$NUM_CLIENTS" \
    --rounds      100 \
    --local_epochs "$LOCAL_EPOCHS" \
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
