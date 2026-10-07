#!/bin/bash
# =====================================================================
# 09_resnet18.sbatch
# ResNet-18 experiments: repeat the core FedAvg vs FedNova comparison
# using a stronger model to raise the accuracy ceiling.
#
# ResNet-18 adapted for CIFAR-10 (~11.2M params vs ~1.3M for CNN).
# Expected centralized ceiling: ~93-95% vs ~86% for CNN.
#
# Array indices (12 total):
#   algo x partition x num_clients
#   algos:      0=fedavg, 1=fednova
#   partitions: iid, noniid
#   clients:    2, 5, 10
#
#   0-2:  fedavg  iid     clients=[2,5,10]
#   3-5:  fedavg  noniid  clients=[2,5,10]
#   6-8:  fednova iid     clients=[2,5,10]
#   9-11: fednova noniid  clients=[2,5,10]
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/09_resnet18.sbatch
# =====================================================================

#SBATCH --job-name=fl_resnet
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=0-11
#SBATCH --output=/blue/cis6931/sujeongjo/logs/resnet_%A_%a.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/resnet_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

# ── Map array index ────────────────────────────────────────────────
ALGOS=(fedavg  fedavg  fedavg
       fedavg  fedavg  fedavg
       fednova fednova fednova
       fednova fednova fednova)

PARTS=(iid    iid    iid
       noniid noniid noniid
       iid    iid    iid
       noniid noniid noniid)

CLIENTS=(2 5 10  2 5 10  2 5 10  2 5 10)

ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
PARTITION=${PARTS[$SLURM_ARRAY_TASK_ID]}
NUM_CLIENTS=${CLIENTS[$SLURM_ARRAY_TASK_ID]}

BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
DATA=$BASE/data
RESULTS=$BASE/results/${ALGO}_${PARTITION}_${NUM_CLIENTS}clients_resnet18_cosine
LOGS=$BASE/logs

mkdir -p "$LOGS" "$RESULTS"

echo "============================================"
echo " ResNet-18 FL (cosine LR)"
echo " algo=$ALGO  partition=$PARTITION  clients=$NUM_CLIENTS"
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
echo "--- Starting ResNet-18 FL simulation ---"
python "$SRC/run_simulation.py" \
    --algo        "$ALGO" \
    --partition   "$PARTITION" \
    --num_clients "$NUM_CLIENTS" \
    --rounds      100 \
    --local_epochs 5 \
    --batch_size  64 \
    --lr          0.01 \
    --alpha       0.5 \
    --model       resnet18 \
    --lr_schedule cosine \
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
