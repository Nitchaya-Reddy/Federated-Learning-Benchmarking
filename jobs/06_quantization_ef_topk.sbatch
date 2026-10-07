#!/bin/bash
# =====================================================================
# 05_quantization_ef_topk.sbatch
# Quantized communication with Error Feedback + Top-k Sparsification.
# Tests FedAvg and FedNova with 2, 4, 8, 16, 32-bit quantization
# combined with Error Feedback and Top-k sparsification (top 10%).
#
# Array indices (20 total):
#   algo x bits x partition
#   algo:      0=quant_fedavg, 1=quant_fednova
#   bits:      2, 4, 8, 16, 32
#   partition: noniid, iid
#
# Index layout:
#   0-4:  quant_fedavg  noniid  bits=[2,4,8,16,32]
#   5-9:  quant_fednova noniid  bits=[2,4,8,16,32]
#  10-14: quant_fedavg  iid     bits=[2,4,8,16,32]
#  15-19: quant_fednova iid     bits=[2,4,8,16,32]
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/05_quantization_ef_topk.sbatch
# =====================================================================
#SBATCH --job-name=fl_quant_ef
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
#SBATCH --output=/blue/cis6931/sujeongjo/logs/quant_ef_%A_%a.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/quant_ef_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

# ── Map array index ────────────────────────────────────────────────
ALGOS=(quant_fedavg quant_fedavg quant_fedavg quant_fedavg quant_fedavg
       quant_fednova quant_fednova quant_fednova quant_fednova quant_fednova
       quant_fedavg quant_fedavg quant_fedavg quant_fedavg quant_fedavg
       quant_fednova quant_fednova quant_fednova quant_fednova quant_fednova)
BITS=(2 4 8 16 32  2 4 8 16 32  2 4 8 16 32  2 4 8 16 32)
PARTS=(noniid noniid noniid noniid noniid
       noniid noniid noniid noniid noniid
       iid    iid    iid    iid    iid
       iid    iid    iid    iid    iid)

ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
QBITS=${BITS[$SLURM_ARRAY_TASK_ID]}
PARTITION=${PARTS[$SLURM_ARRAY_TASK_ID]}
NUM_CLIENTS=10
TOP_K=0.5   # keep top 10% of gradient elements

BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
DATA=$BASE/data
RESULTS=$BASE/results/${ALGO}_${PARTITION}_${NUM_CLIENTS}clients_${QBITS}bit_top50k_ef
LOGS=$BASE/logs

mkdir -p "$LOGS" "$RESULTS"

echo "============================================"
echo " Quantized FL + Error Feedback + Top-k"
echo " algo=$ALGO  bits=$QBITS  partition=$PARTITION"
echo " top_k_ratio=$TOP_K  clients=$NUM_CLIENTS"
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
echo "--- Starting quantized FL simulation (EF + Top-k) ---"

python "$SRC/run_simulation.py" \
    --algo         "$ALGO" \
    --partition    "$PARTITION" \
    --num_clients  "$NUM_CLIENTS" \
    --rounds       100 \
    --local_epochs 5 \
    --batch_size   64 \
    --lr           0.01 \
    --alpha        0.5 \
    --quant_bits   "$QBITS" \
    --top_k_ratio  "$TOP_K" \
    --data_dir     "$DATA" \
    --out_dir      "$RESULTS" \
    --seed         42

EXIT=$?
echo ""
echo "============================================"
echo " Done: $(date)  |  Exit: $EXIT"
echo " Results: $RESULTS"
echo "============================================"
exit $EXIT
