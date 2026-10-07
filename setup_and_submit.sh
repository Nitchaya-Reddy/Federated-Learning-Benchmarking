#!/bin/bash
# =====================================================================
# setup_and_submit.sh
# Run this ONCE after uploading files to HiPerGator.
# It creates the directory structure and submits all jobs in order.
#
# Job order:
#   01 centralized baseline
#   02 FedAvg
#   03 FedNova
#   04 Quantization (baseline)
#   05 Plot results
#   06 Quantization + Error Feedback + Top-k  (sujeongjo contribution)
#   07 Final report
#   08 Pruning
#   09 Local epochs sensitivity
#   10 ResNet-18
#
# Usage (from login node):
#   chmod +x /blue/cis6931/sujeongjo/setup_and_submit.sh
#   /blue/cis6931/sujeongjo/setup_and_submit.sh
# =====================================================================
set -e
BASE=/blue/cis6931/sujeongjo
echo "=== Creating directory structure ==="
mkdir -p "$BASE"/{src,jobs,data,logs,results,figures}
echo "=== Directory layout ==="
echo "$BASE/"
echo "  src/          <- all Python source files"
echo "  jobs/         <- all .sbatch files"
echo "  data/         <- CIFAR-10 (auto-downloaded on first run)"
echo "  logs/         <- SLURM stdout/stderr"
echo "  results/      <- JSON metrics from each experiment"
echo "  figures/      <- plots"
echo ""

# ── Submit jobs ──────────────────────────────────────────────────────
echo "=== Submitting jobs ==="

# 01: centralized baseline (no dependency)
JID1=$(sbatch --parsable "$BASE/jobs/01_centralized.sbatch")
echo "Submitted 01 centralized baseline:        $JID1"

# 02-06: wait for job 1 so CIFAR-10 is fully downloaded first
JID2=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/02_fedavg.sbatch")
echo "Submitted 02 FedAvg array:                $JID2"

JID3=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/03_fednova.sbatch")
echo "Submitted 03 FedNova array:               $JID3"

JID4=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/04_quantization.sbatch")
echo "Submitted 04 Quantization array:          $JID4"

JID5=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/06_quantization_ef_topk.sbatch")
echo "Submitted 06 Quantization EF+TopK array:  $JID5"

JID6=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/08_pruning.sbatch")
echo "Submitted 08 Pruning array:               $JID6"

JID7=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/09_local_epochs.sbatch")
echo "Submitted 09 Local Epochs array:          $JID7"

JID8=$(sbatch --parsable --dependency=afterok:${JID1} "$BASE/jobs/10_resnet18.sbatch")
echo "Submitted 10 ResNet-18 array:             $JID8"

# 05 plot: waits for ALL experiments to finish
JID9=$(sbatch --parsable \
    --dependency=afterany:${JID1}:${JID2}:${JID3}:${JID4}:${JID5}:${JID6}:${JID7}:${JID8} \
    "$BASE/jobs/05_plot_results.sbatch")
echo "Submitted 05 Plotting (depends on all):   $JID9"

# 07 final report: runs after plotting
JID10=$(sbatch --parsable \
    --dependency=afterany:${JID9} \
    "$BASE/jobs/07_final_report.sbatch")
echo "Submitted 07 Final Report:                $JID10"

echo ""
echo "=== All jobs submitted! ==="
echo ""
echo "Monitor progress:"
echo "  squeue -u sujeongjo"
echo "  tail -f $BASE/logs/fedavg_*.out"
echo ""
echo "Check results as they come in:"
echo "  ls $BASE/results/"
echo ""
echo "When complete, view full report:"
echo "  cat $BASE/FINAL_REPORT.txt"
echo ""
echo "Expected total runtime: ~8-12 hours (running in parallel)"
