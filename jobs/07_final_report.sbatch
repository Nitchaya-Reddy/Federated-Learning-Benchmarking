#!/bin/bash
# =====================================================================
# 06_final_report.sbatch
# Collects all results + log excerpts into a single FINAL_REPORT.txt.
# Run after all experiment jobs finish.
#
# Submit with dependency:
#   sbatch --dependency=afterany:<JOB_IDS> \
#          /blue/cis6931/sujeongjo/jobs/06_final_report.sbatch
# =====================================================================

#SBATCH --job-name=fl_final_report
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:15:00
#SBATCH --output=/blue/cis6931/sujeongjo/logs/final_report_%j.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/final_report_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src

module purge
module load pytorch/2.8.0

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SRC:$PYTHONPATH"

python "$SRC/generate_final_log.py" \
    --results_dir "$BASE/results" \
    --logs_dir    "$BASE/logs" \
    --out         "$BASE/FINAL_REPORT.txt"

echo ""
echo "Report written to: $BASE/FINAL_REPORT.txt"
echo "To view: cat $BASE/FINAL_REPORT.txt"
