#!/bin/bash
# =====================================================================
# 05_plot_results.sbatch
# Run AFTER all experiment jobs finish to generate all figures.
#
# Submit: sbatch /blue/cis6931/sujeongjo/jobs/05_plot_results.sbatch
# Or run interactively after checking results are present:
#   srun --account=cis6931 --qos=cis6931 --mem=8gb --time=00:30:00 \
#        --pty bash -i
#   cd /blue/cis6931/sujeongjo
#   python src/plot_results.py
# =====================================================================

#SBATCH --job-name=fl_plots
#SBATCH --account=cis6931
#SBATCH --qos=cis6931
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/blue/cis6931/sujeongjo/logs/plots_%j.out
#SBATCH --error=/blue/cis6931/sujeongjo/logs/plots_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sujeongjo@ufl.edu

BASE=/blue/cis6931/sujeongjo
SRC=$BASE/src
RESULTS=$BASE/results
FIGURES=$BASE/figures
LOGS=$BASE/logs

mkdir -p "$LOGS" "$FIGURES"

echo "Generating figures from results in $RESULTS ..."

module purge
module load pytorch/2.8.0
# Remove any user-installed torch that shadows the module version
pip uninstall -q -y torch torchvision 2>/dev/null || true
pip install --user --quiet matplotlib

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SRC:$PYTHONPATH"

python "$SRC/plot_results.py" \
    --results_dir "$RESULTS" \
    --out_dir     "$FIGURES" \
    --num_clients 10

EXIT=$?
echo "Done: $(date)  |  Exit: $EXIT"
echo "Figures in: $FIGURES"
exit $EXIT
