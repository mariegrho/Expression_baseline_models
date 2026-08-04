#!/bin/bash
#SBATCH --job-name="scorecard"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=12:00:00

# --- Conda setup ---
spack load miniconda3            
source activate thesis 
spack unload miniconda3

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

MODELS=(Basic Rep_M Rep_Z)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

RESULTS_ROOT="results/120_hpf/${MODEL}/all"
OUT_CSV="results/results_summary/${MODEL}/fit_scorecard.csv"

echo "[$(date)] Starting task"
srun python reports/build_fit_scorecard.py \
    --results-root "$RESULTS_ROOT" \
    --out-csv "$OUT_CSV" \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --checkpoint-every 200

echo "[$(date)] Finished task"

# sbatch --array=0-0 reports/scorecard.sh
