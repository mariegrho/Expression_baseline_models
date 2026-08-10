#!/bin/bash
#SBATCH --job-name="score"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# --- Conda setup ---
spack load miniconda3            
source activate thesis 
spack unload miniconda3

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

MODELS=(Basic Rep_M Rep_Z)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

RESULTS_ROOT="results/120_hpf/${MODEL}/full"
OUT_CSV="results/results_summary/${MODEL}/fit_scorecard.csv"

echo "[$(date)] Starting task ${MODEL}"
srun python reports/build_fit_scorecard.py --results-root "$RESULTS_ROOT" --out-csv "$OUT_CSV" --n-workers "$SLURM_CPUS_PER_TASK" --checkpoint-every 200
#srun python reports/recompute_loo_waic.py --results-root "$RESULTS_ROOT" --out-csv "$OUT_CSV" --n-workers "$SLURM_CPUS_PER_TASK"

echo "[$(date)] Finished task"

# sbatch --array=1-2 reports/scorecard.sh
