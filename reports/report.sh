#!/bin/bash
#SBATCH --job-name=report
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err

spack load miniconda3            
source activate thesis 
spack unload miniconda3

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

MODELS=(Basic Rep_M Rep_Z)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Start reporting for $MODEL (array task $SLURM_ARRAY_TASK_ID)"
srun python reports/report.py calc_rho_full_ds "$MODEL" "${SLURM_CPUS_PER_TASK:-8}"
echo "[$(date)] Finished $MODEL."

# sbatch reports/join.sh
# sbatch reports/collect_gof.sh
# sbatch reports/run_param_summary.sh
# sbatch --array=1-2 reports/report.sh