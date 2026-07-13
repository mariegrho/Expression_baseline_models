#!/bin/bash
#SBATCH --job-name=report
#SBATCH --cpus-per-task=8     
#SBATCH --mem=4G             
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err

# --- Conda setup ---
spack load miniconda3            
source activate plots 
spack unload miniconda3

MODEL="Basic"

echo "Start reporting for $MODEL"

srun python reports/report.py calc_rho_full_ds "$MODEL" "${SLURM_CPUS_PER_TASK:-8}"

echo "Finished reporting."

# sbatch reports/report.sh