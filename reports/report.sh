#!/bin/bash
#SBATCH --job-name=report
#SBATCH --cpus-per-task=8     
#SBATCH --mem=4G             
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err

# --- Conda setup ---
spack load miniconda3            
source activate plots 
spack unload miniconda3

for MODEL in Rep_M Rep_Z; do
    echo "Start reporting for $MODEL"
    srun python reports/report.py calc_rho_full_ds "$MODEL" "${SLURM_CPUS_PER_TASK:-8}"
done
echo "[$(date)] Finished all."

# sbatch reports/join.sh
# sbatch reports/collect_gof.sh
# sbatch reports/run_param_summary.sh
# sbatch reports/report.sh