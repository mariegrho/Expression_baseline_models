#!/bin/bash
#SBATCH --job-name="expr"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryberry890@gmail.com

# --- Conda setup ---
spack load miniconda3            
source activate thesis 
spack unload miniconda3

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

# --- Get current gene ---
GENE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/genes_clustered_white.txt)

echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID: $GENE_ID"

srun python simulate.py --gene_id "$GENE_ID" --model_version Rep_M --dataset Medina_Munoz_ribo --plot --t_end 8

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID: $GENE_ID"

# sbatch run_array_job.sh
# sbatch --array=1-1 run_array_job.sh
# watch squeue --me
#sed -i 's/\r$//' data/genes_clustered_white.txt
