#!/bin/bash
#SBATCH --job-name="fit"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

# --mail-user=maryberry890@gmail.com

# --- Conda setup ---
spack load miniconda3            
source activate thesis 
spack unload miniconda3

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

# --- Get current gene ---
GENE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/missing_genes.txt)
#GENE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/genes.txt)

echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID: $GENE_ID"

datasets_120hpf=("White" "Pauli" "BK" "JN")
#datasets_8hpf=("Medina_Munoz_polyA" "Medina_Munoz_ribo")

#model_versions=("ZGA_M" "ZGA_Z")
model_versions=("Rep_M" "Rep_Z")

#for dataset in "${datasets_120hpf[@]}"; do
    #srun python basic_model.py --gene_id "$GENE_ID" --dataset "$dataset" --plot --t_end 120 --skip_duplicates

for model_version in "${model_versions[@]}"; do
    srun python simulate.py --gene_id "$GENE_ID" --model_version "$model_version" --plot --t_end 120 --seed 1
done

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID: $GENE_ID"

# sbatch --array=1-1 run_array_job_loop.sh
# sbatch --array=1-4335%200 run_array_job_loop.sh
# watch squeue --me
# sed -i 's/\r$//' data/genes.txt