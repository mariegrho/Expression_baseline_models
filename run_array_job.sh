#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

# --- Conda setup ---
spack load miniconda3            
source activate thesis 
spack unload miniconda3

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

# --- Get current gene ---
GENE_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/non_div_genes.txt)

echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID: $GENE_ID"

datasets=("White" "Pauli" "BK" "JN" "Medina_Munoz_polyA" "Medina_Munoz_ribo")
model_versions=("Rep_M" "Rep_Z" "Rep_V" "ZGA_M" "ZGA_Z" "Basic")

#srun python basic_model_single.py --gene_id "$GENE_ID" --dataset ${datasets[0]} --plot --t_end 120
srun python simulate.py --gene_id "$GENE_ID" --model_version ${model_versions[1]} --plot --t_end 120 --seed 1
#srun python basic_model.py --gene_id "$GENE_ID" --plot --t_end 120 --seed 1 --skip_duplicates

#srun python reports/compute_gof_src.py --gene_id "$GENE_ID" --model ${model_versions[3]} --t_end 120

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID: $GENE_ID"

# sbatch run_array_job.sh
# sbatch --array=2-10 run_array_job.sh
# sbatch --array=1-6589%50 run_array_job.sh
# watch squeue --me
# sed -i 's/\r$//' data/non_div_genes.txt
# rm results/logs/fit_11939425*1.err
# rm -rf results/120_hpf/Rep_M/all1*
# rm -rf results/120_hpf/Rep_Z/all1*

