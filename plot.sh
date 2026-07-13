#!/bin/bash
#SBATCH --job-name="0plot"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00


# --- Conda setup ---
spack load miniconda3            
source activate plots 
spack unload miniconda3

echo "Start plotting"

srun python evaluate.py

echo "Finished Plotting Job."

## sbatch plot.sh