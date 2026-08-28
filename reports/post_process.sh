#!/bin/bash
#SBATCH --job-name=post
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err    
#SBATCH --time=12:00:00                   
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

spack load miniconda3            
source activate thesis 
spack unload miniconda3

MODELS=(Basic Rep_M Rep_Z)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

GENES=data/genes.txt
RESULTS_DIR=results/120_hpf/${MODEL}/full
OUT_DIR=results/results_summary/${MODEL}

export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

echo "[$(date)] Running collect_results_concurrent() for ${MODEL} to join NETCDF files..."
python3 reports/join_netcdf_faster.py "$RESULTS_DIR" "$GENES" "$OUT_DIR" --mode simulation # genes, res_dir, out_dir, mode
#python3 reports/join_netcdf_faster.py "$RESULTS_DIR" "$GENES" "$OUT_DIR" --mode params # genes, res_dir, out_dir, mode

echo "[$(date)] Finished post_processing."

# sbatch --array=0-2 reports/post_process.sh 
