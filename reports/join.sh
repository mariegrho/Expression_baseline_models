#!/bin/bash
#SBATCH --job-name=0join     
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err    
#SBATCH --time=01:30:00                   
#SBATCH --mem=500MB
#SBATCH --cpus-per-task=1

spack load miniconda3            
source activate thesis 
spack unload miniconda3

MODEL="Rep_Z"
BASE_DIR="results/120_hpf/$MODEL/all"
OUT_FILE="results_summary/$MODEL/gof_by_source_joined.csv"

echo "Start joining gof metrics for model $MODEL"

find "$BASE_DIR" -name "gof_metrics.csv" | sort > files.txt
first_file=$(head -n 1 files.txt)
head -n 1 "$first_file" > "$OUT_FILE"

# Append data rows only
while IFS= read -r f; do
    tail -n +2 "$f"
done < files.txt >> "$OUT_FILE"

echo "Joined gof_metrics.csv files into $OUT_FILE"

# sbatch reports/join.sh
