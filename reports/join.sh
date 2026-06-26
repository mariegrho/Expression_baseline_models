#!/bin/bash
#SBATCH --job-name=join     
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err    
#SBATCH --time=01:30:00                   
#SBATCH --mem=500MB
#SBATCH --cpus-per-task=1

spack load miniconda3            
source activate thesis 
spack unload miniconda3

base_dir="results/120_hpf/"
output_file="${base_dir}/gof_metrics_joined.csv"

find "$base_dir" -name "gof_metrics.csv" | sort > files.txt
first_file=$(head -n 1 files.txt)
head -n 1 "$first_file" > "$output_file"

# Append data rows only
while IFS= read -r f; do
    tail -n +2 "$f"
done < files.txt >> "$output_file"

echo "Joined gof_metrics.csv files into $output_file"

# sbatch reports/join.sh