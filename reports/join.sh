#!/bin/bash
#SBATCH --job-name=0join     
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err    
#SBATCH --time=01:00:00                   
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1

set -euo pipefail

GENE_LIST="data/genes.txt"

TMPDIR_LOCAL="results/tmp"
mkdir -p "$TMPDIR_LOCAL"
export TMPDIR="$TMPDIR_LOCAL"

for MODEL in Basic Rep_M Rep_Z Rep_V; do
    BASE_DIR="results/120_hpf/$MODEL/all"
    OUT_FILE="results/results_summary/$MODEL/gof_by_source_joined.csv"

    mkdir -p "$(dirname "$OUT_FILE")"

    echo "[$(date)] Start joining gof metrics for model $MODEL"

    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    # Build file list directly from gene list — no full-tree find, no grep loop
    while IFS= read -r gene; do
        [ -z "$gene" ] && continue
        f="$BASE_DIR/$gene/gof_metrics.csv"
        [ -f "$f" ] && echo "$f" >> "$tmpdir/filtered_files.txt"
    done < "$GENE_LIST"

    if [ ! -s "$tmpdir/filtered_files.txt" ]; then
        echo "No gof_metrics.csv files found for model $MODEL, skipping."
        rm -rf "$tmpdir"
        trap - EXIT
        continue
    fi

    sort -u "$tmpdir/filtered_files.txt" -o "$tmpdir/filtered_files.txt"

    first_file=$(head -n 1 "$tmpdir/filtered_files.txt")
    head -n 1 "$first_file" > "$OUT_FILE"

    # Append data rows only
    while IFS= read -r f; do
        tail -n +2 "$f"
    done < "$tmpdir/filtered_files.txt" >> "$OUT_FILE"

    echo "Joined gof_metrics.csv files into $OUT_FILE"

    rm -rf "$tmpdir"
    trap - EXIT
done
echo "[$(date)] Finished all."

# sbatch reports/join.sh