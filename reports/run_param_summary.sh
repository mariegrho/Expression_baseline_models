#!/bin/bash
#SBATCH --job-name=params
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Description:
# Aggregates many "report_table_parameter_estimates.csv" files into one summary CSV (serial version).

set -euo pipefail

# --- CONFIGURATION ---
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/Thresh2_nuts"
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/ZGA_Model_svi"
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/Repr2_nuts"

#base_dir="/home/student/m/mgrosseholth/projects/test_sim/output/results"
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/zga_1s_nuts_120"
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/Basic_Model_nuts"
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/ZGA_Mdecay1_nuts_8"
base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/Rep_Mdecay1_nuts_8"

#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/PolyAModel_mean_approx_White1"
#base_dir="/home/student/m/mgrosseholth/projects/test_sim/results/PolyAModel_mean_approx_Winata"

output_file="$base_dir/parameter_fit_summary.csv"

# --- TEMP SETUP ---
#tmpdir=$(mktemp -d)
tmpdir=$(mktemp -d "$base_dir/tmp.XXXXXX")

trap 'rm -rf "$tmpdir"' EXIT

echo "Scanning directories under: $base_dir"

find "$base_dir" -type f -name "report_table_parameter_estimates.csv" > "$tmpdir/files.txt"
total_files=$(wc -l < "$tmpdir/files.txt" | tr -d ' ')
echo "Found $total_files CSV files."

if [ "$total_files" -eq 0 ]; then
  echo "No CSV files found."
  exit 0
fi

# --- PROCESS FILES SERIALLY ---
echo "Reading and processing files ..."

out_raw="$tmpdir/all_raw.csv"
> "$out_raw"

while read -r f; do
  gene_id=$(echo "$f" | grep -oE 'ENSDARG[0-9]+' || true)
  [[ -z "$gene_id" ]] && continue

  # Skip header and process each row
  tail -n +2 "$f" | while IFS=, read -r param val; do
    param=$(echo "$param" | tr -d '[:space:]')
    [[ -z "$param" ]] && continue

    # Extract mean ± std
    if [[ "$val" =~ ([0-9eE\.\-]+)[[:space:]]*±[[:space:]]*([0-9eE\.\-]+) ]]; then
      mean="${BASH_REMATCH[1]}"
      std="${BASH_REMATCH[2]}"
      printf "%s,%s,%s,%s\n" "$gene_id" "$param" "$mean" "$std" >> "$out_raw"
    fi
  done
done < "$tmpdir/files.txt"

echo "Finished reading all files."

# --- BUILD PARAMETER LIST ---
awk -F, '{print $2}' "$out_raw" | sort -u > "$tmpdir/params.txt"

# --- BUILD HEADER ---
{
  printf "GeneID"
  while read -r p; do printf ",%s_mean" "$p"; done < "$tmpdir/params.txt"
  while read -r p; do printf ",%s_std" "$p"; done < "$tmpdir/params.txt"
  printf "\n"
} > "$output_file"

# --- MERGE INTO FINAL TABLE ---
awk -F, -v OFS=',' -v params_file="$tmpdir/params.txt" '
BEGIN {
  while ((getline p < params_file) > 0) {
    params[++n] = p
  }
}
{
  gene=$1; param=$2; mean=$3; std=$4
  mean_val[gene","param] = mean
  std_val[gene","param] = std
  genes[gene]=1
}
END {
  PROCINFO["sorted_in"]="@ind_str_asc"
  for (g in genes) {
    printf "%s", g
    for (i=1; i<=n; i++) printf ",%s", mean_val[g","params[i]]
    for (i=1; i<=n; i++) printf ",%s", std_val[g","params[i]]
    printf "\n"
  }
}' "$out_raw" >> "$output_file"

echo "Combined summary written to: $output_file"


## sbatch run_param_summary.sh
