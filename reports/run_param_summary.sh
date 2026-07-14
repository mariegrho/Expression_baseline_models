#!/bin/bash
#SBATCH --job-name=0params
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

for MODEL in Basic Rep_M Rep_Z; do

    BASE_DIR="results/120_hpf/$MODEL/all"
    OUT_FILE="results_summary/$MODEL/parameter_fit_summary.csv"

    mkdir -p "$(dirname "$OUT_FILE")"

    tmpdir=$(mktemp -d)          # outside BASE_DIR now
    trap 'rm -rf "$tmpdir"' EXIT

    echo "Scanning directories under: $BASE_DIR"
    find "$BASE_DIR" -type f -name "report_table_parameter_estimates.csv" > "$tmpdir/files.txt"
    total_files=$(wc -l < "$tmpdir/files.txt" | tr -d ' ')
    echo "Found $total_files CSV files."

    if [ "$total_files" -eq 0 ]; then
    echo "No CSV files found."
    exit 0
    fi

    echo "Processing files in a single awk pass ..."

    gawk -F',' -v filelist="$tmpdir/files.txt" '
    BEGIN {
        while ((getline line < filelist) > 0) {
            ARGV[ARGC] = line
            ARGC++
        }
        close(filelist)
    }
    FNR == 1 {
        # extract ENSDARG gene id from path, once per file
        if (match(FILENAME, /ENSDARG[0-9]+/)) {
            gene = substr(FILENAME, RSTART, RLENGTH)
        } else {
            gene = ""
        }
        genes[gene] = 1
        next   # skip header row ",mean ± std"
    }
    gene == "" { next }
    {
        param = $1
        gsub(/^[ \t]+|[ \t]+$/, "", param)
        if (param == "") next

        val = $2
        n = split(val, parts, "±")
        if (n != 2) next

        mean = parts[1]; std = parts[2]
        gsub(/^[ \t]+|[ \t]+$/, "", mean)
        gsub(/^[ \t]+|[ \t]+$/, "", std)

        mean_val[gene SUBSEP param] = mean
        std_val[gene SUBSEP param]  = std

        if (!(param in pseen)) { pseen[param] = 1; params[++pn] = param }
    }
    END {
        # sort parameter names for stable column order
        asort_params_n = pn
        for (i = 1; i <= pn; i++) sorted_params[i] = params[i]
        n2 = asort(sorted_params)   # gawk builtin: sorts array in place, returns count

        printf "gene_id"
        for (i = 1; i <= n2; i++) printf ",%s_mean", sorted_params[i]
        for (i = 1; i <= n2; i++) printf ",%s_std",  sorted_params[i]
        printf "\n"

        PROCINFO["sorted_in"] = "@ind_str_asc"
        for (g in genes) {
            if (g == "") continue
            printf "%s", g
            for (i = 1; i <= n2; i++) {
                key = g SUBSEP sorted_params[i]
                printf ",%s", (key in mean_val ? mean_val[key] : "")
            }
            for (i = 1; i <= n2; i++) {
                key = g SUBSEP sorted_params[i]
                printf ",%s", (key in std_val ? std_val[key] : "")
            }
            printf "\n"
        }
    }
    ' /dev/null > "$OUT_FILE"

    echo "Combined summary written to: $OUT_FILE"
done
echo "[$(date)] Finished all."

# sbatch reports/run_param_summary.sh "Rep_M"