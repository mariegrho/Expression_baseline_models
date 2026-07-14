#!/bin/bash
#SBATCH --job-name=0gof
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=500MB
#SBATCH --cpus-per-task=1

for MODEL in Basic Rep_M Rep_Z; do

    BASE_DIR="results/120_hpf/$MODEL/all"
    OUT_FILE="results_summary/$MODEL/goodness_of_fit_summary.csv"
    FILELIST="$(mktemp)"
    DONE_GENES="$(mktemp)"

    mkdir -p "$(dirname "$OUT_FILE")"

    echo "Start gof summary with HDIs for model $MODEL"

    # Header if starting fresh
    if [ ! -s "$OUT_FILE" ]; then
        echo "gene_id,NRMSE,NRMSE_lower,NRMSE_upper,LogLik,LogLik_lower,LogLik_upper,BIC,BIC_lower,BIC_upper" > "$OUT_FILE"
    fi

    # Genes already done (O(1) lookups instead of grep -q per file)
    awk -F',' 'NR>1{print $1}' "$OUT_FILE" > "$DONE_GENES"

    # Build file list once 
    find "$BASE_DIR" -maxdepth 2 -type f -name goodness_of_fit.csv > "$FILELIST"

    awk -F',' -v done_file="$DONE_GENES" -v filelist="$FILELIST" '
    BEGIN {
        while ((getline g < done_file) > 0) seen[g] = 1
        while ((getline line < filelist) > 0) {
            ARGV[ARGC] = line
            ARGC++
        }
    }
    function flush() {
        if (gene != "" && !(gene in seen)) {
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
                gene, nrmse, nrmse_l, nrmse_u, loglik, loglik_l, loglik_u, bic, bic_l, bic_u
        }
    }
    FNR == 1 {
        flush()
        nrmse=nrmse_l=nrmse_u=loglik=loglik_l=loglik_u=bic=bic_l=bic_u=""
        n = split(FILENAME, parts, "/")
        gene = parts[n-1]
    }
    $1=="NRMSE"                              {nrmse   = $2}
    $1=="NRMSE (95%-hdi[lower])"             {nrmse_l = $2}
    $1=="NRMSE (95%-hdi[upper])"             {nrmse_u = $2}
    $1=="Log-Likelihood"                     {loglik   = $2}
    $1=="Log-Likelihood (95%-hdi[lower])"    {loglik_l = $2}
    $1=="Log-Likelihood (95%-hdi[upper])"    {loglik_u = $2}
    $1=="BIC"                                {bic   = $3}
    $1=="BIC (95%-hdi[lower])"               {bic_l = $3}
    $1=="BIC (95%-hdi[upper])"               {bic_u = $3}
    END { flush() }
    ' /dev/null >> "$OUT_FILE"

    # sort genes by GeneID
    {
        head -n 1 "$OUT_FILE"
        tail -n +2 "$OUT_FILE" | sort -t',' -k1,1
    } > "${OUT_FILE}.sorted" && mv "${OUT_FILE}.sorted" "$OUT_FILE"

    rm -f "$FILELIST" "$DONE_GENES"
    echo "✅ Summary with HDIs written to: $OUT_FILE"
done
echo "[$(date)] Finished all."

#chmod +x reports/collect_gof.sh
#./collect_gof.sh
# sbatch reports/collect_gof.sh