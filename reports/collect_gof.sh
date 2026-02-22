#!/bin/bash
#SBATCH --job-name=gof      
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err    
#SBATCH --time=00:30:00                   
#SBATCH --mem=500MB
#SBATCH --cpus-per-task=1                 

# Base directory
BASE_DIR="/home/student/m/mgrosseholth/projects/test_sim/results/Rep_Mdecay1_nuts_8"
#BASE_DIR="/home/student/m/mgrosseholth/projects/test_sim/results/ZGA_Mdecay1_nuts_8"
#BASE_DIR="/home/student/m/mgrosseholth/projects/test_sim/results/ZGA_Mdecay_red_nuts_120"
#BASE_DIR="/home/student/m/mgrosseholth/projects/test_sim/results/Rep_Zdecay1_nuts_120"
#BASE_DIR="/home/student/m/mgrosseholth/projects/test_sim/results/basic_1s_nuts_24"


# Output file
OUT_FILE="$BASE_DIR/goodness_of_fit_summary.csv"

# Write header
echo "GeneID,NRMSE,NRMSE_lower,NRMSE_upper,LogLik,LogLik_lower,LogLik_upper,BIC,BIC_lower,BIC_upper" > "$OUT_FILE"

# Loop through ENSDARG* directories
for dir in "$BASE_DIR"/ENSDARG*/; do
    # Skip if not a directory
    [ -d "$dir" ] || continue

    FILE="${dir}goodness_of_fit.csv"
    [ -f "$FILE" ] || continue

    GENE_ID=$(basename "$dir")

    # Check if this GeneID is already in the output file
    if grep -q "^${GENE_ID}," "$OUT_FILE"; then
        echo "Skipping $GENE_ID (already processed)"
        continue
    fi

    # Extract metrics using grep + awk
    NRMSE=$(grep -m1 "^NRMSE," "$FILE" | awk -F',' '{print $2}')
    NRMSE_LOWER=$(grep -m1 "^NRMSE (95%-hdi\[lower\])," "$FILE" | awk -F',' '{print $2}')
    NRMSE_UPPER=$(grep -m1 "^NRMSE (95%-hdi\[upper\])," "$FILE" | awk -F',' '{print $2}')

    LOG_LIK=$(grep -m1 "^Log-Likelihood," "$FILE" | awk -F',' '{print $2}')
    LOG_LIK_LOWER=$(grep -m1 "^Log-Likelihood (95%-hdi\[lower\])," "$FILE" | awk -F',' '{print $2}')
    LOG_LIK_UPPER=$(grep -m1 "^Log-Likelihood (95%-hdi\[upper\])," "$FILE" | awk -F',' '{print $2}')

    BIC=$(grep -m1 "^BIC," "$FILE" | awk -F',' '{print $3}')
    BIC_LOWER=$(grep -m1 "^BIC (95%-hdi\[lower\])," "$FILE" | awk -F',' '{print $3}')
    BIC_UPPER=$(grep -m1 "^BIC (95%-hdi\[upper\])," "$FILE" | awk -F',' '{print $3}')

    # Write to output file
    echo "$GENE_ID,$NRMSE,$NRMSE_LOWER,$NRMSE_UPPER,$LOG_LIK,$LOG_LIK_LOWER,$LOG_LIK_UPPER,$BIC,$BIC_LOWER,$BIC_UPPER" >> "$OUT_FILE"
done

echo "✅ Summary with HDIs written to: $OUT_FILE"


#chmod +x collect_gof.sh
#./collect_gof.sh
# sbatch collect_gof.sh