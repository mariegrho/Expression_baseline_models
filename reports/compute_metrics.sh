#!/bin/bash
#SBATCH --job-name="metrics"
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --error=results/logs/%x_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --array=1-8
#SBATCH --mem=500MB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maryberry890@gmail.com

# --- Conda setup ---

spack load miniconda3            
source activate thesis 
spack unload miniconda3

CHUNK=500  # genes per job
START=$(( ($SLURM_ARRAY_TASK_ID-1)*CHUNK + 1 ))
END=$(( $SLURM_ARRAY_TASK_ID*CHUNK ))

#sed -n "${START},${END}p" txt_files/Repr2_metric_genes.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt
#sed -n "${START},${END}p" txt_files/Thresh2_metric_genes.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt
sed -n "${START},${END}p" txt_files/zga_z_metric_genes.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt
#sed -n "${START},${END}p" txt_files/zga_metric_genes.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt
#sed -n "${START},${END}p" txt_files/basic_metric_genes.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt

#sed -n "${START},${END}p" txt_files/polyA_metric_genes_white.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt
#sed -n "${START},${END}p" txt_files/polyA_metric_genes_win.txt > chunk_${SLURM_ARRAY_TASK_ID}.txt

python3 compute_metrics_chunk.py chunk_${SLURM_ARRAY_TASK_ID}.txt metrics_${SLURM_ARRAY_TASK_ID}.csv

# cat metrics_*.csv > results/zga_1s_nuts_120/metrics.csv
# cat metrics_*.csv > results/ZGA_Zdecay1_nuts_8/zga_metrics.csv
# cat metrics_*.csv > results/ZGA_Mdecay1_nuts_8/zga_metrics.csv
# cat metrics_*.csv > results/Rep_Mdecay1_nuts_8/Rep_metrics.csv
# cat metrics_*.csv > results/Rep_Zdecay1_nuts_8/Rep_metrics.csv

# cat metrics_*.csv > results/PolyAModel_mean_approx_White/metrics.csv
# cat metrics_*.csv > results/PolyAModel_mean_approx_White/metrics.csv

# rm metrics_*.csv chunk_*.txt
# sbatch compute_metrics.sh
# rm -rf results/logs/metr*

# find /home/student/m/mgrosseholth/projects/test_sim/results/ZGA_Mdecay1_nuts_8/ -type f -name "numpyro_posterior.nc" | sed 's/\/numpyro_posterior.nc//'  > txt_files/zga_m_metric_genes.txt
# find /home/student/m/mgrosseholth/projects/test_sim/results/Rep_Mdecay1_nuts_8/ -type f -name "numpyro_posterior.nc" | sed 's/\/numpyro_posterior.nc//'  > txt_files/rep_m_metric_genes.txt
# find /home/student/m/mgrosseholth/projects/test_sim/results/zga_1s_nuts_120/ -type f -name "numpyro_posterior.nc" | sed 's/\/numpyro_posterior.nc//'  > txt_files/zga_metric_genes.txt
# find /home/student/m/mgrosseholth/projects/test_sim/results/PolyAModel_mean_approx_White1/ -type f -name "numpyro_posterior.nc" | sed 's/\/numpyro_posterior.nc//' > txt_files/polyA_metric_genes_white.txt
# find /home/student/m/mgrosseholth/projects/test_sim/results/PolyAModel_mean_approx_Winata/ -type f -name "numpyro_posterior.nc" | sed 's/\/numpyro_posterior.nc//' > txt_files/polyA_metric_genes_win.txt