# Expression_baseline_models
Five baseline expression model variants are implemented:
- Basic model
- ZGA model (M-decay, Z-decay)
- Repression model (M-decay, Z-decay)

# Model fit on HPC
To execute the model fits as arrayjob on HPC cluster run:
(example: 1000 genes, 50 at a time - adjust as needed.)
sbatch --array=1-1000%50 run_array_job_loop.sh 

# GOF evaluation
To collect the gooodness_of_fit.csv report of each model fit, run:
sbatch reports/collect_gof.sh

To calculate the gof metrics for evaluation:
sbatch --array=0-2 reports/scorecard.sh


# Post processing
To collect the fitted model parameter and trajectories, run:
sbatch --array=0-2 reports/post_process.sh 
Note: need to select the "mode" in post_process.sh first.
For model parameters: -- mode "params", for fitted trajectories: --mode "simulation"
