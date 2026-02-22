import sys, pandas as pd, arviz as az
from pathlib import Path
import report
import numpy as np

genes_file = Path(sys.argv[1])
out_file  = Path(sys.argv[2])

rows = []

for path in genes_file.read_text().splitlines():
    gene = Path(path).name
    try:
        idata = az.from_netcdf(f"{path}/numpyro_posterior.nc")

        obs = idata.observed_data.y.values
        pred = idata.posterior_model_fits.y.mean(dim=("chain","draw")).values
        rows.append({
            "GeneID": gene,

            "Spearman": report.spearman_correlation(idata),
            "NRMSE_range": report.calc_nrmse(obs, pred)[0],
            "MASE": report.calc_mase(obs, pred),
        })

    except Exception as e:
        print(f"Skipping {gene}: {e}")

pd.DataFrame(rows).to_csv(out_file, index=False)


