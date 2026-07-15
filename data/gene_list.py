import xarray as xr
import pandas as pd
from pathlib import Path

#data = xr.load_dataset("dataset_medina_selection_method.nc").sel(selection_method="ribo-")
#data = xr.load_dataset("dataset_medina_selection_method.nc").sel(selection_method="polyA+")

data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc")
mask = data.tpm.max(dim=["time", "source"], skipna=True) >= 1   # relevant expression
data = data.sel(ensembl_gene_id=mask)

genes = data.ensembl_gene_id.values
genes.sort()
print(len(genes))

with open('data/genes.txt', 'w+') as f:
    f.write("\n".join(genes)) 

model="Basic"
base_dir = f"results_summary/120_hpf/{model}/all"
#files = list(Path(base_dir).rglob("gof_metrics.csv"))
#genes_fitted = {f.parent.name for f in files}

fitted = pd.read_csv(f"results_summary/{model}/goodness_of_fit_summary.csv")
genes_fitted = set(fitted.gene_id)

missing_genes = sorted(set(genes) - genes_fitted)
print(f"{len(missing_genes)} genes missing")
with open('data/missing_genes.txt', 'w+') as f:
    f.write("\n".join(missing_genes)) 

print("File written successfully.")

# chmod +x data/gene_list.py
# conda activate thesis
# python data/gene_list.py
