import xarray as xr
import pandas as pd
from pathlib import Path

#data = xr.load_dataset("dataset_medina_selection_method.nc").sel(selection_method="ribo-")
#data = xr.load_dataset("dataset_medina_selection_method.nc").sel(selection_method="polyA+")

data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc")
#mask = data.tpm.max(dim=["time", "source"], skipna=True) >= 1   # relevant expression
mask = (data.tpm.max(dim="time", skipna=True) >= 1).all(dim="source") # Keep only relevantly expressed genes
data = data.sel(ensembl_gene_id=mask)

genes = data.ensembl_gene_id.values
genes.sort()

labels = xr.load_dataset("data/all_gene_cluster_annotation_minmax.nc")
genes = labels.ensembl_gene_id.values
genes.sort()

print(len(genes))

#with open('data/genes.txt', 'w+') as f:
#    f.write("\n".join(genes)) 

model="Rep_M"
base_dir = f"results/results_summary/120_hpf/{model}/all"
#files = list(Path(base_dir).rglob("gof_metrics.csv"))
#genes_fitted = {f.parent.name for f in files}

fitted = pd.read_csv(f"results/results_summary/{model}/goodness_of_fit_summary.csv")
#fitted = pd.read_csv(f"results/results_summary/{model}/gof_by_source_joined.csv", on_bad_lines="skip")
genes_fitted = set(fitted.gene_id.unique())

missing_genes = sorted(set(genes) - genes_fitted)
print(f"{len(missing_genes)} genes missing")
with open('data/missing_genes.txt', 'w+') as f:
    f.write("\n".join(missing_genes)) 

print("File written successfully.")

# chmod +x data/gene_list.py
# conda activate thesis
# python data/gene_list.py
