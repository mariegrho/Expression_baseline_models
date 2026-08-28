import xarray as xr
import pandas as pd
from pathlib import Path

#data = xr.load_dataset("dataset_medina_selection_method.nc").sel(selection_method="ribo-")
#data = xr.load_dataset("dataset_medina_selection_method.nc").sel(selection_method="polyA+")

data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc")
mask = (data.tpm.max(dim="time", skipna=True) > 0.1).all(dim="source") # Keep only relevantly expressed genes
data = data.sel(ensembl_gene_id=mask)
genes = data.ensembl_gene_id.values

# labels = xr.load_dataset("data/all_gene_cluster_annotation_minmax.nc")
# genes = labels.ensembl_gene_id.values

genes.sort()
print("total genes:", len(genes))

with open('data/genes.txt', 'w+') as f:
  f.write("\n".join(genes)) 

with open("/home/student/m/mgrosseholth/projects/Expression_baseline_models/ensdarg_folders.txt") as f:
   fitted = [line.strip() for line in f]
genes_fitted = set(fitted)

print("fitted genes:",len(genes_fitted))

missing_genes = sorted(set(genes) - genes_fitted)
print(f"{len(missing_genes)} genes missing")
# with open('data/missing_genes.txt', 'w+') as f:
#     f.write("\n".join(missing_genes)) 

print("File written successfully.")

# chmod +x data/gene_list.py
# conda activate thesis
# python data/gene_list.py


'''
find results/120_hpf/Rep_Z/full -mindepth 2 -maxdepth 2 -name "numpyro_posterior.nc" \
  | sed -E 's|.*/(ENSDARG[0-9]+)/numpyro_posterior.nc|\1|' \
  | sort > ensdarg_folders.txt

'''
