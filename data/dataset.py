# prepare the datasets

import xarray as xr
import pandas as pd
from functools import lru_cache

# full dataset  of White et al 
def dataset_white():
    # read in data
    tpm_df = pd.read_csv("data/white_salmon_transcripts_tpm.csv")
    meta_df = pd.read_csv("data/samples_white_etal.csv")

    #Melt TPM df to long format
    tpm_long = tpm_df.melt(
    id_vars=["ensembl_gene_id", "ensembl_transcript_id"],
    var_name="sample_id",
    value_name="tpm_values"
    )
    #Merge with metadata
    combined_df = tpm_long.merge(meta_df, on="sample_id", how="left")

    # Sum transcript_id with the same gene_id and sample_id
    agg = combined_df.groupby(["time_hpf", "ensembl_gene_id", "sample_id"])["tpm_values"].sum().reset_index()
    dataset = agg.groupby(["time_hpf", "ensembl_gene_id"])["tpm_values"].mean().to_xarray().to_dataset(name="y").rename({"time_hpf": "time"})

    return dataset


@lru_cache(maxsize=1)
def load_white_dataset(type="mean_tpm"):
    """returns the selected data set: mean_tpm, log2, 8hpf, tpm_8hpf"""
    if type == "mean_tpm":
        return xr.load_dataset("data/white_dataset_mean.nc")
        #return xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc")
        #return xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_median.nc")
    else:
        return xr.load_dataset("data/white_dataset.nc")

# full dataset  of White et al 
def tpm_genedata_white(gene_id, dataset=None, data_type="mean_tpm"):

    if dataset is None:
        dataset = load_white_dataset(type=data_type)
    else:
        dataset = dataset

    try:
        transcript_data = dataset.y.sel(ensembl_gene_id=gene_id).to_dataset(name="y")
    except KeyError:
        raise ValueError(f"gene id {gene_id} not found in dataset.")

    return transcript_data

def prepare_dataset(gene_id, rep_data = None, obs_data=None, data_type="mean_tpm"):
    if rep_data is None:
        #rep = xr.open_dataset("data/repressor_data_raw.nc").drop_vars("ensembl_gene_id")
        rep = xr.open_dataset("data/repressor_new.nc")
    else:
        rep = rep_data
    
    if obs_data is None:
        obs = tpm_genedata_white(gene_id, data_type=data_type)
    else: 
        obs = tpm_genedata_white(gene_id, dataset=obs_data)

    rep_on_obs = rep.interp(time_rep=obs.time).drop_vars("time_rep")
    combined_ds = xr.merge([obs, rep_on_obs])

    return combined_ds

