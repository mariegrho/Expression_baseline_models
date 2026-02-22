# plotting figure on hpc
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az
import xarray as xr

t_end = 8
"""
zga_m_path = f"ZGA_Mdecay_red_nuts_{t_end}"
zga_z_path = f"ZGA_Zdecay_red_nuts_{t_end}"

rep_m_path = f"Rep_Mdecay_red_nuts_{t_end}"
rep_z_path = f"Rep_Zdecay_red_nuts_{t_end}"
"""
basic_path = f"basic_1s_nuts_24"
zga_m_path = f"ZGA_Mdecay1_nuts_{t_end}"
zga_z_path = f"ZGA_Zdecay1_nuts_{t_end}"
rep_m_path = f"Rep_Mdecay1_nuts_{t_end}"
rep_z_path = f"Rep_Zdecay1_nuts_{t_end}"


nrmse = 0.2
spearman = 0.7

fig_path = f"figures_{t_end}hpf"

def merge_datasets():

    #df = pd.read_csv("dataset_structure_white_pauli_JN_BK.csv")
    df = pd.read_csv("dataset_structure_white_cluster_h.csv")
    df = df[df["cluster"] >= 0].sort_values("cluster") 

    # Basic
    basic_gof = pd.read_csv(f"results/{basic_path}/goodness_of_fit_summary.csv")
    basic_metrics = pd.read_csv(f"results/{basic_path}/basic_metrics.csv")
    basic_metrics = basic_metrics[basic_metrics["GeneID"] != "GeneID"].reset_index(drop=True)
    basic_metrics = basic_metrics.merge(basic_gof, on="GeneID")
    basic_metrics["model"] = "Basic"
    basic_metrics = basic_metrics[basic_metrics["GeneID"].isin(df.ensembl_gene_id)]

    # ZGA - M
    zga_m_gof = pd.read_csv(f"results/{zga_m_path}/goodness_of_fit_summary.csv")
    zga_m_metrics = pd.read_csv(f"results/{zga_m_path}/zga_metrics.csv")
    zga_m_metrics = zga_m_metrics[zga_m_metrics["GeneID"] != "GeneID"].reset_index(drop=True)
    zga_m_metrics = zga_m_metrics.merge(zga_m_gof, on="GeneID")
    zga_m_metrics = zga_m_metrics[zga_m_metrics["GeneID"].isin(df.ensembl_gene_id)]
    zga_m_metrics["model"] = "ZGA M-decay"

    # ZGA - Z
    zga_z_gof = pd.read_csv(f"results/{zga_z_path}/goodness_of_fit_summary.csv")
    zga_z_metrics = pd.read_csv(f"results/{zga_z_path}/zga_metrics.csv")
    zga_z_metrics = zga_z_metrics[zga_z_metrics["GeneID"] != "GeneID"].reset_index(drop=True)
    zga_z_metrics = zga_z_metrics.merge(zga_z_gof, on="GeneID")
    zga_z_metrics = zga_z_metrics[zga_z_metrics["GeneID"].isin(df.ensembl_gene_id)]
    zga_z_metrics["model"] = "ZGA Z-decay"

    # Repression - M
    rep_m_gof = pd.read_csv(f"results/{rep_m_path}/goodness_of_fit_summary.csv")
    rep_m_metrics = pd.read_csv(f"results/{rep_m_path}/Rep_metrics.csv")
    rep_m_metrics = rep_m_metrics[rep_m_metrics["GeneID"] != "GeneID"].reset_index(drop=True)
    rep_m_metrics = rep_m_metrics.merge(rep_m_gof, on="GeneID")
    rep_m_metrics["model"] = "Repression M-decay"
    rep_m_metrics = rep_m_metrics[rep_m_metrics["GeneID"].isin(df.ensembl_gene_id)]

    # Repression - Z
    rep_z_gof = pd.read_csv(f"results/{rep_z_path}/goodness_of_fit_summary.csv")
    rep_z_metrics = pd.read_csv(f"results/{rep_z_path}/Rep_metrics.csv")
    rep_z_metrics = rep_z_metrics[rep_z_metrics["GeneID"] != "GeneID"].reset_index(drop=True)
    rep_z_metrics = rep_z_metrics.merge(rep_z_gof, on="GeneID")
    rep_z_metrics["model"] = "Repression Z-decay"
    rep_z_metrics = rep_z_metrics[rep_z_metrics["GeneID"].isin(df.ensembl_gene_id)]

    combined = pd.concat([ basic_metrics, zga_m_metrics, zga_z_metrics, rep_m_metrics, rep_z_metrics]).reset_index(drop=True)
    ## combine with datastructure
    merged = combined.merge(df, left_on="GeneID", right_on="ensembl_gene_id").drop(columns="ensembl_gene_id")
    merged.to_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv", index=False)

    print("finished merging") 


def plot_model_fits_smooth(t_end=120):


    genes = {
        "4.1 " : ("ENSDARG00000104068", "gstp1"),
        #"4.2" : ("ENSDARG00000089697", "nfe2l2b"),
        "4.2" : ("ENSDARG00000042824", "nfe2l2a"),
        "4.3" : ("ENSDARG00000041569", "ces2"),
        "4.4  " : ("ENSDARG00000098315", "cyp1a"),}

    #cmap = sns.color_palette("Dark2", n_colors=5)

    col = sns.color_palette("Dark2")  
    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.flatten()

    for n, cluster in enumerate(genes):

        g_id = genes[cluster][0]

        print(g_id)
        print(genes[cluster][1])

        try:

            obs = az.from_netcdf(f"results/{zga_m_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end)).observed_data.y

            data_zga_m = xr.load_dataset(f"results/{zga_m_path}/{g_id}/posterior_predictive.nc").sel(time=slice(0, t_end))
            data_zga_z = xr.load_dataset(f"results/{zga_z_path}/{g_id}/posterior_predictive.nc").sel(time=slice(0, t_end))

            data_rep_m = xr.load_dataset(f"results/{rep_m_path}/{g_id}/posterior_predictive.nc").sel(time=slice(0, t_end))
            data_rep_z = xr.load_dataset(f"results/{rep_z_path}/{g_id}/posterior_predictive.nc").sel(time=slice(0, t_end))

            data_basic = xr.load_dataset(f"results/{basic_path}/{g_id}/posterior_predictive.nc").sel(time=slice(0, t_end))

            res_zgam = data_zga_m.mean(dim=["draw", "chain"])
            hdi_zgam = az.hdi(data_zga_m, 0.95).y

            res_zgaz = data_zga_z.mean(dim=["draw", "chain"])
            hdi_zgaz = az.hdi(data_zga_z, 0.95).y

            res_basic = data_basic.mean(dim=["draw", "chain"])
            hdi_basic = az.hdi(data_basic, 0.95).y

            res_rm = data_rep_m.mean(dim=["draw", "chain"])
            hdi_rm = az.hdi(data_rep_m, 0.95).y

            res_rz = data_rep_z.mean(dim=["draw", "chain"])
            hdi_rz = az.hdi(data_rep_z, 0.95).y

            ax[n].plot(obs.time, obs, "k", ls="", marker="o", ms=3, alpha=0.5) # obs

            ax[n].plot(res_basic.time, res_basic.y, c=color_dict["Basic"], lw=1.5, label="Basic")
            ax[n].plot(res_zgam.time, res_zgam.y, c=color_dict["ZGA M-decay"], ls="-.", lw=1.5, label="ZGA (M-decay)")
            ax[n].plot(res_zgaz.time, res_zgaz.y, c=color_dict["ZGA Z-decay"], ls=":", lw=1.5, label="ZGA (Z-decay)")
            ax[n].plot(res_rm.time, res_rm.y, c=color_dict["Repression M-decay"], ls="dashdot", lw=1.5, label="Repression (M-decay)")
            ax[n].plot(res_rz.time, res_rz.y, c=color_dict["Repression Z-decay"], ls="dotted", lw=1.5, label="Repression (Z-decay)")
                                
            ax[n].fill_between(res_basic.time, *hdi_basic.values.T, color=color_dict["Basic"], alpha=0.05,)
            ax[n].fill_between(hdi_zgam.time, *hdi_zgam.values.T, color=color_dict["ZGA M-decay"], alpha=0.05, )
            ax[n].fill_between(res_zgaz.time, *hdi_zgaz.values.T, color=color_dict["ZGA Z-decay"], alpha=0.05, )
            ax[n].fill_between(res_rm.time, *hdi_rm.values.T, color=color_dict["Repression M-decay"], alpha=0.05,)
            ax[n].fill_between(res_rz.time, *hdi_rz.values.T, color=color_dict["Repression Z-decay"], alpha=0.05,)
            
            ax[n].set(xlabel="time (hpf)", ylabel="expression (TPM)", title = f"{cluster}: {g_id} ({genes[cluster][1]})")

        except Exception as e:
            print(f"Could not process gene {g_id} in cluster {cluster}: {e}")
            continue

    # collect handles from all axes
    handles = []
    labels = []

    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # remove duplicates while preserving order
    unique = dict(zip(labels, handles))

    # place unified legend
    ax[1].legend(unique.values(), unique.keys(), bbox_to_anchor=(1,0.8),
                 ncols=1, title="Model", frameon=False)

    plt.suptitle(f"Mean model fits")
    plt.tight_layout()
    plt.savefig(f"{fig_path}/fits/Smooth_Model_fit_overview_{t_end}.png", dpi=300)

    print("finished model fits")



def plot_model_fits(t_end=120, title=""):

    genes = {
        "0" : ("ENSDARG00000002445", "prdm1a"),
        "1" : ("ENSDARG00000075113", "nanog"),
        "2" : ("ENSDARG00000116539", "ndr1-203"),
        "3" : ("ENSDARG00000002084", "lamb2"),
        "4" : ("ENSDARG00000042824", "nrf2"),
    }

    genes = {
        "0" : ("ENSDARG00000002445", "prdm1a"),
        "1" : ("ENSDARG00000008239", ""),
        "2" : ("ENSDARG00000105000", ""),
        "3" : ("ENSDARG00000002084", "lamb2"),
        "4" : ("ENSDARG00000001354", ""),
    }

    genes = {
        "4.4 " : ("ENSDARG00000098315", "cyp1a"),
        "4.4" : ("ENSDARG00000098315", "cyp1a"),
        "4.2 " : ("ENSDARG00000042824", "nrf2"),
        "4.2" : ("ENSDARG00000042824", "nrf2"),
    }

    genes = {
        "4.1 " : ("ENSDARG00000104068", "gstp1"),
        "4.2" : ("ENSDARG00000089697", "nfe2l2b"),
        "4.3" : ("ENSDARG00000042824", "ces2"),
        "4.4  " : ("ENSDARG00000098315", "cyp1a"),
        #"4.4" : ("ENSDARG00000070021", "cyp3c4"),
        #"4.4 " : ("ENSDARG00000103295", "cyp3a65"),
        #"4.2" : ("ENSDARG00000042824", "nfe2l2a"), # ENSDARG00000089697
    }

    cmap = sns.color_palette("Dark2", n_colors=5)

    #fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    for n, cluster in enumerate(genes):

        g_id = genes[cluster][0]

        print(g_id)
        print(genes[cluster][1])

        try:
            """
            if n in [1,3]:
                t_end = 24
            else:
                t_end = 120
            """
            
            data_zga_m = az.from_netcdf(f"results/{zga_m_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))
            data_zga_z = az.from_netcdf(f"results/{zga_z_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))

            data_rep_m = az.from_netcdf(f"results/{rep_m_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))
            data_rep_z = az.from_netcdf(f"results/{rep_z_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))

            data_basic = az.from_netcdf(f"results/{basic_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))

            obs = data_zga_m.observed_data.y
            res_zgam = data_zga_m.posterior_model_fits.median(dim=["draw", "chain"])
            hdi_zgam = az.hdi(data_zga_m.posterior_model_fits, 0.95).y

            res_zgaz = data_zga_z.posterior_model_fits.median(dim=["draw", "chain"])
            hdi_zgaz = az.hdi(data_zga_z.posterior_model_fits, 0.95).y

            res_basic = data_basic.posterior_model_fits.median(dim=["draw", "chain"])
            hdi_basic = az.hdi(data_basic.posterior_model_fits, 0.95).y

            res_rm = data_rep_m.posterior_model_fits.median(dim=["draw", "chain"])
            hdi_rm = az.hdi(data_rep_m.posterior_model_fits, 0.95).y

            res_rz = data_rep_z.posterior_model_fits.median(dim=["draw", "chain"])
            hdi_rz = az.hdi(data_rep_z.posterior_model_fits, 0.95).y

            ax[n].plot(obs.time, obs, "k", ls="", marker="o", ms=3, alpha=0.5) # obs

            ax[n].plot(res_zgam.time, res_zgam.y, c=cmap[0], ls="-.", lw=1.5, label="ZGA (M-decay)")
            ax[n].plot(res_zgaz.time, res_zgaz.y, c=cmap[1], ls="-.", lw=1.5, label="ZGA (Z-decay)")
            ax[n].plot(res_basic.time, res_basic.y, c=cmap[2], ls="dashed", lw=1.5, label="Basic")
            ax[n].plot(res_rm.time, res_rm.y, c=cmap[3], ls="dashdot", lw=1.5, label="Repression (M-decay)")
            ax[n].plot(res_rz.time, res_rz.y, c=cmap[4], ls="dotted", lw=1.5, label="Repression (Z-decay)")
                                
            ax[n].fill_between(obs.time, *hdi_zgam.values.T, color=cmap[0], alpha=0.05, )
            ax[n].fill_between(obs.time, *hdi_zgaz.values.T, color=cmap[1], alpha=0.05, )
            ax[n].fill_between(obs.time, *hdi_basic.values.T, color=cmap[2], alpha=0.05,)
            ax[n].fill_between(obs.time, *hdi_rm.values.T, color=cmap[3], alpha=0.05,)
            ax[n].fill_between(obs.time, *hdi_rz.values.T, color=cmap[4], alpha=0.05,)
            
            ax[n].set(xlabel="time (hpf)", ylabel="expression (TPM)", title = f"{cluster}: {g_id} ({genes[cluster][1]})")

        except Exception as e:
            print(f"Could not process gene {g_id} in cluster {cluster}: {e}")
            continue

    # collect handles from all axes
    handles = []
    labels = []

    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # remove duplicates while preserving order
    unique = dict(zip(labels, handles))

    # place unified legend
    ax[1].legend(unique.values(), unique.keys(), bbox_to_anchor=(1,0.7),
                 ncols=1, title="Model", frameon=False)

    plt.suptitle(f"Mean model fits")
    plt.tight_layout()
    plt.savefig(f"{fig_path}/fits/Model_fit_overview_{t_end}.png", dpi=300)

    print("finished model fits")


def plot_model_fits2(title=""):

    genes = {
        "4" : ("ENSDARG00000002445", "prdm1a"),
        "2" : ("ENSDARG00000075113", "nanog"),
        "0" : ("ENSDARG00000116539", "ndr1-203"),
        "3" : ("ENSDARG00000002084", "lamb2"),
        "1" : ("ENSDARG00000042824", "nrf2"),
    }

    genes = {
        "3" : ("ENSDARG00000001057", ""),
        "4" : ("ENSDARG00000114958", ""),
        "2" : ("ENSDARG00000115954", ""),
        "0" : ("ENSDARG00000002084", ""),
        "1" : ("ENSDARG00000098315", "cyp1a"),
    }

    genes = {
        "4.4" : ("ENSDARG00000098315", "cyp1a"),
        "4.2" : ("ENSDARG00000042824", "nrf2"),
    }


    cmap = sns.color_palette("Dark2", n_colors=5)

    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax = ax.flatten()

    #cluster = "2"
    g_id = genes[cluster][0]
    print(g_id)

    t_end = 120

    try:
        data_zga_m = az.from_netcdf(f"results/{zga_m_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))
        data_zga_z = az.from_netcdf(f"results/{zga_z_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))

        data_rep_m = az.from_netcdf(f"results/{rep_m_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))
        data_rep_z = az.from_netcdf(f"results/{rep_z_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))

        data_basic = az.from_netcdf(f"results/{basic_path}/{g_id}/numpyro_posterior.nc").sel(time=slice(0, t_end))


        obs = data_zga_m.observed_data.y
        res_zgam = data_zga_m.posterior_model_fits.mean(dim=["draw", "chain"])
        hdi_zgam = az.hdi(data_zga_m.posterior_model_fits, 0.95).y

        res_zgaz = data_zga_z.posterior_model_fits.mean(dim=["draw", "chain"])
        hdi_zgaz = az.hdi(data_zga_z.posterior_model_fits, 0.95).y

        res_basic = data_basic.posterior_model_fits.mean(dim=["draw", "chain"])
        hdi_basic = az.hdi(data_basic.posterior_model_fits, 0.95).y

        res_rm = data_rep_m.posterior_model_fits.mean(dim=["draw", "chain"])
        hdi_rm = az.hdi(data_rep_m.posterior_model_fits, 0.95).y

        res_rz = data_rep_z.posterior_model_fits.mean(dim=["draw", "chain"])
        hdi_rz = az.hdi(data_rep_z.posterior_model_fits, 0.95).y

        ax[0].plot(obs.time, obs, "k", ls="", marker="o", ms=3, alpha=0.5) # obs
        ax[1].plot(obs.time, obs, "k", ls="", marker="o", ms=3, alpha=0.5) # obs

        ax[0].plot(res_zgam.time, res_zgam.y, c=cmap[0], ls="-.", lw=1.5, label="ZGA (M-decay)")
        ax[0].plot(res_zgaz.time, res_zgaz.y, c=cmap[1], ls="-.", lw=1.5, label="ZGA (Z-decay)")
        ax[0].plot(res_basic.time, res_basic.y, c=cmap[2], ls="dashed", lw=1.5, label="Basic")
        
        ax[1].plot(res_basic.time, res_basic.y, c=cmap[2], ls="dashed", lw=1.5, label="Basic")
        ax[1].plot(res_rm.time, res_rm.y, c=cmap[3], ls="dashdot", lw=1.5, label="Repression (M-decay)")
        ax[1].plot(res_rz.time, res_rz.y, c=cmap[4], ls="dotted", lw=1.5, label="Repression (Z-decay)")
                            
        ax[0].fill_between(obs.time, *hdi_zgam.values.T, color=cmap[0], alpha=0.05, )
        ax[0].fill_between(obs.time, *hdi_zgaz.values.T, color=cmap[1], alpha=0.05, )
        ax[0].fill_between(obs.time, *hdi_basic.values.T, color=cmap[2], alpha=0.05,)

        ax[1].fill_between(obs.time, *hdi_basic.values.T, color=cmap[2], alpha=0.05,)
        ax[1].fill_between(obs.time, *hdi_rm.values.T, color=cmap[3], alpha=0.05,)
        ax[1].fill_between(obs.time, *hdi_rz.values.T, color=cmap[4], alpha=0.05,)
        
        ax[0].set(xlabel="time (hpf)", ylabel="expression (TPM)", title = f"{cluster}: {g_id} ({genes[cluster][1]})")
        ax[1].set(xlabel="time (hpf)", ylabel="expression (TPM)", title = f"{cluster}: {g_id} ({genes[cluster][1]})")
        ax[0].legend()
        ax[1].legend()

    except Exception as e:
        print(f"Could not process gene {g_id} in cluster {cluster}: {e}")

    #ax[1][3].legend(bbox_to_anchor=(1,spearman), ncols=1, title="model")
    plt.suptitle(f"Mean model fits")
    plt.tight_layout()
    plt.savefig(f"{fig_path}/fits/Model_fit_overview_{t_end}_{g_id}_{cluster}.png", dpi=300)

    print("finished model fits")


def plot_metrics():

    #ds = pd.read_csv("dataset_structure_white_pauli_JN_BK.csv")
    #ds = pd.read_csv("dataset_structure_white_cluster.csv")
    data = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv").sort_values("cluster")

    #df = data[(data["Spearman"] > spearman) & (data["NRMSE_range"] ) & (data["MASE"] < 1)]
    df = data

    models = ["Basic", "ZGA M-decay", "ZGA Z-decay",  "Repression M-decay",  "Repression Z-decay",] 
    cluster = df.cluster.unique()
    cluster = [0.0]
    #metrics = ["RMSLE", "Spearman", "BIC", "NRMSE_range"]
    #ylabels = ["RMSLE", "Spearman's ρ", "BIC", "NRMSE"]
    metrics = ["BIC", "MASE", "NRMSE_range", "Spearman"]
    ylabels = ["BIC", "MASE", "NRMSE",  "Spearman's ρ"]

    col = sns.color_palette("Dark2")  
    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    cluster_names = {
        0.0: "SD",
        1.0: "TU pre-ZGA",
        2.0: "TU post-ZGA",
        3.0: "SU ZGA",
        4.0: "SU post-ZGA",}

    fig, ax = plt.subplots(len(metrics), len(cluster), figsize=(len(cluster)*2, 1.5*len(metrics)), sharey="row", sharex="row")

    for r, metr in enumerate(metrics):
        for c, pat in enumerate(cluster):

            data = df[df["cluster"] == pat].copy()

            #palette = sns.color_palette("Dark2", n_colors=2)
            #palette_dict = dict(zip(df.type.unique(), palette))
            #marker_dict = {"maternal": "x", "zygotic": "o"}

            ax[2][c].axvline(x=nrmse, color='k', linestyle='-.', linewidth=0.5)
            ax[1][c].axvline(x=1, color='k', linestyle='-.', linewidth=0.5)
            ax[3][c].axvline(x=spearman, color='k', linestyle='-.', linewidth=0.5)

            sns.pointplot(
                data=data, x=metr, y="model",
                order = models,
                hue = "model",
                palette = color_dict,
                #hue = "type", hue_order=["maternal", "zygotic"],
                #markers = ["x", "o"],
                estimator="median",
                markersize=4, marker="D",
                #dodge=0.3,
                linestyle="none",
                errorbar = ('pi', 90), capsize=.2,
                #palette=palette_dict,
                ax=ax[r][c],
                legend = False,
                #legend = True if c == 0 and r == 0 else False,
                err_kws = {"alpha":0.6,'linewidth': 1.5,},
            )

            ax[r][c].set(xlabel="", ylabel=ylabels[r])
            ax[r][c].grid(True)
            ax[0][c].set_title(f"({int(pat)}) {cluster_names[pat]}", fontsize=10) 
            #ax[2][c].set_yticks([1, 0.5, 0, -0.5, -1])
            #ax[0][len(cluster)-1].legend(loc=(1.02, 0.3), title="purely zygotic")
            #ax[r][0].set(ylabel=ylabels[r])
            #ax[r][c].tick_params(axis='x', labelrotation=90)
            ax[r][c].set_yticks(models)
            ax[r][c].set_yticklabels(["Basic", "ZGA-M", "ZGA-Z",  "Repression-M",  "Repression-Z",], fontsize=8)

    #legend_handles = [
    #    Line2D([0], [0], marker=marker_dict[t], color=palette_dict[t],
    #        linestyle="none", markersize=8, label=t)
    #    for t in ["maternal", "zygotic"] ]
    #fig.legend(handles=legend_handles, bbox_to_anchor=(1.0, 0.99), frameon=False)

    #plt.suptitle("Model performance of maternal and zygotic genes \n ")
    plt.suptitle(f"Model performance across clusters ({t_end} hpf)", fontsize=11)
    plt.tight_layout()
    #plt.savefig("{fig_path}/model_cluster_comparision_metrics_median_GOF.png")
    plt.savefig(f"{fig_path}/model_cluster_comparision_metrics_median_all.png")
    plt.show()

    print("finished metrics")

def plot_metrics_violin():

    #ds = pd.read_csv("dataset_structure_white_pauli_JN_BK.csv")
    #ds = pd.read_csv("dataset_structure_white_cluster.csv")
    data = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv").sort_values("cluster")
    df = data[(data["Spearman"] > spearman) & (data["NRMSE_range"] < nrmse) & (data["MASE"] < 1)]
    df.cluster.unique()

    models = ["Basic", "ZGA M-decay", "ZGA Z-decay",  "Repression M-decay",  "Repression Z-decay",] 
    cluster = df.cluster.unique()
    #metrics = ["RMSLE", "Spearman", "BIC", "NRMSE_range"]
    #ylabels = ["RMSLE", "Spearman's ρ", "BIC", "NRMSE"]
    metrics = ["NRMSE_range", "MASE",  "Spearman", "BIC",]
    ylabels = ["NRMSE",  "MASE", "Spearman's ρ", "BIC",]

    cluster_names = {
        0.0: "SD",
        1.0: "TU pre-ZGA",
        2.0: "TU post-ZGA",
        3.0: "SU ZGA",
        4.0: "SU post-ZGA",}

    fig, ax = plt.subplots(len(metrics), len(cluster), figsize=(len(cluster)*3, 3*len(metrics)), sharey="row", sharex=True)

    for r, metr in enumerate(metrics):
        for c, pat in enumerate(cluster):

            data = df[df["cluster"] == pat].copy()
            data = data[(data["NRMSE_range"] < 1) & (data["MASE"] < 2) ]

            # restrict model categories for correct x-axis behavior
            valid_models = models
            dodge = 0.3

            palette = sns.color_palette("Dark2", n_colors=2)
            palette_dict = dict(zip(df.type.unique(), palette))

            marker_dict = {"maternal": "x", "zygotic": "o"}

            sns.violinplot(
                data=data, x="model", y=metr,
                order = valid_models,
                hue = "type", hue_order=["maternal", "zygotic"],
                #estimator="median",
                dodge=dodge,
                palette=palette_dict,
                alpha = 0.5,
                ax=ax[r][c],
                legend = False,
                #legend = True if c == 0 and r == 0 else False,
                #err_kws = {"alpha":0.6,'linewidth': 1.5,},
            )

            ax[0][c].axhline(y=0.2, color='k', linestyle='--', linewidth=0.5)
            ax[1][c].axhline(y=1, color='k', linestyle='--', linewidth=0.5)
            ax[2][c].axhline(y=spearman, color='k', linestyle='--', linewidth=0.5)

            #ax[2][c].set_yticks([1, 0.5, 0, -0.5, -1])
            ax[r][0].set(ylabel=ylabels[r])
            ax[r][c].set_xticks(models)
            ax[r][c].set_xticklabels(["Basic", "ZGA-M", "ZGA-Z",  "Repression-M",  "Repression-Z",], rotation=90)

            ax[r][c].set(xlabel="")
            ax[r][c].grid(True)
            ax[0][c].set(title=f"({int(pat)}) {cluster_names[pat]}") # (n={len(data)})")

    legend_handles = [
        Line2D([0], [0], marker=marker_dict[t], color=palette_dict[t],
            linestyle="none", markersize=8, label=t)
        for t in ["maternal", "zygotic"] ]
    fig.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(1, 0.99))

    plt.suptitle("Model performance of maternal and zygotic genes \n ")
    plt.tight_layout()
    plt.savefig(f"{fig_path}/model_cluster_comparision_metrics_violin_GOF.png")
    plt.show()

    print("finished metrics")



def plot_parameters():
    ## only accepted genes:
    df = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    df_filtered = df[(df["Spearman"] > spearman) & (df["NRMSE_range"] < nrmse) & (df["MASE"] < 1.0)]

    print(df_filtered.value_counts("model"))
    basic_genes = df_filtered[df_filtered["model"] == "Basic"].GeneID.unique()
    zga_m_genes = df_filtered[df_filtered["model"] == "ZGA M-decay"].GeneID.unique()
    zga_z_genes = df_filtered[df_filtered["model"] == "ZGA Z-decay"].GeneID.unique()
    rep_m_genes = df_filtered[df_filtered["model"] == "Repression M-decay"].GeneID.unique()
    rep_z_genes = df_filtered[df_filtered["model"] == "Repression Z-decay"].GeneID.unique()

    data = pd.read_csv("dataset_structure_white_cluster_h.csv") #dtype={8: str}
    #data = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv",  ) #dtype={8: str}

    data = data[data["cluster"].isna() == False]
    ds = data[['ensembl_gene_id','cluster','tpm_level', "type"]]

    basic_params = pd.read_csv(f"results/{basic_path}/parameter_fit_summary.csv",)
    basic_params_merge = basic_params.merge(ds, left_on="GeneID", right_on="ensembl_gene_id", how="left").drop(columns=["ensembl_gene_id"])
    basic_params_merge = basic_params_merge[basic_params_merge["GeneID"].isin(basic_genes)]

    zga_params = pd.read_csv(f"results/{zga_m_path}/parameter_fit_summary.csv")
    zga_params_merge = zga_params.merge(ds, left_on="GeneID", right_on="ensembl_gene_id", how="left").drop(columns=["ensembl_gene_id"])
    zga_params_merge = zga_params_merge[zga_params_merge["GeneID"].isin(zga_m_genes)]

    zga_paramsz = pd.read_csv(f"results/{zga_z_path}/parameter_fit_summary.csv")
    zga_params_mergez = zga_paramsz.merge(ds, left_on="GeneID", right_on="ensembl_gene_id", how="left").drop(columns=["ensembl_gene_id"])
    zga_params_mergez = zga_params_mergez[zga_params_mergez["GeneID"].isin(zga_z_genes)]

    r_params = pd.read_csv(f"results/{rep_m_path}/parameter_fit_summary.csv")
    r_params_merge = r_params.merge(ds, left_on="GeneID", right_on="ensembl_gene_id", how="left").drop(columns=["ensembl_gene_id"])
    r_params_merge = r_params_merge[r_params_merge["GeneID"].isin(rep_m_genes)]

    r_paramsz = pd.read_csv(f"results/{rep_z_path}/parameter_fit_summary.csv")
    r_params_mergez = r_paramsz.merge(ds, left_on="GeneID", right_on="ensembl_gene_id", how="left").drop(columns=["ensembl_gene_id"])
    r_params_mergez = r_params_mergez[r_params_mergez["GeneID"].isin(rep_z_genes)]


    def plot_params_cluster(ds_params_merge, params, title, model_name):

        palette = sns.color_palette("Dark2", n_colors=5)
        palette_dict = dict(zip(ds.cluster.unique(), palette))

        color_dict = {0.0: "tab:red",
                        1.0: "tab:blue",
                        2.0: "tab:green", 
                        3.0: "tab:purple",
                        4.0: "tab:orange"}

        fix, ax = plt.subplots(1, len(params), figsize = (len(params)*3, 3))

        if "beta_mean" in params:
            data = ds_params_merge[ds_params_merge["beta_mean"] > 0.0]
        else:
            data = ds_params_merge

        for i, param in enumerate(params):
            sns.kdeplot(data=data, x=param, log_scale=True, 
                        hue="cluster", 
                        palette=color_dict,
                        common_norm=False,
                        legend= True if i == len(params)-1 else False,
                        fill=False, 
                        ax=ax[i])
            
            ax[i].set(title=title[i], xlabel=title[i])
        sns.move_legend(ax[-1], loc=(1.02, 0.3))
        plt.suptitle(model_name)
        plt.tight_layout()
        plt.savefig(f"{fig_path}/{model_name}_parameter_fit_distribution_cluster.png")
        plt.show()

    model_name="Repression M-decay"
    ds_params_merge = r_params_merge
    params = ["delta_m_mean", "delta_z_mean", "beta_mean",  "alpha_mean", "t_zga_mean", "dt_rep_mean"]
    title = [r"$\delta_m$", r"$\delta_z$", r"$\beta$", r"$\alpha$", r"$t_{zga}$", r"$t_{rep}$"]
    plot_params_cluster(ds_params_merge, params, title, model_name)

    model_name="Repression Z-decay"
    ds_params_merge = r_params_mergez
    params = ["delta_z_mean", "beta_mean",  "alpha_mean", "t_zga_mean", "dt_rep_mean"]
    title = [ r"$\delta_z$", r"$\beta$", r"$\alpha$", r"$t_{zga}$", r"$t_{rep}$"]
    plot_params_cluster(ds_params_merge, params, title, model_name)

    ## Basic
    model_name="Basic"
    ds_params_merge = basic_params_merge
    params = ["beta_mean", "delta_mean"]
    title = [r"$\beta$", r"$delta_z$"]
    plot_params_cluster(ds_params_merge, params, title, model_name)

    ## ZGA
    model_name = "ZGA M-decay"
    ds_params_merge = zga_params_merge
    params = ["delta_m_mean","delta_z_mean", "beta_mean", "t_zga_mean"]
    title = [ r"$\delta_m$", r"$\delta_z$",r"$\beta$", r"$t_{zga}$"]
    plot_params_cluster(ds_params_merge, params, title, model_name)

    ## ZGA
    model_name = "ZGA Z-decay"
    ds_params_merge = zga_params_mergez
    params = ["delta_z_mean", "beta_mean", "t_zga_mean"]
    title = [ r"$\delta_z$",r"$\beta$", r"$t_{zga}$"]
    plot_params_cluster(ds_params_merge, params, title, model_name)


    def params_multiplot(ds_params_merge, cluster, params, title, model_name, hue="type"):

        if hue == "type":
            palette = sns.color_palette("husl", n_colors=2)
            palette_dict = dict(zip(ds_params_merge.type.unique(), palette))
            order = ["maternal", "zygotic"]

        if hue == "tpm_level":
            palette = sns.color_palette("Dark2", n_colors=3)
            palette_dict = dict(zip(ds.tpm_level.unique(), palette))
            order = ["low", "medium", "high"]

        data1 = ds_params_merge[ds_params_merge["beta_mean"] > 0.0]

        fig, ax = plt.subplots(len(params), len(cluster), figsize=(3*len(cluster),3*len(params)), sharex="row")

        for p, pat in enumerate(cluster):
            for i, param in enumerate(params):

                data = data1[data1["cluster"] == pat]

                sns.kdeplot(data=data, x=param, log_scale=True, 
                            hue=hue, hue_order=order,
                            ax=ax[i][p], 
                            fill=True, alpha=0.05,
                            legend=True if (i==0) &(p==4) else False, 
                            palette=palette_dict,
                            )
                
                ax[i][p].set(xlabel=title[i])
            ax[0][p].set(title=f"cluster {pat}")

            if model_name == "ZGA":
                ax[2][p].set(xlim=(1e-1,120))
        sns.move_legend(ax[0][4], loc=(1.02, 0.3))
        plt.suptitle(f"{model_name} model - parameter fit distribution")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/{model_name}_parameter_fit_distribution_{hue}.png")
        plt.show()


    ## Basic
    model_name="Basic"
    ds_params_merge = basic_params_merge
    cluster = [0, 1, 2, 3, 4]
    params = ["delta_mean", "beta_mean"]
    title = [r"$\delta$",r"$\beta$"]
    #params = ["delta_mean", "beta_mean", "t_zga_mean"]
    #title = [ r"$\delta$", r"$\beta$", r"$t_{zga}$"]

    params_multiplot(ds_params_merge, cluster, params, title, model_name, "type")
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "tpm_level")

    ## ZGA
    model_name = "ZGA M-decay"
    ds_params_merge = zga_params_merge
    params = ["delta_m_mean","delta_z_mean", "beta_mean", "t_zga_mean"]
    title = [ r"$\delta_m$", r"$\delta_z$", r"$\beta$", r"$t_{zga}$"]
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "type")
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "tpm_level")
    
    ## ZGA
    model_name = "ZGA Z-decay"
    ds_params_merge = zga_params_mergez
    params = ["delta_z_mean", "beta_mean", "t_zga_mean"]
    title = [ r"$\delta_z$",r"$\beta$", r"$t_{zga}$"]
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "type")
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "tpm_level")


    model_name="Repression M-decay"
    ds_params_merge = r_params_merge
    cluster = [0, 1, 2, 3, 4]
    params = ["delta_m_mean", "delta_z_mean", "beta_mean", "t_zga_mean", "dt_rep_mean"]
    title = [ r"$\delta_m$", r"$\delta_z$", r"$\beta$", r"$t_{zga}$", r"$t_{rep}$"]
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "type")
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "tpm_level")

    model_name="Repression Z-decay"
    ds_params_merge = r_params_mergez
    cluster = [0, 1, 2, 3, 4]
    params = ["delta_z_mean", "beta_mean", "t_zga_mean", "dt_rep_mean"]
    title = [ r"$\delta_z$", r"$\beta$", r"$t_{zga}$", r"$t_{rep}$"]
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "type")
    params_multiplot(ds_params_merge, cluster, params, title, model_name, "tpm_level")

    def box_plot():

        # basic
        fig, ax = plt.subplots(1,2, figsize=(12, 4), sharey=False)
        #sns.boxplot(data=basic_params_merge, x="cluster", y="delta_m_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)
        sns.boxplot(data=basic_params_merge, x="cluster", y="delta_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)

        #sns.boxplot(data=basic_params_merge, x="cluster", y="delta_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)
        sns.boxplot(data=basic_params_merge, x="cluster", y="beta_mean", hue="type", gap=.1, ax=ax[1]  , legend=True, fliersize=2)
        #sns.boxplot(data=basic_params_merge, x="cluster", y="t_zga_mean", hue="type", gap=.1, ax=ax[2], legend=True, fliersize=2)

        #ax[0].set(title="delta_m (mean)", ylabel="", yscale="log")
        ax[0].set(title="delta_z (mean)", ylabel="", yscale="log")
        ax[1].set(title="beta (mean)", ylabel="", yscale="log")
        #ax[2].set(title="t_zga (mean)", ylabel="",  ylim=(0.001,120))

        sns.move_legend(ax[-1], "best")
        plt.suptitle("Basic model parameters by cluster and type")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/Basic_parameters.png")
        #plt.show()

        # repression
        fig, ax = plt.subplots(1,5, figsize=(16, 4))
        #.boxplot(data=r_params_merge, x="cluster", y="delta_r_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)
        sns.boxplot(data=r_params_merge, x="cluster", y="delta_z_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)
        sns.boxplot(data=r_params_merge, x="cluster", y="beta_mean", hue="type", gap=.1, ax=ax[1],legend=False, fliersize=2)
        sns.boxplot(data=r_params_merge, x="cluster", y="alpha_mean", hue="type", gap=.1, ax=ax[2],legend=False, fliersize=2)
        sns.boxplot(data=r_params_merge, x="cluster", y="t_zga_mean", hue="type", gap=.1, ax=ax[3], legend=False, fliersize=2)
        sns.boxplot(data=r_params_merge, x="cluster", y="dt_rep_mean", hue="type", gap=.1, ax=ax[4], legend=True,fliersize=2)

        #ax[0].set(title="delta_r (mean)", ylabel="", yscale="log")
        ax[0].set(title="delta_z (mean)", ylabel="", yscale="log")
        ax[1].set(title="beta (mean)", ylabel="", yscale="log")
        ax[2].set(title="alpha (mean)", ylabel="", yscale="log")
        ax[3].set(title="t_zga (mean)", ylabel="",  ylim=(0.001, 20))
        ax[4].set(title="dt_rep (mean)", ylabel="", ylim=(0.001, 50))

        sns.move_legend(ax[-1], "best")
        plt.suptitle("Repression model parameters by cluster and type")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/Repression_parameters.png")
        #plt.show()

        # ZGA
        fig, ax = plt.subplots(1,3, figsize=(16, 4))
        #sns.boxplot(data=zga_params_merge, x="cluster", y="delta_r_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)
        sns.boxplot(data=zga_params_merge, x="cluster", y="delta_z_mean", hue="type", gap=.1, ax=ax[0], legend=False, fliersize=2)
        sns.boxplot(data=zga_params_merge, x="cluster", y="beta_mean", hue="type", gap=.1, ax=ax[1],legend=False, fliersize=2)
        sns.boxplot(data=zga_params_merge, x="cluster", y="t_zga_mean", hue="type", gap=.1, ax=ax[2], legend=True, fliersize=2)

        #ax[0].set(title="delta_r (mean)", ylabel="", yscale="log")
        ax[0].set(title="delta_z (mean)", ylabel="", yscale="log")
        ax[1].set(title="beta (mean)", ylabel="", yscale="log")
        ax[2].set(title="t_zga (mean)", ylabel="",  ylim=(0.001, 70))

        sns.move_legend(ax[-1], "best")
        plt.suptitle("ZGA model parameters by cluster and type")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/ZGA_parameters.png")
        #plt.show()


    def scatter_plot():
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(4,5, figsize=(12, 10), sharex=True, sharey=True)
        #fig, ax = subplots_with_row_titles(3, 4, row_titles=["Basic", "ZGA", "Threshold"], subplot_kw={"figsize":(12,10)}, sharex=True, sharey=True)

        palette = sns.color_palette("Dark2", n_colors=ds.type.nunique())
        palette_dict = dict(zip(ds.type.unique(), palette))

        for p, pat in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
            data_basic = basic_params_merge[basic_params_merge["cluster"] == pat]
            data_zga = zga_params_merge[zga_params_merge["cluster"] == pat]
            data_tm = tm_params_merge[tm_params_merge["cluster"] == pat]
            data_r = r_params_merge[r_params_merge["cluster"] == pat]

            sns.scatterplot(data=data_basic, x="delta_z_mean",  y="beta_mean", ax=ax[0][p], s=10, hue="type", palette=palette_dict, legend=True if p==4 else False)
            sns.scatterplot(data=data_zga, x="delta_z_mean",  y="beta_mean", ax=ax[1][p], s=10, hue="type", palette=palette_dict, legend=False)
            sns.scatterplot(data=data_tm, x="delta_z_mean",  y="beta_mean", ax=ax[2][p], s=10, hue="type", palette=palette_dict, legend=False)
            sns.scatterplot(data=data_r, x="delta_z_mean",  y="beta_mean", ax=ax[3][p], s=10, hue="type", palette=palette_dict, legend=False)

            for i in range(4):
                #ax[i][p].vlines(x=1, ymin=0, ymax=3500, colors="k", linestyles="dashed", lw=1)
                #ax[i][p].hlines(y=1, xmin=0, xmax=5000, colors="k", linestyles="dashed", lw=1)
                ax[i][p].set_yscale("log")
                ax[i][p].set_xscale("log")
                ax[i][p].set(title=pat, xlabel="delta_z", ylabel="beta", )#xlim=(0.01, 1), ylim=(0.0001, 5000))


            #ax[1][p].vlines(x=0.126, ymin=0, ymax=3500, colors="grey", linestyles="-", lw=1)
            # ax[2][p].vlines(x=0.126, ymin=0, ymax=3500, colors="grey", linestyles="-", lw=1)

        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

        ax[0][0].text(0.05,1.05, "Basic", transform=ax[0][0].transAxes, bbox=props)
        ax[1][0].text(0.05,1.05, "ZGA",  transform=ax[1][0].transAxes,bbox=props)
        ax[2][0].text(0.05,1.05, "Threshold", transform=ax[2][0].transAxes,bbox=props)
        ax[3][0].text(0.05,1.05, "Repression", transform=ax[3][0].transAxes,bbox=props)

        plt.suptitle("mean model parameter comparison")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/scatter_parameters.png")
        plt.show()

    #box_plot()
    #scatter_plot()

    print("finished params")

def plot_venn():

    import pandas as pd

    col = sns.color_palette("Set2")  
    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    # filter for best fitted genes
    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    filtered = merged[(merged["Spearman"] > spearman) & (merged["NRMSE_range"] < nrmse) & (merged["MASE"] < 1.0)]

    #print(filtered.value_counts("model"))
    zga_m = set(filtered.loc[filtered["model"] == "ZGA M-decay", "GeneID"])
    zga_z = set(filtered.loc[filtered["model"] == "ZGA Z-decay", "GeneID"])
    rep_m = set(filtered.loc[filtered["model"] == "Repression M-decay", "GeneID"])
    rep_z = set(filtered.loc[filtered["model"] == "Repression Z-decay", "GeneID"])

    basic = set(filtered.loc[filtered["model"] == "Basic", "GeneID"])

    # venn diagramm
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn3

    ## venn 2
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    venn2([basic, zga_m], set_labels=["Basic", "ZGA-M"], ax=axs[0], set_colors=(color_dict["Basic"], color_dict["ZGA M-decay"]))
    venn2([basic, zga_z], set_labels=["Basic", "ZGA-Z"], ax=axs[1], set_colors=(color_dict["Basic"], color_dict["ZGA Z-decay"]))
    venn2([basic, rep_m], set_labels=["Basic", "Repression-M"], ax=axs[2], set_colors=(color_dict["Basic"], color_dict["Repression M-decay"]))
    venn2([basic, rep_z], set_labels=["Basic", "Repression-Z"], ax=axs[3], set_colors=(color_dict["Basic"], color_dict["Repression Z-decay"]))

    axs[0].set(title=f"Basic (n={len(basic)}) &\n ZGA-M (n={len(zga_m)})")
    axs[1].set(title=f"Basic (n={len(basic)}) &\n ZGA-Z (n={len(zga_z)})")
    axs[2].set(title=f"Basic (n={len(basic)}) &\n Repression-M (n={len(rep_m)})")
    axs[3].set(title=f"Basic (n={len(basic)}) &\n Repression-Z (n={len(rep_z)})")

    plt.suptitle("pairwise comparison of accepted fits \n ")
    plt.tight_layout()
    plt.savefig(f"{fig_path}/venn_pairwise1.png")
    plt.show()

        ## venn 2
    fig, axs = plt.subplots(1, 2, figsize=(6, 4))

    venn2([zga_z, zga_m], set_labels=["ZGA-Z", "ZGA-M"], ax=axs[0], set_colors=("tab:blue", "tab:green"))
    venn2([rep_z, rep_m], set_labels=["Repression-Z", "Repression-M"], ax=axs[1], set_colors=("tab:red", "tab:orange"))

    axs[0].set(title=f"ZGA-Z (n={len(zga_z)}) &\n ZGA-M (n={len(zga_m)})")
    axs[1].set(title=f"Repression-Z (n={len(rep_z)}) &\n Repression-M (n={len(rep_m)})")

    plt.suptitle("pairwise comparison of accepted fits \n ")
    plt.tight_layout()
    plt.savefig(f"{fig_path}/venn_pairwise2.png")
    plt.show()
    
    print("finished venn")



def plot_venn_cluster():

    import pandas as pd
    # venn diagramm
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn3

    col = sns.color_palette("Set2")  
    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    # filter for best fitted genes
    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    filtered1 = merged[(merged["Spearman"] > spearman) & (merged["NRMSE_range"] < nrmse) & (merged["MASE"] < 1.0)]

    print(f"Overall accepted fits: {len(filtered1.GeneID.unique())}")
    print(f" % : {len(filtered1.GeneID.unique()) / len(merged.GeneID.unique())} \n--------")

    print(filtered1.value_counts("model"))
        
    for c in range(5):
        
        filtered = filtered1[filtered1["cluster"] == c]
        #print(filtered.value_counts("model"))
        
        zga_m = set(filtered.loc[filtered["model"] == "ZGA M-decay", "GeneID"])
        zga_z = set(filtered.loc[filtered["model"] == "ZGA Z-decay", "GeneID"])
        rep_m = set(filtered.loc[filtered["model"] == "Repression M-decay", "GeneID"])
        rep_z = set(filtered.loc[filtered["model"] == "Repression Z-decay", "GeneID"])
        basic = set(filtered.loc[filtered["model"] == "Basic", "GeneID"])


        ## venn 2
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))

        venn2([basic, zga_m], set_labels=["Basic", "ZGA-M"], ax=axs[0], set_colors=(color_dict["Basic"], color_dict["ZGA M-decay"]))
        venn2([basic, zga_z], set_labels=["Basic", "ZGA-Z"], ax=axs[1], set_colors=(color_dict["Basic"], color_dict["ZGA Z-decay"]))
        venn2([basic, rep_m], set_labels=["Basic", "Repression-M"], ax=axs[2], set_colors=(color_dict["Basic"], color_dict["Repression M-decay"]))
        venn2([basic, rep_z], set_labels=["Basic", "Repression-Z"], ax=axs[3], set_colors=(color_dict["Basic"], color_dict["Repression Z-decay"]))

        axs[0].set(title=f"Basic (n={len(basic)}) &\n ZGA-M (n={len(zga_m)})")
        axs[1].set(title=f"Basic (n={len(basic)}) &\n ZGA-Z (n={len(zga_z)})")
        axs[2].set(title=f"Basic (n={len(basic)}) &\n Repression-M (n={len(rep_m)})")
        axs[3].set(title=f"Basic (n={len(basic)}) &\n Repression-Z (n={len(rep_z)})")

        plt.suptitle(f"Accepted fits, cluster {c} (Spearman > {spearman}, NRMSE < {nrmse})")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/venn_pairwise_cluster{c}.png")

        ## venn 2
        fig, axs = plt.subplots(1, 2, figsize=(6, 4))

        venn2([zga_z, zga_m], set_labels=["ZGA Z-decay", "ZGA M-decay"], ax=axs[0], set_colors=("tab:blue", "tab:green"))
        venn2([rep_z, rep_m], set_labels=["Repression Z-decay", "Repression M-decay"], ax=axs[1], set_colors=("tab:red", "tab:orange"))

        axs[0].set(title=f"ZGA-Z (n={len(zga_z)}) & \n ZGA-M (n={len(zga_m)})")
        axs[1].set(title=f"Repression-Z (n={len(rep_z)})& \n Repression-M (n={len(rep_m)})")

        plt.suptitle(f"Accepted fits, cluster {c} (Spearman > {spearman}, NRMSE < {nrmse})")
        plt.tight_layout()
        plt.savefig(f"{fig_path}/venn_pairwise2_cluster{c}.png")
        plt.show()
        plt.close()

    print("finished venn_cluster")

def plot_acceptedfits():

    order = ["Basic", "ZGA M-decay", "ZGA Z-decay",  "Repression M-decay",  "Repression Z-decay",] 
    label= []

    print("\n--- Start accepted fits, count plot --- \n")

    cluster_names = {
        0.0: "SD",
        1.0: "TU pre-ZGA",
        2.0: "TU post-ZGA",
        3.0: "SU ZGA",
        4.0: "SU post-ZGA",}

    col = sns.color_palette("Set2")  
    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    filtered1 = merged[(merged["Spearman"] > spearman) & (merged["NRMSE_range"] < nrmse) & (merged["MASE"] < 1.0)]

    fig, ax = plt.subplots(1, 5, figsize=(10, 3))

    for c in range(5):
        filtered = filtered1[filtered1["cluster"] == c]
        print(f"cluster {c}: ", filtered.value_counts("model"))

        sns.countplot(filtered, x="model", hue="model", palette=color_dict, ax=ax[c], order=order)
        ax[c].set(title=f"({c}) {cluster_names[c]}", xlabel="", ylabel="")
        ax[c].set_xticks(order)
        ax[c].set_xticklabels(["Basic", "ZGA-M", "ZGA-Z",  "Repression-M",  "Repression-Z",], rotation=90)

    plt.suptitle(f"Accepted fits per cluster (MASE < 1.0, Spearman > {spearman}, NRMSE < {nrmse})", fontsize='x-large', )
    plt.tight_layout()
    plt.savefig(f"{fig_path}/countplot_cluster2.png")
    plt.show()
    

def plot_accepted_heatmap():

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    order = ["Basic", "ZGA M-decay", "ZGA Z-decay",  "Repression M-decay",  "Repression Z-decay",] 
    label = ["(0) SD", "(1) TU pre-ZGA", "(2) TU post-ZGA", "(3) SU ZGA", "(4) SU post-ZGA"]
    cluster_names = {
        0: "SD",
        1: "TU pre-ZGA",
        2: "TU post-ZGA",
        3: "SU ZGA",
        4: "SU post-ZGA",}

    # Load original data
    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    merged = merged[merged["model"] != "Degradation"]
    accepted = merged[(merged["Spearman"] > spearman) & (merged["NRMSE_range"] < nrmse) & (merged["MASE"] < 1.0)]

    count_model = accepted.model.value_counts()
    count_cluster = [accepted[accepted["cluster"] == 0].GeneID.unique().size,
                    accepted[accepted["cluster"] == 1].GeneID.unique().size,
                    accepted[accepted["cluster"] == 2].GeneID.unique().size,
                    accepted[accepted["cluster"] == 3].GeneID.unique().size,
                    accepted[accepted["cluster"] == 4].GeneID.unique().size, ]

    merged_counts = merged.groupby(["model", "cluster"]).size().reset_index(name="merged")
    accepted_counts = accepted.groupby(["model", "cluster"]).size().reset_index(name="accepted")
    result = pd.merge(merged_counts, accepted_counts, how="left", on=["model", "cluster"])
    result["accepted"] = result["accepted"].fillna(0).astype(int)
    result["ratio"] = result["accepted"] / result["merged"] * 100

    # Pivot for heatmap
    pivot = result.pivot(index="cluster", columns="model", values="ratio") 
    pivot = pivot[order]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    #cmap = sns.color_palette("blend:#ffd3b6,#dcedc1", as_cmap=True)
    cmap = sns.color_palette("blend:#f37736,#fdf498,#7bc043", as_cmap=True)
    #sns.heatmap(pivot, annot=True, linewidths=0.5,  cmap="YlGnBu", cbar_kws={'label': '% accepted', 'shrink': 0.6, 'pad': 0.02 })
    sns.heatmap(pivot, annot=True, linewidths=0.5,  cmap=cmap, alpha=0.8, cbar_kws={'label': '% accepted', 'shrink': 0.75, 'pad': 0.02 })

    ax.set(title="Accepted fits per cluster", xlabel="Model", ylabel="Cluster")
    ax.set_xticklabels([f"Basic\n({count_model['Basic']})", 
                        f"ZGA-M\n({count_model['ZGA M-decay']})", f"ZGA-Z\n({count_model['ZGA Z-decay']})",  
                        f"Repression-M\n({count_model['Repression M-decay']})",  f"Repression-Z\n({count_model['Repression Z-decay']})"])
    ax.set_yticklabels([f"{label[0]} \n ({count_cluster[0]})", 
                        f"{label[1]} \n ({count_cluster[1]})", f"{label[2]} \n ({count_cluster[2]})",  
                        f"{label[3]} \n ({count_cluster[3]})", f"{label[4]} \n ({count_cluster[4]})"], 
                        rotation=0)
    #ax.set_yticklabels(label, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/heatmap_accepted_cluster.png")
    plt.show()


def plot_metrics_distribution1():

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    merged = merged[merged["NRMSE_range"] < 1]

    color_dict = {'SD': "tab:red",
              'TU pre-ZGA': "tab:blue",
              'TU post-ZGA': "tab:green", 
              'SU ZGA': "tab:purple",
              'SU post-ZGA': "tab:orange"}
    
    order = ["SD", "TU pre-ZGA", "TU post-ZGA", "SU ZGA", "SU post-ZGA"]

    for model in merged.model.unique():
        data = merged[merged["model"] == model]

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

        sns.histplot(data, x="MASE", bins=50, ax=ax0, hue="cluster_name", hue_order=order, palette=color_dict, legend=False)
        ax0.set_xlabel("MASE")
        ax0.axvline(x=1, color="k", ls="dashed", lw=1.5)

        sns.histplot(data, x="NRMSE_range", bins=50, ax=ax1, hue="cluster_name", hue_order=order, palette=color_dict, legend=False)
        ax1.set_xlabel("NRMSE")
        ax1.axvline(x=nrmse, color="k", ls="dashed", lw=1.5)

        sns.histplot(data, x="Spearman", bins=50, ax=ax2, hue="cluster_name", hue_order=order, palette=color_dict)
        ax2.set_xlabel("Spearman's $\\rho$")
        ax2.axvline(x=spearman, color="k", ls="dashed", lw=1.5)
        #ax2.legend(title="Cluster")
        sns.move_legend(ax2, loc="upper left", title="Cluster", frameon=False)


        plt.suptitle(f"Distribution of Metrics for {model} Model", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{fig_path}/metrics_cluster_distribution_{model}.png")
        plt.show()
    
    print("finished metrics distribution1")


def plot_metrics_distribution():

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    merged = merged[merged["NRMSE_range"] < 1]

    color_dict = {'SD': "tab:red",
              'TU pre-ZGA': "tab:blue",
              'TU post-ZGA': "tab:green", 
              'SU ZGA': "tab:purple",
              'SU post-ZGA': "tab:orange"}

    for model in merged.model.unique():
        data = merged[merged["model"] == model]

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

        sns.histplot(data, x="MASE", bins=50, ax=ax0,)
        ax0.set_xlabel("MASE")
        ax0.axvline(x=1, color="k", ls="dashed", lw=1.5)

        sns.histplot(data, x="NRMSE_range", bins=50, ax=ax1)
        ax1.set_xlabel("NRMSE")
        ax1.axvline(x=nrmse, color="k", ls="dashed", lw=1.5)

        sns.histplot(data, x="Spearman", bins=50, ax=ax2)
        ax2.set_xlabel("Spearman's $\\rho$")
        ax2.axvline(x=spearman, color="k", ls="dashed", lw=1.5)
        #ax2.legend(title="Cluster")

        plt.suptitle(f"Distribution of Metrics for {model} Model", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{fig_path}/metrics_distribution_{model}.png")
        plt.show()
    
    print("finished metrics distribution")

def plot_metrics_distribution2():

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    data = merged[merged["NRMSE_range"] < 1]

    order = ["Basic", "ZGA M-decay", "ZGA Z-decay",  "Repression M-decay",  "Repression Z-decay",] 
    col = sns.color_palette("Dark2")  

    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    ax0.axvline(x=1, color="grey", ls="dashed", lw=1.5)
    ax0.fill_between(x=[0.0, 1.0], y1=0, y2=1, color="lightgrey", alpha=0.3)
    sns.kdeplot(data, x="MASE",  ax=ax0, hue="model", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    ax0.set(xlabel="MASE", xlim=(-0.1, 2.1))
    ax0.xaxis.set_inverted(False)
    ax0.grid(True)
    #ax0.set_xticks([0, 1, 2, 3])

    ax1.axvline(x=nrmse, color="grey", ls="dashed", lw=1.5)
    ax1.axvline(x=0.1, color="grey", ls="dotted", lw=1.5)
    ax1.fill_between(x=[0.0, nrmse], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="NRMSE_range", ax=ax1, hue="model", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    #ax1.legend(frameon=False,  loc="lower right")
    ax1.xaxis.set_inverted(False)
    ax1.grid(True)
    ax1.set_xlabel("NRMSE")
    
    ax2.axvline(x=spearman, color="grey", ls="dashed", lw=1.5)
    ax2.axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
    ax2.fill_between(x=[1.0, spearman], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="Spearman", ax=ax2, hue="model", hue_order=order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True,)
    ax2.xaxis.set_inverted(False)
    ax2.yaxis.set_inverted(False)
    ax2.set(xlabel="Spearman's $\\rho$", xlim=(-1.1, 1.1))
    #ax2.set_xticks([1, spearman, 0.0, -0.5, -1.0])
    #ax2.set_xticks([-1, -0.5, 0.0, -0.5, -1.0])
    ax2.grid(True)

    sns.move_legend(ax2, loc=(1.02, 0.3), title="Model", frameon=False)

    plt.suptitle(f"Goodness of fit metrics distribution, full dataset", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/metrics_cluster_distribution_modelhue_cumu.png")
    plt.show()
    
    print("finished metrics distribution2")

def plot_metrics_distribution3():

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    data = merged[merged["NRMSE_range"] < 1]

    color_dict = {'SD': "tab:red",
            'TU pre-ZGA': "tab:blue",
            'TU post-ZGA': "tab:green", 
            'SU ZGA': "tab:purple",
            'SU post-ZGA': "tab:orange"}
    order = ["SD", "TU pre-ZGA", "TU post-ZGA", "SU ZGA", "SU post-ZGA"]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    ax0.axvline(x=1, color="grey", ls="dashed", lw=1.5)
    ax0.fill_between(x=[0.0, 1.0], y1=0, y2=1, color="lightgrey", alpha=0.3)
    sns.kdeplot(data, x="MASE",  ax=ax0, hue="cluster_name", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    ax0.set(xlabel="MASE", xlim=(-0.1, 2.1))
    ax0.xaxis.set_inverted(False)
    ax0.grid(True)
    #ax0.set_xticks([0, 1, 2, 3])

    ax1.axvline(x=nrmse, color="grey", ls="dashed", lw=1.5)
    ax1.fill_between(x=[0.0, nrmse], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="NRMSE_range", ax=ax1, hue="cluster_name", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    #ax1.legend(frameon=False,  loc="lower right")
    ax1.xaxis.set_inverted(False)
    ax1.grid(True)
    ax1.set_xlabel("NRMSE")
    
    ax2.axvline(x=spearman, color="grey", ls="dashed", lw=1.5)
    ax2.fill_between(x=[1.0, spearman], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="Spearman", ax=ax2, hue="cluster_name", hue_order=order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True,)
    ax2.xaxis.set_inverted(False)
    ax2.yaxis.set_inverted(False)
    ax2.set(xlabel="Spearman's $\\rho$", xlim=(-1.1, 1.1))
    #ax2.set_xticks([1, spearman, 0.0, -0.5, -1.0])
    #ax2.set_xticks([-1, -0.5, 0.0, -0.5, -1.0])
    ax2.grid(True)

    #sns.move_legend(ax2, loc=(1.02, 0.3), title="Cluster", frameon=False)

    plt.suptitle(f"Goodness of fit metrics distribution, full dataset", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/metrics_cluster_distribution_clusterhue_cumu.png")
    plt.show()
    
    print("finished metrics distribution3")

def plot_metrics_distribution4():

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    data = merged[merged["NRMSE_range"] < 1]

    color_dict = {'SD': "tab:red",
            'TU pre-ZGA': "tab:blue",
            'TU post-ZGA': "tab:green", 
            'SU ZGA': "tab:purple",
            'SU post-ZGA': "tab:orange"}
    order = ["SD", "TU pre-ZGA", "TU post-ZGA", "SU ZGA", "SU post-ZGA"]

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex="col")

    ax0.axvline(x=1, color="grey", ls="dashed", lw=1.5)
    ax0.fill_between(x=[0.0, 1.0], y1=0, y2=1, color="lightgrey", alpha=0.3)
    sns.kdeplot(data, x="MASE",  ax=ax0, hue="cluster_name", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    ax0.set(xlabel="MASE", xlim=(-0.1, 2.1))
    ax0.xaxis.set_inverted(False)
    ax0.grid(True)
    #ax0.set_xticks([0, 1, 2, 3])

    ax1.axvline(x=nrmse, color="grey", ls="dashed", lw=1.5)
    ax1.axvline(x=0.1, color="grey", ls="dotted", lw=1.5)
    ax1.fill_between(x=[0.0, nrmse], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="NRMSE_range", ax=ax1, hue="cluster_name", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    #ax1.legend(frameon=False,  loc="lower right")
    ax1.xaxis.set_inverted(False)
    ax1.grid(True)
    ax1.set_xlabel("NRMSE")
    
    ax2.axvline(x=spearman, color="grey", ls="dashed", lw=1.5)
    ax2.axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
    ax2.fill_between(x=[1.0, spearman], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="Spearman", ax=ax2, hue="cluster_name", hue_order=order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True,)
    ax2.xaxis.set_inverted(False)
    ax2.yaxis.set_inverted(False)
    ax2.set(xlabel="Spearman's $\\rho$", xlim=(-1.1, 1.1))
    ax2.grid(True)
    sns.move_legend(ax2, loc=(1.02, 0.3), title="Cluster", frameon=False)

    ###---- model hue ----###

    merged = pd.read_csv(f"{fig_path}/merged_metrics_gof_cluster_{t_end}hpf.csv")
    data = merged[merged["NRMSE_range"] < 1]

    order = ["Basic", "ZGA M-decay", "ZGA Z-decay",  "Repression M-decay",  "Repression Z-decay",] 
    col = sns.color_palette("Dark2")  
    color_dict = {
        "Basic": col[7],  # grey
        "ZGA M-decay": col[4],  # green
        "ZGA Z-decay": col[0],  # green
        "Repression M-decay": col[1], #orange
        "Repression Z-decay": col[3]} #pink

    ax3.axvline(x=1, color="grey", ls="dashed", lw=1.5)
    ax3.fill_between(x=[0.0, 1.0], y1=0, y2=1, color="lightgrey", alpha=0.3)
    sns.kdeplot(data, x="MASE",  ax=ax3, hue="model", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    ax3.set(xlabel="MASE", xlim=(-0.1, 2.1))
    ax3.xaxis.set_inverted(False)
    ax3.grid(True)
    #ax0.set_xticks([0, 1, 2, 3])

    ax4.axvline(x=nrmse, color="grey", ls="dashed", lw=1.5)
    ax4.axvline(x=0.1, color="grey", ls="dotted", lw=1.5)
    ax4.fill_between(x=[0.0, nrmse], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="NRMSE_range", ax=ax4, hue="model", hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,)
    #ax1.legend(frameon=False,  loc="lower right")
    ax4.xaxis.set_inverted(False)
    ax4.grid(True)
    ax4.set_xlabel("NRMSE")
    
    ax5.axvline(x=spearman, color="grey", ls="dashed", lw=1.5)
    ax5.axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
    ax5.fill_between(x=[1.0, spearman], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="Spearman", ax=ax5, hue="model", hue_order=order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True,)
    ax5.xaxis.set_inverted(False)
    ax5.yaxis.set_inverted(False)
    ax5.set(xlabel="Spearman's $\\rho$", xlim=(-1.1, 1.1))
    ax5.grid(True)
    sns.move_legend(ax5, loc=(1.02, 0.3), title="Model", frameon=False)


    #sns.move_legend(ax2, loc=(1.02, 0.3), title="Cluster", frameon=False)

    plt.suptitle(f"Goodness of fit metrics distribution, full dataset", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/metrics_cluster_distribution_clustermodelhue_cumu.png")
    plt.show()
    
    print("finished metrics distribution4")


def main():
    
    merge_datasets()
    
    plot_metrics()
    #plot_metrics_violin()
    #plot_metrics_distribution()
    #plot_metrics_distribution1()
    #plot_metrics_distribution2()
   # plot_metrics_distribution3()
   # plot_metrics_distribution4()
    #plot_model_fits(t_end=24)
    #plot_model_fits_smooth(t_end=120)
    #plot_model_fits_smooth(t_end=24)
    #plot_model_fits_smooth(t_end=48)
    #plot_parameters()
    #plot_venn()
    #plot_venn_cluster()
    #plot_acceptedfits()
    #plot_accepted_heatmap()
    


main()


# sbatch plot.sh



