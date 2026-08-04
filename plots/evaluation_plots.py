'''Evaluation and plots of baseline model fits'''

import xarray as xr
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' ----- Global Parmeters ----- '''

FIG_PATH = "./figures"

NRMSE_thres = 0.45
RHO_thres = 0.7

cluster = xr.load_dataset("data/all_gene_cluster_annotation_minmax.nc")
cluster_order = ["SD", "DSD", "SU", "DSU"]
cluster_names = {0 : "SD", 1 : "DSD", 2 : "SU", 3 : "DSU"}
col_c = sns.color_palette("Set1", n_colors=4)  
cluster_color_dict = {"SD": col_c[0], "DSD": col_c[1], "SU": col_c[2], "DSU": col_c[3], }

col_m = sns.color_palette("Dark2")  
mod_color_dict = {"Basic": col_m[7],"Rep_M": col_m[1], "Rep_Z": col_m[4], "Rep_V": col_m[2]} 
model_order = ["Basic", "Rep_M", "Rep_Z"]

col_s = sns.color_palette("twilight_shifted")  
src_color_dict =  {'White':col_s[0], 'Pauli':col_s[1], 'JN':col_s[4], 'BK':col_s[5]}

# ================================================================================================

def combine_ds(save_csv=False):

    print("Combine Datasets...")

    gof_all_list = []
    gof_src_list = []
    params_list = []

    """
    data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean.nc")
    mask = (data.tpm.max(dim="time", skipna=True) >= 1).all(dim="source") # Keep only relevantly expressed genes
    data = data.sel(ensembl_gene_id=mask)
    genes = data.ensembl_gene_id.values
    """
    labels = xr.load_dataset("data/all_gene_cluster_annotation_minmax.nc")
    genes = labels.ensembl_gene_id.values

    for model in model_order:

        # load data and combine with cluster assignment
        gof_all = pd.read_csv(f"results/results_summary/{model}/goodness_of_fit_summary.csv")
        gof_src = pd.read_csv(f"results/results_summary/{model}/gof_by_source_joined.csv")
        params = pd.read_csv(f"results/results_summary/{model}/parameter_fit_summary.csv")

        gof_all = gof_all[gof_all["gene_id"].isin(genes)]
        gof_src = gof_src[gof_src["gene_id"].isin(genes)]
        params = params[params["gene_id"].isin(genes)]

        if model == "Basic":
            gof_src = gof_src.drop(columns=["accepted"])
        
        # Build a lookup table from the DataArray's coordinates
        lookup = pd.DataFrame({
            "ensembl_gene_id": cluster["ensembl_gene_id"].values,
            "supercluster_no": cluster["supercluster"].values,
            "subcluster_no": cluster["subcluster"].values,
        }).set_index("ensembl_gene_id")
        lookup["supercluster"]=lookup["supercluster_no"].map(cluster_names)

        # Map onto df by gene_id
        for df in [gof_all, gof_src, params]:
            df["supercluster"] = df["gene_id"].map(lookup["supercluster"])
            df["supercluster_no"] = df["gene_id"].map(lookup["supercluster_no"])
            df["subcluster_no"] = df["gene_id"].map(lookup["subcluster_no"])
            df["model"] = model  # track source model

        gof_all_list.append(gof_all)
        gof_src_list.append(gof_src)
        params_list.append(params)

    gof_all_combined = pd.concat(gof_all_list, ignore_index=True)
    gof_src_combined = pd.concat(gof_src_list, ignore_index=True)
    params_combined = pd.concat(params_list, ignore_index=True)

    if save_csv:
        gof_all_combined.to_csv("results/results_summary/gof_all_combined.csv", index=False)
        gof_src_combined.to_csv("results/results_summary/gof_src_combined.csv", index=False)
        params_combined.to_csv("results/results_summary/params_combined.csv", index=False)

    return gof_all_combined, gof_src_combined, params_combined


def point_plot_metrics_all(gof_all_combined):
    '''plot NRMSE & BIC over clusters'''

    print("Plot GOF metrics - point plot")

    df = gof_all_combined.sort_values("supercluster_no")

    supercluster = pd.unique(df["supercluster"].dropna())
    metrics = ["BIC", "WAIC", "NRMSE", "rho", "pearsonr"]

    df = df[df["WAIC"] > -100]

    fig, ax = plt.subplots(len(metrics), len(supercluster), figsize=(len(supercluster)*2, 1.5*len(metrics)), sharey="row", sharex="row")

    for r, metr in enumerate(metrics):
        for c, clstr in enumerate(supercluster):

            data = df[df["supercluster"] == clstr].copy()
            
            if metr == "NRMSE":
                ax[r][c].axvline(x=NRMSE_thres, color='k', linestyle='-.', linewidth=0.5)
            if metr == "rho":
                ax[r][c].axvline(x=RHO_thres, color='k', linestyle='-.', linewidth=0.5)

            sns.pointplot(
                data=data, x=metr, y="model",
                order = model_order,
                hue = "model",
                palette = mod_color_dict,
                estimator="median",
                markersize=4, marker="D",
                linestyle="none",
                errorbar = ('pi', 90), capsize=.2,
                ax=ax[r][c],
                legend = False,
                err_kws = {"alpha":0.6,'linewidth': 1.5,},
            )

            ax[r][c].set(xlabel="", ylabel=metr)
            ax[r][c].grid(True)
            ax[0][c].set_title(f"{clstr}", fontsize=10) 
            ax[r][c].set_yticks(model_order)
            ax[r][c].set_yticklabels(model_order, fontsize=8)

    plt.suptitle(f"Model performance across superclusters", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/point_plot_metrics_mean.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_distribution_all(gof_all_combined, hue_key = "model"):

    print("Plot GOF metrics - distribution by model")

    color_dict = {"source": src_color_dict, "model": mod_color_dict}[hue_key]

    data = gof_all_combined[gof_all_combined["NRMSE"] < 1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4), sharey=True, sharex="col", )

    ax1.axvline(x=RHO_thres, color="grey", ls="dashed", lw=1.5)
    ax1.axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
    ax1.fill_between(x=[1.0, RHO_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="rho", hue=hue_key, hue_order=model_order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True, legend=False, ax=ax1 )
    ax1.xaxis.set_inverted(False)
    ax1.yaxis.set_inverted(False)
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set(title="Spearman's $\\rho$",xlabel="$\\rho$", xlim=(-1.1, 1.1))
    ax1.grid(True)

    ax2.axvline(x=RHO_thres, color="grey", ls="dashed", lw=1.5)
    ax2.axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
    ax2.fill_between(x=[1.0, RHO_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="pearsonr", hue=hue_key, hue_order=model_order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True, legend=False, ax=ax2 )
    ax2.xaxis.set_inverted(False)
    ax2.yaxis.set_inverted(False)
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax2.set(title="Pearson's r",xlabel="Pearson's r", xlim=(-1.1, 1.1))
    ax2.grid(True)

    ax3.axvline(x=NRMSE_thres, color="grey", ls="dashed", lw=1.5)
    ax3.axvline(x=0.1, color="grey", ls="dotted", lw=1.5)
    ax3.fill_between(x=[0.0, NRMSE_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="NRMSE", hue=hue_key, hue_order=model_order, palette=color_dict, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,legend=True, ax=ax3, )
    ax3.xaxis.set_inverted(False)
    ax3.grid(True)
    ax3.set(title="NRMSE", xlabel="NRMSE")
    
    sns.move_legend(ax3, loc=(1.02, 0.3), title=hue_key, frameon=False)

    plt.suptitle(f"Goodness of fit metrics distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/gof_distribution_{hue_key}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_distribution_src(gof_src, hue_key = "source"):

    print("Plot GOF metrics - distribution")

    color_dict = {"source": src_color_dict, "model": mod_color_dict}[hue_key]

    data = gof_src[gof_src["NRMSE"] < 1]

    models = gof_src.model.unique()
    order = gof_src.source.unique()

    fig, ax = plt.subplots(len(models), 3, figsize=(13, 4*len(models)), sharey=True, sharex="col", squeeze=False)

    for m, mod in enumerate(models):

        props = dict(boxstyle='round', facecolor=mod_color_dict[mod], alpha=0.2)
        ax[m][0].text(0.05, .9, mod, transform=ax[m][0].transAxes, bbox=props)

        ax[m][0].axvline(x=1, color="grey", ls="dashed", lw=1.5)
        ax[m][0].fill_between(x=[0.0, 1.0], y1=0, y2=1, color="lightgrey", alpha=0.3)
        sns.kdeplot(data, x="MASE",  ax=ax[m][0], hue=hue_key, hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True,)
        ax[m][0].set(title="MASE", xlabel="MASE", xlim=(-0.1, 2.1))
        ax[m][0].xaxis.set_inverted(False)
        ax[m][0].grid(True)

        ax[m][1].axvline(x=NRMSE_thres, color="grey", ls="dashed", lw=1.5)
        ax[m][1].axvline(x=0.1, color="grey", ls="dotted", lw=1.5)
        ax[m][1].fill_between(x=[0.0, NRMSE_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
        sns.kdeplot(data, x="NRMSE", ax=ax[m][1], hue=hue_key, hue_order=order, palette=color_dict, legend=False, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True,)
        ax[m][1].xaxis.set_inverted(False)
        ax[m][1].grid(True)
        ax[m][1].set(title="NRMSE", xlabel="NRMSE")
        
        ax[m][2].axvline(x=RHO_thres, color="grey", ls="dashed", lw=1.5)
        ax[m][2].axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
        ax[m][2].fill_between(x=[1.0, RHO_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
        sns.kdeplot(data, x="rho", ax=ax[m][2], hue=hue_key, hue_order=order, palette=color_dict, linewidth=1.5,
                        cumulative=True, common_norm=False, common_grid=True,)
        ax[m][2].xaxis.set_inverted(False)
        ax[m][2].yaxis.set_inverted(False)
        ax[m][2].set(title="Spearman's $\\rho$",xlabel="$\\rho$", xlim=(-1.1, 1.1))
        ax[m][2].grid(True)

        sns.move_legend(ax[m][2], loc=(1.02, 0.3), title=hue_key, frameon=False)

    plt.suptitle(f"Goodness of fit metrics distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/gof_distribution_{hue_key}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accepted_heatmap(gof_src, hue_key="source"):

    fig, ax = plt.subplots(1, len(model_order), figsize=(5*len(model_order), 4))

    for m, model_name in enumerate(model_order):
        print(f"Plot heatmap for model {model_name}...")

        merged = (gof_src[gof_src["model"] == model_name]).sort_values(["supercluster_no", "gene_id"])
        sources = merged.source.unique()

        # Load original data
        accepted = merged[(merged["rho"] > RHO_thres) & (merged["NRMSE"] < NRMSE_thres) ]

        count_model = accepted[hue_key].value_counts()
        count_cluster = accepted.groupby("supercluster")["gene_id"].nunique()

        merged_counts = merged.groupby([hue_key, "supercluster"]).size().reset_index(name="merged")
        accepted_counts = accepted.groupby([hue_key, "supercluster"]).size().reset_index(name="accepted")
        result = pd.merge(merged_counts, accepted_counts, how="left", on=[hue_key, "supercluster"])

        result["accepted"] = result["accepted"].fillna(0).astype(int)
        result["ratio"] = result["accepted"] / result["merged"] * 100

        # Pivot for heatmap
        pivot = result.pivot(index="supercluster", columns=hue_key, values="ratio") 

        # decide column order
        if hue_key == "model":
            pivot = pivot.reindex(index=cluster_order, columns=model_order)
        if hue_key == "source":
            pivot = pivot.reindex(index=cluster_order, columns=sources)

        # Plot heatmap
        cmap = sns.color_palette("blend:#f37736,#fdf498,#7bc043", as_cmap=True)
        sns.heatmap(pivot, annot=True, linewidths=0.5, vmin=0, vmax=100, cmap=cmap, alpha=0.8, 
                    cbar_kws={'label': '% accepted', 'shrink': 0.75, 'pad': 0.02 }, ax=ax[m])

        ax[m].set(title=f"- {model_name} -", xlabel=hue_key, ylabel="supercluster")

        ax[m].set_xticklabels([f"{c}\n(n={count_model.get(c, 0)})" for c in pivot.columns])
        ax[m].set_yticklabels([f"{c}\n(n={count_cluster.get(c, 0)})" for c in pivot.index], rotation=0)

    plt.suptitle("Accepted model fits [%] per supercluster and source")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/heatmap_accepted_cluster_{hue_key}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_params_cluster_model(ds_params, model_name):

    print(f"Plot fitted parameter distribution - model {model_name}")

    params_dict = {
                "Basic": (["delta_mean", "beta_mean"], [r"$\delta$",r"$\beta$"]),
                "Rep_M": (["delta_m_mean", "delta_z_mean", "alpha_mean", "beta_mean", "t_zga", "t_reg"], 
                   [ r"$\delta_m$", r"$\delta_z$", r"$\alpha$", r"$\beta$", r"$t_{zga}$", r"$t_{reg}$"]),
                "Rep_Z": (["delta_m_mean", "delta_z_mean", "alpha_mean", "beta_mean", "t_zga", "t_reg"], 
                   [ r"$\delta_m$", r"$\delta_z$", r"$\alpha$", r"$\beta$", r"$t_{zga}$", r"$t_{reg}$"]),
                "Rep_V": (["delta_m_mean", "delta_z_mean", "alpha_mean", "beta_mean", "t_zga", "t_reg", "t_deg_mean"], 
                   [ r"$\delta_m$", r"$\delta_z$", r"$\alpha$", r"$\beta$", r"$t_{zga}$", r"$t_{reg}$", r"$t_{deg}$"]) 
                   }
    
    ds_params = ds_params[ds_params["model"] == model_name]
    params = params_dict[model_name][0]
    title = params_dict[model_name][1]

    fix, ax = plt.subplots(int((len(params)+1)/2), 2, figsize = (8, int((len(params)+1)/2*3)),)
    ax = ax.flatten()

    if "beta_mean" in params:
        data = ds_params[ds_params["beta_mean"] > 0.0]
    if "t_zga" in params:
        ds_params["t_zga"] = np.minimum(ds_params["t_zga_mean"],  ds_params["t_rep_mean"])
        ds_params["t_reg"] = np.maximum(ds_params["t_zga_mean"],  ds_params["t_rep_mean"])
        data = ds_params[ds_params["t_reg"] < 120.0]
    else:
        data = ds_params

    for i, param in enumerate(params):
        sns.kdeplot(data=data, x=param, log_scale=True, 
                    hue="supercluster", 
                    palette=cluster_color_dict,
                    common_norm=False,
                    legend= True if i == len(params)-1 else False,
                    fill=False, 
                    ax=ax[i])
        
        ax[i].set(title=title[i], xlabel=title[i])
    sns.move_legend(ax[len(params)-1], loc=(1.02, 0.3))
    plt.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/params_cluster_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rep_params_violin(ds_params):

    print("plot params violin plot for Rep-Models")

    params = ["t_zga", "t_reg"]
    title = ["$t_{zga}$", "$t_{reg}$"]

    df = ds_params[ds_params["model"] != "Basic" ].copy()

    df["t_zga"] = np.minimum(df["t_zga_mean"],  df["t_rep_mean"])
    df["t_reg"] = np.maximum(df["t_zga_mean"],  df["t_rep_mean"])
    df = df[df["t_reg"] < 120.0]
    
    fig, ax = plt.subplots(1, len(params), figsize = (10, 4.5), sharey=True, sharex=True)

    for i, param in enumerate(params):
        sns.violinplot(data=df, x=param, y="supercluster", log_scale=True, hue="model", 
                        split=True, inner="quart",palette=mod_color_dict, order=cluster_order,
                        legend= True if i == len(params)-1 else False, ax=ax[i])
        
        ax[i].set(title=title[i], xlabel=title[i])
        ax[i].grid(True)

    # move legend outside last axes
    sns.move_legend(ax[-1], loc=(1.01, 0.4), frameon=False)
    fig.suptitle("Estimated parameters of transcription onset $t_{zga}$ and regulation onset $t_{reg}$")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/violin_params.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_regulation_direction(ds_params):

    print("Plot regulation indicator r ...")

    df = ds_params[ds_params["model"] != "Basic" ].sort_values("supercluster_no")
    df["r"] = df["beta_mean"] / df["alpha_mean"]
    df = df[df["r"] > 1e-5]

    palette = sns.color_palette("Accent") 

    panels = ["Repression M-decay", "Repression Z-decay",]
    models = ["Rep_M", "Rep_Z",]

    fig, axes = plt.subplots(1, len(models), figsize=(5.5*len(models), 3.5), sharey=True, sharex=True)
    for i, (ax, model) in enumerate(zip(axes, models)):
        sub = df[df["model"] == model].copy()

        sns.stripplot( data=sub, x="r", y="supercluster", log_scale=True, hue="subcluster_no", order=cluster_order,
                        dodge=True, size=3, alpha=0.7, palette=palette, ax=ax, legend=(i == 1), zorder=1)

        # ---- overlay median tick marks per supercluster ----
        sns.pointplot(data=sub, x="r", y="supercluster", markers="|", markersize=40, linestyles="", hue="supercluster",
                        estimator="median", palette=cluster_color_dict, ax=ax, legend=False, zorder=3)

        ax.axvline(1, color="black", linestyle="--", linewidth=1, zorder=2)
        ax.set(title=panels[i], xlabel = "indicator of regulation $r = \\beta / \\alpha$", ylabel="")

    # shared legend on the right
    sns.move_legend(axes[1], loc=(1.02, 0.3), frameon=False, title="Subcluster")

    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/regulation_indicator.png", dpi=150, bbox_inches="tight")
    plt.close()



def plot_regulation_direction_2(ds_params):

    print("Plot regulation indicator r ...")

    df = ds_params[ds_params["model"] != "Basic" ].sort_values("supercluster_no")
    df["r"] = df["beta_mean"] / df["alpha_mean"]
    df = df[np.isfinite(df["r"]) & (df["r"] > 1e-5)]

    palette_m = sns.color_palette("Oranges") 
    palette_z = sns.color_palette("Greens_r") 

    panels = ["Repression M-decay", "Repression Z-decay",]
    models = ["Rep_M", "Rep_Z",]

    marker_dict = ["o", "x", "^"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), sharey=True, sharex=True)
    sns.violinplot( data=df, x="r", y="supercluster", log_scale=True, hue="model", palette=mod_color_dict,
                        split=True, inner = "quart", alpha=0.7, legend=True, zorder=1)
    # ---- overlay median tick marks per supercluster ----

    sub_m = df[df["model"] == "Rep_M"].copy()
    sub_z = df[df["model"] == "Rep_Z"].copy()
    sns.pointplot(data=sub_m, x="r", y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                estimator="median", palette=palette_m, ax=ax, legend=True, zorder=3)
    sns.pointplot(data=sub_z, x="r", y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                estimator="median", palette=palette_z, ax=ax, legend=True, zorder=3)

    ax.axvline(1, color="black", linestyle="--", linewidth=1, zorder=2)
    ax.set(title="Regulation indicator r", xlabel = "$r = \\beta / \\alpha$", ylabel="Subcluster")

    # shared legend on the right
    sns.move_legend(ax, loc=(1.02, 0.3), frameon=False, title="Subcluster")

    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/regulation_indicator_2.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_t_params(ds_params):

    df = ds_params[ds_params["model"] != "Basic" ].sort_values("supercluster_no")

    params = ["t_zga_mean", "t_reg_mean"]
    title = ["$t_{zga}$", "$t_{reg}$"]

    #df["t_zga"] = np.minimum(df["t_zga_mean"],  df["t_rep_mean"])
    #df["t_reg"] = np.maximum(df["t_zga_mean"],  df["t_rep_mean"])
    df = df[df["t_reg_mean"] < 120.0]
    
    palette_m = sns.color_palette("Oranges") 
    palette_z = sns.color_palette("Greens_r") 

    marker_dict = ["o", "x", "^"]

    panels = ["Repression M-decay", "Repression Z-decay",]
    models = ["Rep_M", "Rep_Z",]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, sharex=True)

    sns.violinplot( data=df, x=params[0], y="supercluster", log_scale=True, hue="model", palette=mod_color_dict,
                        split=True, inner = "quart", alpha=0.7, legend=False, zorder=1, ax=ax1)
    sns.violinplot( data=df, x=params[1], y="supercluster", log_scale=True, hue="model", palette=mod_color_dict,
                        split=True, inner = "quart", alpha=0.7, legend=True, zorder=1, ax=ax2)
    # ---- overlay median tick marks per supercluster ----

    sub_m = df[df["model"] == "Rep_M"].copy()
    sub_z = df[df["model"] == "Rep_Z"].copy()

    # ZGA
    sns.pointplot(data=sub_m, x=params[0], y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                       estimator="median", palette=palette_m, ax=ax1, legend=False, zorder=3)
    sns.pointplot(data=sub_z, x=params[0], y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                    estimator="median", palette=palette_z, ax=ax1, legend=False, zorder=3)
    # REP
    sns.pointplot(data=sub_m, x=params[1], y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                estimator="median", palette=palette_m, ax=ax2, legend=True, zorder=3)
    sns.pointplot(data=sub_z, x=params[1], y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                    estimator="median", palette=palette_z, ax=ax2, legend=True, zorder=3)

    ax1.axvline(3, color="black", linestyle="--", linewidth=1, zorder=2, label="ZGA")
    ax1.set(title=title[0], xlabel = "time (hpf)", ylabel="")

    ax2.axvline(3, color="black", linestyle="--", linewidth=1, zorder=2, label="ZGA")
    ax2.set(title=title[1], xlabel = "time (hpf)", ylabel="")

    # shared legend on the right
    sns.move_legend(ax2, loc=(1.02, 0.2), frameon=False, title="Subcluster")

    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/t_params_violin_cluster.png", dpi=300)
    plt.show()

def plot_half_life(ds_params):

    df = ds_params[ds_params["model"] != "Basic" ].sort_values("supercluster_no")

    params = ["t_m_half", "t_z_half"]
    title = ["$t_{m, 1/2}$", "$t_{z, 1/2}$"]

    df["t_m_half"] = np.log(2) / df["delta_m_mean"]
    df["t_z_half"] = np.log(2) / df["delta_z_mean"]

    df = df[df["t_reg_mean"] < 120.0]
    
    palette_m = sns.color_palette("Oranges") 
    palette_z = sns.color_palette("Greens_r") 

    marker_dict = ["o", "x", "^"]

    panels = ["Repression M-decay", "Repression Z-decay",]
    models = ["Rep_M", "Rep_Z",]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, sharex=True)

    sns.violinplot( data=df, x="t_m_half", y="supercluster", log_scale=True, hue="model", palette=mod_color_dict,
                        split=True, inner = "quart", alpha=0.7, legend=False, zorder=1, ax=ax1)
    sns.violinplot( data=df, x="t_z_half", y="supercluster", log_scale=True, hue="model", palette=mod_color_dict,
                        split=True, inner = "quart", alpha=0.7, legend=True, zorder=1, ax=ax2)
    # ---- overlay median tick marks per supercluster ----

    sub_m = df[df["model"] == "Rep_M"].copy()
    sub_z = df[df["model"] == "Rep_Z"].copy()

    # ZGA
    sns.pointplot(data=sub_m, x="t_zga_mean", y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                       estimator="median", palette=palette_m, ax=ax1, legend=False, zorder=3)
    sns.pointplot(data=sub_z, x="t_zga_mean", y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                    estimator="median", palette=palette_z, ax=ax1, legend=False, zorder=3)
    # REP
    sns.pointplot(data=sub_m, x="t_z_half", y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                estimator="median", palette=palette_m, ax=ax2, legend=True, zorder=3)
    sns.pointplot(data=sub_z, x="t_z_half", y="supercluster", markers=marker_dict, markersize=3, linestyles="", hue="subcluster_no",
                    estimator="median", palette=palette_z, ax=ax2, legend=True, zorder=3)

    ax1.axvline(3, color="black", linestyle="--", linewidth=1, zorder=2, label="ZGA")
    ax1.set(title=title[0], xlabel = "time (hpf)", ylabel="")

    ax2.axvline(3, color="black", linestyle="--", linewidth=1, zorder=2, label="ZGA")
    ax2.set(title=title[1], xlabel = "time (hpf)", ylabel="")

    # shared legend on the right
    sns.move_legend(ax2, loc=(1.02, 0.2), frameon=False, title="Subcluster")

    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/half_lifes_violin_cluster.png", dpi=300)
    plt.show()

def barplot_accepted(gof_all):

    gof_all["accepted"] = (gof_all["rho"] > RHO_thres) & (gof_all["NRMSE"] < NRMSE_thres)


    plt.figure(figsize=(5, 2.5))
    sns.barplot(gof_all, x="model", y="accepted", estimator="mean", hue="model", palette=mod_color_dict)
    plt.ylabel("accepted (%)")
    plt.title("Accepted model fits")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/barplot_accepted.png", dpi=300)
    plt.close()

def barplot_accepted_subcluster(gof_all):

    palette = sns.color_palette("Accent") 
    gof_all["accepted"] = (gof_all["rho"] > RHO_thres) & (gof_all["NRMSE"] < NRMSE_thres)

    plt.figure(figsize=(15, 5))
    sns.catplot(gof_all, x ="supercluster", y="accepted", hue="subcluster_no", col="model", kind="bar", estimator="mean", palette=palette)
    plt.ylabel("accepted (%)")
    #plt.suptitle("Accepted model fits")
    #plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/barplot_accepted_subcluster.png", dpi=300)
    plt.close()


def plot_peak_expression():

    ds = cluster.copy().tpm.mean("source")
    df = pd.DataFrame({"ensembl_gene_id": ds.ensembl_gene_id.values,
                    "supercluster_no":ds.supercluster.values, "subluster_no":ds.subcluster.values,})
    stats = xr.Dataset({"t_peak": ds.idxmax(dim="time"),})
    stats_df = stats.to_dataframe().reset_index().drop(columns=["supercluster", "subcluster"])
    df = df.merge(stats_df, on="ensembl_gene_id", how="left")
    df["supercluster"] = df["supercluster_no"].map(cluster_names)

    #time = cluster.sel(source="White").dropna(dim="time", how="all", subset=["tpm"]).time.values
    time = np.array([  0., 8.  ,24.  ,  36.  ,  48.  ,  72.  ,  96.  , 120.  ])

    fig, ax =plt.subplots(1,1, figsize=(8,4))
    sns.violinplot(df, x="t_peak", y="supercluster", hue="supercluster", palette=cluster_color_dict, order=cluster_order, ax=ax, inner="box")
    #sns.stripplot(ds, x="t_max", y="cluster_name", palette=colors, order=order, ax=ax,)
    #sns.move_legend(ax, loc=(1.01, 0.3), frameon=False, title="Cluster")
    ax.set(xlabel="Time of peak expression (hpf)", title="Expression peak timing", ylabel="")
    plt.xticks(time.astype(int))
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/violin_peak_epxression.png")
    plt.close()


def plot_params_cluster(params_all):

    df = params_all[params_all["model"] != "Basic"].copy()
    df["r"] = df["beta_mean"] / df["alpha_mean"]
    df = df[(df["r"] > 1e-5) & (df["t_zga_mean"] < 120)]

    #df = df[df["t_reg_mean"] < 120.0]

    df_z = df[df["model"] == "Rep_Z"].copy()
    df_m = df[df["model"] == "Rep_M"].copy()

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), sharex=True, sharey=True)
    sns.scatterplot(df_m, x="r", y="t_zga_mean", ax=ax1,  hue="supercluster", s=5, palette=cluster_color_dict, alpha=0.7, legend=False)
    sns.scatterplot(df_z, x="r", y="t_zga_mean", ax=ax2, hue="supercluster", s=5, palette=cluster_color_dict, alpha=0.7, legend=True)

    sns.kdeplot(df_m, x="r", y="t_zga_mean",  hue="supercluster", ax=ax1, palette=cluster_color_dict, legend=False, log_scale=True, fill=False, levels=3)
    sns.kdeplot(df_z, x="r",  y="t_zga_mean", hue="supercluster", ax=ax2, palette=cluster_color_dict, legend=True, log_scale=True, fill=False, levels=3)

    ax1.set(xlabel= "$r = \\beta / \\alpha$", ylabel="$t_{zga}$", title="M-decay", xscale="log", yscale="log", xlim=(1e-6, 1e2), ylim=(0, 120))
    ax2.set(xlabel="$r = \\beta / \\alpha$", ylabel="$t_{zga}$", title="Z-decay", xscale="log", yscale="log", xlim=(1e-6, 1e2), ylim=(0, 120))

    ax1.axvline(1, c="k", ls="--")
    ax2.axvline(1, c="k", ls="--")

    ax1.axhline(3, c="gray", ls="--", label="ZGA")
    ax2.axhline(3, c="gray", ls="--", label="ZGA")

    # Annotations
    ax1.text(x=2e-6, y=7e1, s="DSD", bbox={"color":cluster_color_dict["DSD"], "alpha":0.5, "boxstyle":"round"})
    ax1.text(x=2e-6, y=1e-1, s="SD", bbox={"color":cluster_color_dict["SD"], "alpha":0.5, "boxstyle":"round"})
    ax1.text(x=2e1, y=7e1, s="DSU", bbox={"color":cluster_color_dict["DSU"], "alpha":0.5, "boxstyle":"round"})
    ax1.text(x=2e1, y=1e-1, s="SU", bbox={"color":cluster_color_dict["SU"], "alpha":0.5, "boxstyle":"round"})

    ax2.text(x=2e-6, y=7e1, s="DSD", bbox={"color":cluster_color_dict["DSD"], "alpha":0.5, "boxstyle":"round"})
    ax2.text(x=2e-6, y=1e-1, s="SD", bbox={"color":cluster_color_dict["SD"], "alpha":0.5, "boxstyle":"round"})
    ax2.text(x=2e1, y=7e1, s="DSU", bbox={"color":cluster_color_dict["DSU"], "alpha":0.5, "boxstyle":"round"})
    ax2.text(x=2e1, y=1e-1, s="SU", bbox={"color":cluster_color_dict["SU"], "alpha":0.5, "boxstyle":"round"})

    plt.legend(title="Cluster", frameon=False, loc=(1.01, 0.3))
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/scatterplot_params_cluster.png")
    plt.close()


## Upset plot

'''-----------------------
            PLOT 
--------------------------'''

gof_all_combined, gof_src_combined, params_combined = combine_ds(save_csv=True)

'''-- GOF Metrics plots --'''

point_plot_metrics_all(pd.read_csv("results/results_summary/gof_all_combined.csv",))
plot_metrics_distribution_all(pd.read_csv("results/results_summary/gof_all_combined.csv",), hue_key = "model")
plot_metrics_distribution_src(pd.read_csv("results/results_summary/gof_src_combined.csv",), hue_key="source")
plot_accepted_heatmap(pd.read_csv("results/results_summary/gof_src_combined.csv",), hue_key="source")
barplot_accepted(pd.read_csv("results/results_summary/gof_all_combined.csv",))
barplot_accepted_subcluster(pd.read_csv("results/results_summary/gof_all_combined.csv",))

'''-- Parameter plots --'''
for model in model_order:
    plot_params_cluster_model(pd.read_csv("results/results_summary/params_combined.csv",), model)
plot_rep_params_violin(pd.read_csv("results/results_summary/params_combined.csv",))

plot_regulation_direction(pd.read_csv("results/results_summary/params_combined.csv",))
plot_regulation_direction_2(pd.read_csv("results/results_summary/params_combined.csv",))
plot_t_params(pd.read_csv("results/results_summary/params_combined.csv",))
plot_half_life(pd.read_csv("results/results_summary/params_combined.csv",))

plot_params_cluster(pd.read_csv("results/results_summary/params_combined.csv",))

'''-- Peak time --'''
plot_peak_expression()

# sbatch plots/plot.sh