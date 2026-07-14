'''Evaluation and plots of baseline model fits'''

import xarray as xr
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' ----- Global Parmeters ----- '''

FIG_PATH = "figures"

NRMSE_thres = 0.2
RHO_thres = 0.7

cluster = xr.load_dataset("data/all_gene_cluster_annotation_minmax.nc")
cluster_order = ["SD", "DSD", "SU", "DSU"]
cluster_names = {0 : "SD", 1 : "DSD", 2 : "SU", 3 : "DSU"}
col_c = sns.color_palette("Set1", n_colors=4)  
cluster_color_dict = {"SD": col_c[0], "DSD": col_c[1], "SU": col_c[2], "DSU": col_c[3], }

col_m = sns.color_palette("Dark2")  
mod_color_dict = {"Basic": col_m[7],"Rep_M": col_m[1], "Rep_Z": col_m[3]} 
model_order = ["Basic", "Rep_M", "Rep_Z"]

col_s = sns.color_palette("twilight_shifted")  
src_color_dict =  {'White':col_s[0], 'Pauli':col_s[1], 'JN':col_s[4], 'BK':col_s[5]}

# ================================================================================================

def combine_ds(save_csv=False):

    print("Combine Datasets...")

    gof_all_list = []
    gof_src_list = []
    params_list = []

    for model in ["Basic", "Rep_M", "Rep_Z"]:

        # load data and combine with cluster assignment
        gof_all = pd.read_csv(f"results_summary/{model}/goodness_of_fit_summary.csv")
        gof_src = pd.read_csv(f"results_summary/{model}/gof_by_source_joined.csv")
        params = pd.read_csv(f"results_summary/{model}/parameter_fit_summary.csv")
        
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
        gof_all_combined.to_csv("results_summary/gof_all_combined.csv", index=False)
        gof_src_combined.to_csv("results_summary/gof_src_combined.csv", index=False)
        params_combined.to_csv("results_summary/params_combined.csv", index=False)

    return gof_all_combined, gof_src_combined, params_combined


def point_plot_metrics_all(gof_all_combined):
    '''plot NRMSE & BIC over clusters'''

    print("Plot GOF metrics - point plot")

    df = gof_all_combined.sort_values("supercluster_no")

    models = pd.unique(df["model"].dropna())
    supercluster = pd.unique(df["supercluster"].dropna())
    metrics = ["BIC", "NRMSE", "rho"]

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
                order = models,
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
            ax[r][c].set_yticks(models)
            ax[r][c].set_yticklabels(models, fontsize=8)

    plt.suptitle(f"Model performance across superclusters", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/point_plot_metrics_mean.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics_distribution_all(gof_all_combined, hue_key = "model"):

    print("Plot GOF metrics - distribution by model")

    color_dict = {"source": src_color_dict, "model": mod_color_dict}[hue_key]

    data = gof_all_combined[gof_all_combined["NRMSE"] < 1]

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True, sharex="col", )

    ax[0].axvline(x=RHO_thres, color="grey", ls="dashed", lw=1.5)
    ax[0].axvline(x=0.9, color="grey", ls="dotted", lw=1.5)
    ax[0].fill_between(x=[1.0, RHO_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="rho", hue=hue_key, hue_order=model_order, palette=color_dict, linewidth=1.5,
                    cumulative=True, common_norm=False, common_grid=True, legend=False, ax=ax[0] )
    ax[0].xaxis.set_inverted(False)
    ax[0].yaxis.set_inverted(False)
    ax[0].set_xticks([-1, -0.5, 0, 0.5, 1])
    ax[0].set(title="Spearman's $\\rho$",xlabel="$\\rho$", xlim=(-1.1, 1.1))
    ax[0].grid(True)

    ax[1].axvline(x=NRMSE_thres, color="grey", ls="dashed", lw=1.5)
    ax[1].axvline(x=0.1, color="grey", ls="dotted", lw=1.5)
    ax[1].fill_between(x=[0.0, NRMSE_thres], y1=0, y2=1, color="lightgrey", alpha=0.3, label="Accepted fits")
    sns.kdeplot(data, x="NRMSE", hue=hue_key, hue_order=model_order, palette=color_dict, linewidth=1.5,
                cumulative=True, common_norm=False, common_grid=True,legend=True, ax=ax[1], )
    ax[1].xaxis.set_inverted(False)
    ax[1].grid(True)
    ax[1].set(title="NRMSE", xlabel="NRMSE")
    
    sns.move_legend(ax[1], loc=(1.02, 0.3), title=hue_key, frameon=False)

    plt.suptitle(f"Goodness of fit metrics distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/gof_distribution_{hue_key}.png", dpi=300, bbox_inches="tight")
    plt.show()


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
    plt.show()


def plot_accepted_heatmap(gof_src, hue_key="source"):

    model_order = ["Basic", "Rep_M", "Rep_Z"] 

    fig, ax = plt.subplots(1, len(model_order), figsize=(5*len(model_order), 4))

    for m, model_name in enumerate(model_order):
        print(f"Plot heatmap for model {model_name}...")

        merged = (gof_src[gof_src["model"] == model_name]).sort_values(["supercluster_no", "gene_id"])
        cluster_order = merged.supercluster.dropna().unique()
        
        sources = merged.source.unique()

        # Load original data
        accepted = merged[(merged["rho"] > RHO_thres) & (merged["NRMSE"] < NRMSE_thres) & (merged["MASE"] < 1.0)]

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
    plt.show()


def plot_params_cluster(ds_params, model_name):

    print(f"Plot fitted parameter distribution - model {model_name}")

    params_dict = {"Basic": (["delta_mean", "beta_mean"], [r"$\delta$",r"$\beta$"]),
                   "Rep_M": (["delta_m_mean", "delta_z_mean", "alpha_mean", "beta_mean", "t_zga_mean", "t_rep_mean"], [ r"$\delta_m$", r"$\delta_z$", r"$\alpha$", r"$\beta$", r"$t_{zga}$", r"$t_{reg}$"]),
                   "Rep_Z": (["delta_m_mean", "delta_z_mean", "alpha_mean", "beta_mean", "t_zga_mean", "t_rep_mean"], [ r"$\delta_m$", r"$\delta_z$", r"$\alpha$", r"$\beta$", r"$t_{zga}$", r"$t_{reg}$"])}
    
    params = params_dict[model_name][0]
    title = params_dict[model_name][1]

    fix, ax = plt.subplots(int(len(params)/2), 2, figsize = (8, int(len(params)/2*3)),)
    ax = ax.flatten()

    if "beta_mean" in params:
        data = ds_params[ds_params["beta_mean"] > 0.0]
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
    sns.move_legend(ax[-3], loc=(1.02, 0.3))
    plt.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/params_cluster_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_rep_params_violin(ds_params):

    print("plot params violin plot for Rep-Models")

    params = ["t_zga_mean", "t_rep_mean"]
    title = ["$t_{zga}$", "$t_{reg}$"]
    df = ds_params[ds_params["model"] != "Basic" ].copy()
    colors = sns.color_palette("Set2", n_colors=2)
    
    fig, ax = plt.subplots(1, len(params), figsize = (10, 4.5), sharey=True, sharex=True)

    for i, param in enumerate(params):
        sns.violinplot(data=df, x=param, y="supercluster", log_scale=True, hue="model", 
                        split=True, inner="quart",palette=colors, order=cluster_order,
                        legend= True if i == len(params)-1 else False, ax=ax[i])
        
        ax[i].set(title=title[i], xlabel=title[i])
        ax[i].grid(True)

    # move legend outside last axes
    sns.move_legend(ax[-1], loc=(1.01, 0.4), frameon=False)
    fig.suptitle("Estimated parameters of transcription onset $t_{zga}$ and regulation onset $t_{reg}$")
    plt.tight_layout()
    plt.savefig(f"{FIG_PATH}/violin_params.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_regulation_direction(ds_params):

    print("Plot regulation indicator r ...")

    df = ds_params[ds_params["model"] != "Basic" ].sort_values("supercluster_no")
    df["r"] = df["beta_mean"] / df["alpha_mean"]
    df = df[df["r"] > 1e-5]

    palette = sns.color_palette("Accent") 

    panels = ["Repression M-decay", "Repression Z-decay"]
    models = ["Rep_M", "Rep_Z"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True, sharex=True)
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
    plt.show()


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


## Upset plot

'''-----------------------
            PLOT 
--------------------------'''

#gof_all_combined, gof_src_combined, params_combined = combine_ds(save_csv=True)

'''-- GOF Metrics plots --'''
point_plot_metrics_all(pd.read_csv("results_summary/gof_all_combined.csv"))
plot_metrics_distribution_all(pd.read_csv("results_summary/gof_all_combined.csv"), hue_key = "model")
#plot_metrics_distribution(pd.read_csv("results_summary/gof_src_combined.csv"), hue_key="source")
#plot_accepted_heatmap(pd.read_csv("results_summary/gof_src_combined.csv"), hue_key="source")


'''-- Parameter plots --'''
#for model in params_combined.model.unique():
#    plot_params_cluster(pd.read_csv("results_summary/params_combined.csv"), model)
#plot_rep_params_violin(pd.read_csv("results_summary/params_combined.csv"))
#plot_regulation_direction(pd.read_csv("results_summary/params_combined.csv"))

'''-- Peak time --'''
#plot_peak_expression()


# sbatch plot.sh