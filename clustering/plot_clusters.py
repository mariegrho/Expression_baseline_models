"""
plot_clusters.py

Visualizatios for fuzzy gene expression clustering.

Inputs:
    - _gene_cluster_annotation_.nc
    - _superclusters_centers.nc
    - _superclusters_membership.nc 
    - normalized trajectories 

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("Set1")


# =====================================================
# 1. Plot cluster prototype trajectories
# =====================================================

def plot_cluster_centers(centers_da):

    plt.figure(figsize=(12, 6))

    for k in centers_da.cluster.values:

        plt.plot(
            centers_da.time.values,
            centers_da.sel(cluster=k).values,
            linewidth=2,
            label=f"Cluster {k}",
            c=colors[k] )

    plt.xlabel("Time")
    plt.ylabel("Normalized expression")
    plt.title("Fuzzy cluster prototype trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/cluster_centers.png")
    #plt.show()
    plt.close()


# =====================================================
# 2. Plot clusters with variability (if membership provided)
# =====================================================

def plot_cluster_with_variance(trajectories_da, labels, n_clusters=None):

    data = trajectories_da.values

    if n_clusters is None:
        n_clusters = labels.max() + 1

    plt.figure(figsize=(12, 6))

    for k in range(n_clusters):

        idx = labels == k
        if idx.sum() == 0:
            continue

        cluster_data = data[idx]

        mean = cluster_data.mean(axis=0)
        std = cluster_data.std(axis=0)

        plt.plot(trajectories_da.time.values,mean,label=f"Cluster {k}", c=colors[k] )
        plt.fill_between(trajectories_da.time.values,mean - std,mean + std,alpha=0.2, color=colors[k])

    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title("Cluster mean trajectories with variability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/cluster_with_variance.png")
    #plt.show()
    plt.close()


# =====================================================
# 3. Plot fuzzy membership heatmap
# =====================================================

def plot_membership_heatmap(membership_da, sc, max_genes=1000):

    membership = membership_da.values
    membership = membership[:max_genes]

    dominant_cluster = np.argmax(membership, axis=1)
    order = np.argsort(dominant_cluster)
    membership_sorted = membership[order]
    n_clusters = membership.shape[1]


    plt.figure(figsize=(8, 8))
    plt.imshow(membership_sorted,aspect="auto",cmap="viridis")
    plt.colorbar(label="Membership strength")

    plt.xticks(np.arange(n_clusters), labels=[f"Cluster {k}" for k in range(n_clusters)])
    plt.ylabel("Genes (subset)")
    plt.title(f"Fuzzy membership heatmap, Supercluster {sc}")

    plt.tight_layout()
    plt.savefig(f"figs/membership_heatmap_Supercluster_{sc}.png")
    #plt.show()
    plt.close()



def plot_mean_membership_heatmap(membership_da, sc):
    membership = membership_da.values
    n_clusters = membership.shape[1]

    dominant_cluster = np.argmax(membership, axis=1)

    mean_membership_matrix = np.zeros((n_clusters, n_clusters))
    for k in range(n_clusters):
        genes_in_k = membership[dominant_cluster == k]
        if len(genes_in_k) > 0:
            mean_membership_matrix[k] = genes_in_k.mean(axis=0)
        else:
            mean_membership_matrix[k] = np.nan

    plt.figure(figsize=(6, 4))
    plt.imshow(mean_membership_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean membership strength")

    plt.xticks(np.arange(n_clusters), labels=[f"Cluster {k}" for k in range(n_clusters)], rotation=45, ha="right")
    plt.yticks(np.arange(n_clusters), labels=[f"Assigned to {k}" for k in range(n_clusters)])
    plt.xlabel("Mean membership in cluster")
    plt.ylabel("Dominant cluster")
    plt.title(f"Mean fuzzy membership per assigned cluster, Supercluster {sc}")

    # annotate cells with values
    for i in range(n_clusters):
        for j in range(n_clusters):
            val = mean_membership_matrix[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                          color="white" if val < mean_membership_matrix.max() / 2 else "black",
                          fontsize=7)

    plt.tight_layout()
    plt.savefig(f"figs/mean_membership_heatmap_Supercluster_{sc}.png")
    plt.close()


# =====================================================
# 4. Plot representative genes per cluster
# =====================================================

def plot_representative_genes(trajectories_da, labels, cluster_id, top_n=50):

    data = trajectories_da.values
    idx = np.where(labels == cluster_id)[0]

    if len(idx) == 0:
        print(f"No genes found for cluster {cluster_id}")
        return

    plt.figure(figsize=(10, 5))

    for i in idx[:top_n]:
        plt.plot(trajectories_da.time.values,data[i],alpha=0.4)

    mean_curve = data[idx].mean(axis=0)

    plt.plot(
        trajectories_da.time.values,
        mean_curve,
        color="black",
        linewidth=3,
        label="Mean"
    )

    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title(f"Cluster {cluster_id} representative genes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/representative_genes_Cluster_{cluster_id}.png")
    #plt.show()
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_super_subcluster_grid1(trajectories_da, super_labels, sub_labels, figsize=None):

    #cluster_names = {0 : "SD", 1 : "DSD", 2 : "SU", 3 : "DSU"}
    cluster_names = {0 : "SD", 1:"SU", 2:"DSU", 3: "SU", 4: "", np.nan: "N/A"}
    col_c = sns.color_palette("Set1", n_colors=8)  
    cluster_color_dict = {"SD": col_c[0], "DSD": col_c[1], "SU": col_c[2], "DSU": col_c[3], "TU-SU": col_c[4], "TU": col_c[7], }

    data = trajectories_da.values
    t = trajectories_da.time.values

    super_ids = np.unique(super_labels)
    # group subclusters per supercluster
    groups = {sc: np.unique(sub_labels[(super_labels == sc) & (sub_labels >= 0)]) for sc in super_ids}

    # estimate layout
    nrows = len(super_ids)
    ncols = max(len(v) for v in groups.values()) + 1

    if figsize is None:
        figsize = (2 * ncols, 1.5 * nrows)

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize,sharex=True,sharey="row")
    axes = np.atleast_2d(axes)

    for i, sc in enumerate(super_ids):
        sub_ids = groups[sc]

        sc_name = cluster_names[sc]

        # plot mean super cluster in the first row
        idx = np.where((super_labels == sc))[0]
        subset = data[idx]
        mean_sc = subset.mean(axis=0)
        
        axes[i, 0].plot(t, mean_sc, linewidth=2, color=cluster_color_dict[sc_name])
        axes[i, 0].set_title(f"Supercluster {sc_name} ({len(subset)})", fontsize=9)

        for j, sb in enumerate(sub_ids):

            if j >= ncols:
                break

            ax = axes[i, j+1]
            idx = np.where((super_labels == sc) &(sub_labels == sb))[0]
            if len(idx) == 0:
                ax.set_visible(False)
                continue

            subset = data[idx]
            mean = subset.mean(axis=0)

            # plot individual trajectories (limit for readability)
            for g in range(min(100, len(subset))):
                ax.plot(t, subset[g], alpha=0.15, linewidth=0.8, color="grey")
            ax.plot(t, mean, linewidth=2, color=cluster_color_dict[sc_name])
            ax.set_title(f"{sc_name} - {sb} ({len(subset)})", fontsize=9)
            ax.set_ylim(-2.5, 2.5)

        # hide unused columns in this row
        for j in range(len(sub_ids) + 1, ncols):
            axes[i, j].set_visible(False)

    fig.suptitle("Subcluster trajectories - shape-based FCM cluster", fontsize=12)
    plt.tight_layout()
    plt.savefig("figs/super_subcluster_grid.png", dpi=300)
    #plt.show()
    plt.close()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    NORMALISATION = "zscore"
    t_end = 120
    data = "all"
    #data = "joint"

    print(f"[Info] Plotting clusters for dataset: {data}, normalization: {NORMALISATION}, t_end: {t_end}")

    labels = xr.load_dataset(f"results/{data}_gene_cluster_annotation_{NORMALISATION}.nc")
    #labels = xr.load_dataset(f"results/joint_gene_params_cluster_annotation.nc").tpm

    super_labels = labels.supercluster.values
    sub_labels = labels.subcluster.values

    trajectories_da = xr.load_dataarray(f"results/{data}_normalized_trajectories_{t_end}_{NORMALISATION}.nc")
    #trajectories_da = xr.load_dataarray("joint_simulation_results.nc")

    trajectories_da = trajectories_da.sel(ensembl_gene_id=labels.ensembl_gene_id)

    print("[Info] Plotting supercluster x subcluster grid...")

    # Plot super × sub cluster grid
    plot_super_subcluster_grid1(trajectories_da, super_labels, sub_labels)

    print("[Info] Plotting supercluster centers...")
    
    plot_cluster_centers(xr.load_dataarray(f"results/{data}_superclusters_centers.nc"))
    plot_cluster_with_variance(trajectories_da, super_labels,n_clusters=None)

    for cluster_id in np.unique(super_labels):
        plot_representative_genes(trajectories_da, super_labels, cluster_id, top_n=50)

    print("[Info] Plotting fuzzy cluster membership...")
    
    membership_da = xr.load_dataarray(f"results/{data}_superclusters_membership.nc")
    #plot_membership_heatmap(membership_da, "All", max_genes=500)
    plot_mean_membership_heatmap(membership_da, sc=data)
    
    for sc in np.unique(super_labels):
        membership_da = xr.load_dataarray(f"results/{data}_supercluster_{sc}_membership.nc")
        #plot_membership_heatmap(membership_da, sc, max_genes=500)
        plot_mean_membership_heatmap(membership_da, sc)