import numpy as np
import xarray as xr

from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
N_CLUSTERS = 6
N_JOBS = -1          # parallel; -1: use all available CPA cores
LENGTH_SCALE = 5.0  # GP smoothness


# =========================
# GP FITTING PER GENE
# =========================
def fit_gp_for_gene(gene_id, ds, time_grid, study_encoder):

    gene = ds.sel(ensembl_gene_id=gene_id)

    times_all = []
    expr_all = []
    study_all = []

    for i, source in enumerate(ds.source.values):

        y = gene.tpm.sel(source=source).values
        t = gene.time.values

        mask = ~np.isnan(y)

        if mask.sum() < 3:
            continue

        times_all.append(t[mask])
        expr_all.append(y[mask])
        study_all.extend([source] * mask.sum())

    if len(times_all) == 0:
        return None

    t = np.concatenate(times_all)
    y = np.concatenate(expr_all)
    s = np.array(study_all).reshape(-1, 1)

    study_onehot = study_encoder.transform(s)

    X = np.column_stack([t, study_onehot])

    kernel = ConstantKernel(1.0) * RBF(length_scale=LENGTH_SCALE) + WhiteKernel()

    gp = GaussianProcessRegressor(kernel=kernel,normalize_y=True,n_restarts_optimizer=2)

    try:
        gp.fit(X, y)
    except Exception:
        return None

    # ---- Predict on reference grid ----
    # assume study effect = average (zeros)
    study_ref = np.zeros((len(time_grid), study_onehot.shape[1]))
    Xpred = np.column_stack([time_grid, study_ref])

    mean = gp.predict(Xpred)

    return mean


# =========================
# GAUSSIAN MIXTURE PIPELINE
# =========================
def run_pipeline(ds):

    genes = ds.ensembl_gene_id.values[0:5000]
    time_grid = ds.time.values

    # encode study/source
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(ds.source.values.reshape(-1, 1))

    print("Fitting GP per gene... (this may take a while)")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(fit_gp_for_gene)(g, ds, time_grid, encoder)
        for g in genes
    )

    # filter failed genes
    curves = []
    valid_genes = []

    for g, r in zip(genes, results):
        if r is not None and np.isfinite(r).all():
            curves.append(r)
            valid_genes.append(g)

    curves = np.vstack(curves)

    print(f"Valid genes: {len(valid_genes)}")

    # =========================
    # DIMENSIONALITY REDUCTION
    # =========================
    pca = PCA(n_components=20)
    emb = pca.fit_transform(curves)

    # =========================
    # CLUSTERING
    # =========================
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        random_state=0
    )

    labels = gmm.fit_predict(emb)

    return valid_genes, labels, emb, curves, time_grid


import numpy as np
import matplotlib.pyplot as plt

# =========================
# PLOT RESULTS
# =========================
def plot_cluster_means(curves, labels, time_grid, n_clusters=None):

    curves = np.asarray(curves)
    labels = np.asarray(labels)

    if n_clusters is None:
        n_clusters = labels.max() + 1

    plt.figure(figsize=(12, 6))

    for k in range(n_clusters):

        idx = labels == k
        if idx.sum() == 0:
            continue

        cluster_curves = curves[idx]
        mean_curve = cluster_curves.mean(axis=0)
        std_curve = cluster_curves.std(axis=0)

        plt.plot(time_grid, mean_curve, label=f"Cluster {k}")
        plt.fill_between(time_grid, mean_curve - std_curve,mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Time")
    plt.ylabel("Expression (GP-smoothed)")
    plt.title("Cluster mean gene expression trajectories")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("GP_clustering.png")
    plt.show()


import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def select_k_bic(embeddings, k_range=range(2, 31)):

    bics = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",random_state=0)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))

    best_k = k_range[np.argmin(bics)]

    plt.plot(list(k_range), bics, marker='o')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("BIC (lower is better)")
    plt.title("Model selection for GP gene clusters")
    plt.show()

    return best_k, bics


# =========================
# SAVE RESULTS
# =========================
def save_results(valid_genes, labels, curves, time_grid, path="gp_clusters.npz"):

    np.savez(path,genes=np.array(valid_genes),
        labels=labels,curves=curves,time=time_grid)
    print(f"Saved to {path}")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    # assume dataset is loaded as ds
    ds =  xr.load_dataset("genes_tpms_white_pauli_JN_BK_mean.nc")
    #ds["tpm"] = np.log2(ds.tpm+1)
    ds["tpm"] = (ds["tpm"] - ds["tpm"].mean(dim=("time", "source"))) / ds["tpm"].std(dim=("time", "source"))

    #X = ds.tpm
    #ds["tpm"] = (X - X.mean(dim=("time", "source"))) / X.std(dim=("time", "source"))

    genes, labels, emb, curves, time_grid = run_pipeline(ds)
    best_k, bics = select_k_bic(emb)
    print("Best K:", best_k)

    plot_cluster_means(curves, labels, time_grid)

    save_results(genes, labels, curves, time_grid)