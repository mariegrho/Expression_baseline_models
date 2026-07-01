import numpy as np
import xarray as xr

from pygam import LinearGAM, s
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
N_CLUSTERS = 8
N_JOBS = -1
N_SPLINES = 5   # GAM smoothness control


# =========================
# FIT GAM FOR ONE GENE
# =========================
def fit_and_predict_gene(gene_id, ds, time_grid):

    gene = ds.sel(ensembl_gene_id=gene_id)

    all_t = []
    all_y = []

    for source in ds.source.values:

        y = gene.tpm.sel(source=source).values
        t = gene.time.values

        mask = ~np.isnan(y)

        if mask.sum() > 0:
            all_t.append(t[mask])
            all_y.append(y[mask])

    if len(all_t) == 0:
        return None

    t = np.concatenate(all_t)
    y = np.concatenate(all_y)

    # need at least a few points
    if len(t) < 5:
        return None

    try:
        gam = LinearGAM(s(0, n_splines=N_SPLINES))
        gam.fit(t, y)

        pred = gam.predict(time_grid)

        return pred

    except Exception:
        return None


# =========================
# MAIN PIPELINE
# =========================
def run_gam_pipeline(ds):

    genes = ds.ensembl_gene_id.values[:5000]
    time_grid = ds.time.values

    print("Fitting GAMs per gene...")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(fit_and_predict_gene)(g, ds, time_grid)
        for g in genes
    )

    curves = []
    valid_genes = []

    for g, r in zip(genes, results):
        if r is not None and np.isfinite(r).all():
            curves.append(r)
            valid_genes.append(g)

    curves = np.vstack(curves)

    print(f"Valid genes: {len(valid_genes)}")
    '''
    # =========================
    # PCA embedding
    # =========================
    pca = PCA(n_components=10)
    emb = pca.fit_transform(curves)

    # =========================
    # Clustering
    # =========================
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        random_state=0
    )

    labels = gmm.fit_predict(emb)
    '''
    emb = []
    labels = GaussianMixture(n_components=6).fit_predict(curves)

    return valid_genes, labels, emb, curves, time_grid


# =========================
# PLOTTING CLUSTER MEANS
# =========================
import matplotlib.pyplot as plt

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

        mean = cluster_curves.mean(axis=0)
        std = cluster_curves.std(axis=0)

        plt.plot(time_grid, mean, label=f"Cluster {k}")
        plt.fill_between(time_grid, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Time")
    plt.ylabel("Expression (GAM-smoothed)")
    plt.title("GAM-based gene expression clusters")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# OPTIONAL: SELECT K (BIC)
# =========================
def select_k_bic(emb, k_range=range(2, 25)):

    bics = []

    for k in k_range:

        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gmm.fit(emb)
        bics.append(gmm.bic(emb))

    best_k = k_range[np.argmin(bics)]

    plt.plot(list(k_range), bics, marker='o')
    plt.xlabel("K")
    plt.ylabel("BIC")
    plt.title("GAM embedding clustering: model selection")
    plt.show()

    return best_k, bics


# =========================
# RUN EVERYTHING
# =========================
if __name__ == "__main__":

    ds =  xr.load_dataset("genes_tpms_white_pauli_JN_BK_mean.nc")
    X = ds.tpm
    ds["tpm"] = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    genes, labels, emb, curves, time_grid = run_gam_pipeline(ds)

    plot_cluster_means(curves, labels, time_grid)

    #best_k, bics = select_k_bic(emb)
    #print("Best K:", best_k)