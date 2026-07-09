import matplotlib.pyplot as plt
import arviz as az

def plot_model_results(results, gene_id, model_version, path=None, show=False):
    """Custom plot function to visualize the results."""

    obs = results.observed_data
    result = results.posterior_model_fits.mean(dim=["draw", "chain", "source"])
    hdi = az.hdi(results.posterior_model_fits, 0.95).y.mean("source")
                                    
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    markers = ["o", "s", "x", "^"]
    colors =    [(0.6551633986928105, 0.6405228758169934, 0.8091503267973856), 
                (0.5338562091503267, 0.5019607843137255, 0.7359477124183006), 
                (0.41856209150326795, 0.38614379084967315, 0.5994771241830065), 
                (0.30928104575163395, 0.2930718954248366, 0.39973856209150327)]
    for i, src in enumerate(obs.source):
        ax.plot(obs.time, obs.y.sel(source=src), ls="", marker=markers[i], ms=4, color=colors[i], alpha=.9, label=src.item())

    ax.fill_between(result.time, *hdi.values.T, color="grey", alpha=0.2, label="95% hdi")
    ax.plot(result.time, result.y, color="k", lw=2, label ="$total$")

    if "M" in result.data_vars:
        ax.plot(result.time, result.M, label="M", c="indianred", ls="--") 
    if "Z" in result.data_vars:
        ax.plot(result.time, result.Z, label="Z", c="cornflowerblue", ls="-.")

    if model_version != "Basic":
        t_zga_fit = results.posterior.mean(dim=["draw", "chain",]).t_zga
        if t_zga_fit <= obs.time.max():
            ax.scatter(x=t_zga_fit, y=0, color="green", marker="^", label="$t_{zga}$"+f" = {t_zga_fit:.1f}")

    if model_version in ["Rep_Z", "Rep_M"]:
        t_reg_fit = results.posterior.mean(dim=["draw", "chain",]).t_rep
        if t_reg_fit <= obs.time.max():
            ax.scatter(x=t_reg_fit, y=0, color="crimson", marker="^", label="$t_{reg}$"+f" = {t_reg_fit:.1f}")

    ax.set(xlabel="time (hpf)", ylabel="expression (TPM)", title=f"{gene_id} ({model_version})")
    ax.legend(loc=(1.01, 0.1))
    plt.tight_layout()

    if path is not None:
        plt.savefig(f"{path}/posterior_model_fits_{gene_id}_{model_version}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()