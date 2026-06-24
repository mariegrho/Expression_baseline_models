import matplotlib.pyplot as plt
import arviz as az


def plot_model_results(results, gene_id, model_version, path=None, show=False):
    """Custom plot function to visualize the results."""

    obs = results.observed_data
    predictions = results.posterior_predictive.mean(dim=["draw", "chain"])
    result = results.posterior_model_fits.mean(dim=["draw", "chain"])
    hdi = az.hdi(results.posterior_model_fits, 0.95).y
    hdi_pp = az.hdi(results.posterior_predictive, 0.95).y
                                    
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    ax.plot(obs.time, obs.y, ls="", marker="o", ms=3, color="dimgray", alpha=.7, label="$y_{obs}$")

    ax.fill_between(obs.time, *hdi.values.T, color="grey", alpha=0.1, label="95% hdi (pmf)")
    ax.plot(result.time, result.y, color="k", label ="$total$")

    #ax.plot(predictions.time, predictions.y, color="grey", ls="dashed", label ="$y_{pp}$")
    #ax.fill_between(obs.time, *hdi_pp.values.T, color="0.8", alpha=0.2, label="95% hdi (pp)")

    if "M" in result.data_vars:
        ax.plot(result.time, result.M, label="M", c="indianred", ls="--") 
    if "Z" in result.data_vars:
        ax.plot(result.time, result.Z, label="Z", c="cornflowerblue", ls="-.")

    t_zga_fit = results.posterior.mean(dim=["draw", "chain"]).t_zga
    if t_zga_fit <= obs.time.max():
        ax.scatter(x=t_zga_fit, y=0, color="green", marker="^", label=f"t_zga = {t_zga_fit:.1f}")

    if model_version in ["Rep_Z", "Rep_M"]:
        t_reg_fit = results.posterior.mean(dim=["draw", "chain"]).dt_rep + t_zga_fit
        if t_reg_fit <= obs.time.max():
            ax.scatter(x=t_reg_fit, y=0, color="crimson", marker="^", label=f"t_rep = {t_reg_fit:.1f}")

    ax.set(xlabel="time (hpf)", ylabel="expression (TPM)", title=f"{gene_id} ({model_version})")
    ax.legend(loc=(1.01, 0.4))
    plt.tight_layout()

    if path is not None:
        plt.savefig(f"{path}/posterior_model_fits_{gene_id}_{model_version}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()