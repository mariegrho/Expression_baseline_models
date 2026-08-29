"""
Plotting helpers for summarizing the fit-quality scorecard's triage columns
(converged, good_fit, reliable_loo, well_calibrated, weakly_identified,
status) across all genes.

Requires: matplotlib, seaborn, pandas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

col_m = sns.color_palette("Dark2")  
mod_color_dict = {"Basic": col_m[7],"M-decay": col_m[1], "Z-decay": col_m[4], "Rep_V": col_m[2]} 
model_order = ["Basic", "M-decay", "Z-decay"]

col_c = sns.color_palette("Set1", n_colors=4)  
cluster_color_dict = {"SD": col_c[0], "DSD": col_c[1], "SU": col_c[2], "DSU": col_c[3], "N/A":"grey"}

def format_label(label):
    return label.replace("_", " ").title()



def plot_pass_rates(df: pd.DataFrame,
                     criteria: list = ("converged", "good_fit", "pattern_reproduction", "reliable_loo", "well_calibrated"),
                     labels: dict = None, hue: str = None, ax=None, fig_path: str="figures", title: str = None):
    """
    Horizontal bar chart comparing the % of genes passing each boolean
    criterion. If `hue` is given (e.g. hue="model"), bars are grouped
    per criterion so pass rates can be compared across model variants
    directly, instead of running this plot once per variant.
    """
    labels = labels or {c: c.replace("_", " ").title() for c in criteria}


    rows = []
    if hue is not None:
        
        for c in criteria:
            for h_val, sub in df.groupby(hue):
                valid = sub[c].dropna()
                n_valid = len(valid)
                pass_pct = valid.mean() if n_valid > 0 else np.nan
                rows.append({"criterion": labels[c], hue: h_val, "pass_pct": pass_pct, "n_valid": n_valid})
    else:
        for c in criteria:
            valid = df[c].dropna()
            n_valid = len(valid)
            pass_pct = valid.mean() if n_valid > 0 else np.nan
            rows.append({"criterion": labels[c], "pass_pct": pass_pct, "n_valid": n_valid})

    plot_df = pd.DataFrame(rows)

    # order criteria by overall pass rate (averaged across hue groups if present)
    order = (plot_df.groupby("criterion")["pass_pct"].mean().sort_values().index.tolist())

    n_hue_levels = df[hue].nunique() if hue is not None else 1
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 0.2 * len(criteria) * max(n_hue_levels, 1) + 1.5))

    n_total = len(df)/n_hue_levels if hue is not None else len(df)
    if hue is not None:
        palette = {"model":mod_color_dict, "supercluster":cluster_color_dict}[hue]
        sns.barplot(data=plot_df, y="criterion", x="pass_pct", hue=hue, palette=palette, order=order, ax=ax, orient="h")
        ax.legend(title=hue.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        sns.barplot(data=plot_df, y="criterion", x="pass_pct", order=order, ax=ax, orient="h", color="#4C72B0")

    for container in ax.containers:
        #ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)
        ax.bar_label(container, labels=[f"{v.get_width()*100:.1f}%" for v in container], padding=3, fontsize=8)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Proportion of genes passing")
    ax.set_ylabel("")

    
    if title is not None:
        ax.set_title(f"Proportion of genes passing fit-quality criteria (n = {int(len(df)):,} genes) (model = {title})")
    else:
        ax.set_title(f"Proportion of genes passing fit-quality criteria (n = {int(n_total):,} genes)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/model_fit_eval_pass_rates_{hue}_{title}.png", dpi=300)

    return ax


def plot_status_breakdown(df: pd.DataFrame, status_col: str = "status", status_dict: dict = None,
                           hue: str = None, ax=None, palette: dict = None, fig_path: str ="figures", title: str = None):
    """
    Horizontal, frequency-sorted bar chart of the mutually-exclusive
    `status` triage bucket. If `hue` is given, bars are grouped by hue
    and shown as WITHIN-GROUP proportions (not raw counts) -- otherwise a
    model variant with more genes would visually dominate regardless of
    actual fit quality, which isn't the comparison you want.
    """

    def status_label(x):
        return status_dict.get(x, x)


    if hue is not None:
        palette = {"model":mod_color_dict, "supercluster":cluster_color_dict}[hue]

        prop_df = (df.groupby(hue)[status_col].value_counts(normalize=True).rename("proportion").reset_index()) 

        # Replace status codes with the human-readable labels from status_dict
        prop_df[status_col] = prop_df[status_col].map(status_label)

        order = df[status_col].value_counts().sort_values().index.tolist()
        order = [status_label(x) for x in order]

        if ax is None:
            n_hue_levels = df[hue].nunique()
            fig, ax = plt.subplots(figsize=(7.5, 0.2 * len(order) * n_hue_levels + 1.5) )

        sns.barplot(data=prop_df, y=status_col, x="proportion", hue=hue, palette=palette, order=order, ax=ax, orient="h")
        for container in ax.containers:
            ax.bar_label(container, labels=[f"{v.get_width()*100:.1f}%" for v in container], padding=3, fontsize=8)
        ax.legend(title=hue.replace("_", " ").title(), loc="upper right", frameon=False)
        #ax.set_yticklabels([format_label(t.get_text()) for t in ax.get_yticklabels()])        
        ax.set_xlabel("Proportion of genes")
        ax.set_xlim(0, 1.0)

        if title is not None:
            ax.set_title(f"GOF summary by {hue} (model = {title})")
        else:
            ax.set_title(f"GOF summary by {hue} (n = {int(len(df)/3):,} genes)") 

    else:
        counts = df[status_col].value_counts()
        pct = (counts / counts.sum() * 100)
        plot_df = pd.DataFrame({"count": counts, "pct": pct}).sort_values("count")

        # map raw status codes to the descriptions in status_dict
        plot_df.index = plot_df.index.map(status_label)

        default_palette = {
            status_dict["fully_accepted"]: "#55A868",
            status_dict["non_con"]: "#C44E52",
            status_dict["poor_fit"]: "#DD8452",
            status_dict["pattern"]: "#CCB974",
            status_dict["low_error"]: "#8172B2",
        }
    
        palette = palette or default_palette
        colors = [palette.get(status, "#999999") for status in plot_df.index]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 0.6 * len(plot_df) + 1.5))

        #bars = ax.barh([format_label(status) for status in plot_df.index], plot_df["count"], color=colors)
        bars = ax.barh(plot_df.index, plot_df["count"], color=colors)
        for bar, (status, row) in zip(bars, plot_df.iterrows()):
            ax.text( bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['count']:,}  ({row['pct']:.1f}%)", va="center", ha="left", fontsize=9, )
        ax.set_xlim(0, counts.max() * 1.2)
        ax.set_xlabel("Number of genes")
        ax.set_title(f"Status breakdown  (n = {counts.sum():,} genes)")

    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_path}/model_fit_status_{hue}_{title}.png", dpi=300)

    return ax


def plot_scorecard_summary(df: pd.DataFrame,
                            criteria: list = ("converged", "good_fit", "reliable_loo", "well_calibrated"),
                            status_col: str = "status", hue: str = None):
    """Combined figure: pass-rate panel + status breakdown panel, stacked."""
    n_hue_levels = df[hue].nunique() if hue is not None else 1
    fig, axes = plt.subplots( 2, 1, figsize=(8, (len(criteria) + df[status_col].nunique()) * 0.3 * n_hue_levels + 3),
        gridspec_kw={"height_ratios": [len(criteria), df[status_col].nunique()]}, )
    plot_pass_rates(df, criteria=criteria, hue=hue, ax=axes[0])
    plot_status_breakdown(df, status_col=status_col, hue=hue, ax=axes[1])
    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    # Example, single model:
    # res_z = pd.read_csv("fit_scorecard_flagged.csv")
    # fig, axes = plot_scorecard_summary(res_z)
    #
    # Example, comparing model variants (requires a "model" column in the
    # scorecard identifying which variant/timepoint each row came from --
    # e.g. add this when concatenating multiple scorecards together):
    # combined = pd.concat([
    #     pd.read_csv("fit_scorecard_RepM_flagged.csv").assign(model="Rep_M"),
    #     pd.read_csv("fit_scorecard_RepZ_flagged.csv").assign(model="Rep_Z"),
    # ])
    # fig, axes = plot_scorecard_summary(combined, hue="model")
    # fig.savefig("scorecard_summary_by_model.png", dpi=150, bbox_inches="tight")
    pass
