import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"
figures_dir.mkdir(parents=True, exist_ok=True)

l2_file = results_dir / "dc_subject_level_AIFI_L2_composition_clean.csv"
l3_file = results_dir / "dc_subject_level_AIFI_L3_composition_clean.csv"

l2 = pd.read_csv(l2_file)
l3 = pd.read_csv(l3_file)

AGE_ORDER = ["Young Adult", "Older Adult"]

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out

def star_label(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def subtype_stats(df, subtype_col, prefix):
    df = df[df["subject.ageGroup"].isin(AGE_ORDER)].copy()
    rows = []
    for subtype, sub in df.groupby(subtype_col):
        y = sub.loc[sub["subject.ageGroup"] == "Young Adult", "proportion"].dropna().astype(float)
        o = sub.loc[sub["subject.ageGroup"] == "Older Adult", "proportion"].dropna().astype(float)
        pval = mannwhitneyu(y, o, alternative="two-sided").pvalue if len(y) and len(o) else np.nan
        rows.append({
            subtype_col: subtype,
            "young_n": len(y),
            "older_n": len(o),
            "young_mean": y.mean() if len(y) else np.nan,
            "older_mean": o.mean() if len(o) else np.nan,
            "delta_older_minus_young": (o.mean() - y.mean()) if len(y) and len(o) else np.nan,
            "p_value": pval,
        })
    stats = pd.DataFrame(rows)
    stats["fdr"] = bh_fdr(stats["p_value"].values)
    stats = stats.sort_values("delta_older_minus_young", ascending=False)
    stats.to_csv(results_dir / f"{prefix}_subtype_stats.csv", index=False)
    return stats

def add_jitter(n, center, width=0.08, seed=42):
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-width, width, size=n)

def plot_boxstrip(df, stats_df, subtype_col, subtypes, outname, title):
    plot_df = df[df[subtype_col].isin(subtypes) & df["subject.ageGroup"].isin(AGE_ORDER)].copy()
    fig, axes = plt.subplots(1, len(subtypes), figsize=(4 * len(subtypes), 4.8), squeeze=False)
    axes = axes[0]

    for i, (ax, subtype) in enumerate(zip(axes, subtypes)):
        sub = plot_df[plot_df[subtype_col] == subtype].copy()
        young = sub.loc[sub["subject.ageGroup"] == "Young Adult", "proportion"].dropna().astype(float).values
        older = sub.loc[sub["subject.ageGroup"] == "Older Adult", "proportion"].dropna().astype(float).values

        ax.boxplot([young, older], positions=[0, 1], widths=0.5, patch_artist=False, showfliers=False)
        ax.scatter(add_jitter(len(young), 0, seed=100+i), young, s=20, alpha=0.8)
        ax.scatter(add_jitter(len(older), 1, seed=200+i), older, s=20, alpha=0.8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Young", "Older"])
        ax.set_title(subtype, fontsize=10)
        ax.set_ylabel("Donor-level proportion")

        row = stats_df.loc[stats_df[subtype_col] == subtype]
        if not row.empty:
            fdr = row["fdr"].iloc[0]
            delta = row["delta_older_minus_young"].iloc[0]
            ymax = max(np.max(young) if len(young) else 0, np.max(older) if len(older) else 0)
            pad = max(ymax * 0.12, 0.03)
            yline = ymax + pad
            ytext = ymax + 1.7 * pad
            ax.plot([0, 0, 1, 1], [yline, ytext, ytext, yline], lw=1)
            ax.text(0.5, ytext, f"{star_label(fdr)}\nΔ={delta:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{outname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{outname}.pdf", bbox_inches="tight")
    plt.close(fig)

# Stats
dc_l2_stats = subtype_stats(l2, "AIFI_L2", "dc_L2")
dc_l3_stats = subtype_stats(l3, "AIFI_L3", "dc_L3")

# Main panels
plot_boxstrip(
    l2, dc_l2_stats, "AIFI_L2",
    ["cDC2", "pDC", "cDC1", "ASDC"],
    "Figure_dc_L2_age_boxplots",
    "DC L2 composition shifts with age"
)

plot_boxstrip(
    l3, dc_l3_stats, "AIFI_L3",
    ["HLA-DRhi cDC2", "CD14+ cDC2", "pDC", "cDC1", "ISG+ cDC2"],
    "Figure_dc_L3_age_boxplots",
    "DC L3 composition shifts with age"
)

print("Done.")
print(results_dir / "dc_L2_subtype_stats.csv")
print(results_dir / "dc_L3_subtype_stats.csv")
print(figures_dir / "Figure_dc_L2_age_boxplots.png")
print(figures_dir / "Figure_dc_L3_age_boxplots.png")
