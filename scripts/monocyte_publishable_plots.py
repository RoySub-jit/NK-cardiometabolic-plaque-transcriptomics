import math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

project_dir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis")
results_dir = project_dir / "results" / "first_dataset"
figures_dir = project_dir / "figures" / "first_dataset"
figures_dir.mkdir(parents=True, exist_ok=True)

l2_file = results_dir / "monocyte_subject_level_AIFI_L2_composition_clean.csv"
l3_file = results_dir / "monocyte_subject_level_AIFI_L3_composition_clean.csv"

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

        if len(y) == 0 or len(o) == 0:
            pval = np.nan
        else:
            pval = mannwhitneyu(y, o, alternative="two-sided").pvalue

        rows.append({
            subtype_col: subtype,
            "young_n": len(y),
            "older_n": len(o),
            "young_mean": y.mean() if len(y) else np.nan,
            "older_mean": o.mean() if len(o) else np.nan,
            "young_median": y.median() if len(y) else np.nan,
            "older_median": o.median() if len(o) else np.nan,
            "delta_older_minus_young": (o.mean() - y.mean()) if len(y) and len(o) else np.nan,
            "p_value": pval,
        })

    stats = pd.DataFrame(rows)
    valid = stats["p_value"].notna()
    stats["fdr"] = np.nan
    if valid.any():
        stats.loc[valid, "fdr"] = bh_fdr(stats.loc[valid, "p_value"].values)

    stats = stats.sort_values("delta_older_minus_young", ascending=False)
    stats.to_csv(results_dir / f"{prefix}_subtype_stats.csv", index=False)
    return stats


def add_jitter(n, center, width=0.08):
    if n == 0:
        return np.array([])
    rng = np.random.default_rng(42)
    return center + rng.uniform(-width, width, size=n)


def plot_boxstrip(df, stats_df, subtype_col, subtypes, outname, title):
    plot_df = df[df[subtype_col].isin(subtypes) & df["subject.ageGroup"].isin(AGE_ORDER)].copy()

    n_panels = len(subtypes)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.8), squeeze=False)
    axes = axes[0]

    for ax, subtype in zip(axes, subtypes):
        sub = plot_df[plot_df[subtype_col] == subtype].copy()
        young = sub.loc[sub["subject.ageGroup"] == "Young Adult", "proportion"].dropna().astype(float).values
        older = sub.loc[sub["subject.ageGroup"] == "Older Adult", "proportion"].dropna().astype(float).values

        bp = ax.boxplot(
            [young, older],
            positions=[0, 1],
            widths=0.5,
            patch_artist=False,
            showfliers=False
        )

        ax.scatter(add_jitter(len(young), 0), young, s=20, alpha=0.8)
        ax.scatter(add_jitter(len(older), 1), older, s=20, alpha=0.8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Young", "Older"], rotation=0)
        ax.set_title(subtype, fontsize=10)
        ax.set_ylabel("Donor-level proportion")

        stat_row = stats_df.loc[stats_df[subtype_col] == subtype]
        if not stat_row.empty:
            fdr = stat_row["fdr"].iloc[0]
            delta = stat_row["delta_older_minus_young"].iloc[0]
            ymax = max(np.max(young) if len(young) else 0, np.max(older) if len(older) else 0)
            yline = ymax * 1.08 if ymax > 0 else 0.05
            ytext = ymax * 1.14 if ymax > 0 else 0.06
            ax.plot([0, 0, 1, 1], [yline, ytext, ytext, yline], lw=1)
            ax.text(0.5, ytext, f"{star_label(fdr)}\nΔ={delta:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{outname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{outname}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_effect_summary(stats_df, subtype_col, outname, title):
    df = stats_df.copy()
    df = df.sort_values("delta_older_minus_young", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.45 * len(df))))
    y = np.arange(len(df))
    x = df["delta_older_minus_young"].values

    ax.axvline(0, lw=1)
    ax.barh(y, x)
    ax.set_yticks(y)
    ax.set_yticklabels(df[subtype_col].tolist(), fontsize=9)
    ax.set_xlabel("Mean donor-level proportion difference\n(Older Adult − Young Adult)")
    ax.set_title(title)

    for yi, (_, row) in enumerate(df.iterrows()):
        label = star_label(row["fdr"])
        ax.text(
            row["delta_older_minus_young"] + (0.002 if row["delta_older_minus_young"] >= 0 else -0.002),
            yi,
            label,
            va="center",
            ha="left" if row["delta_older_minus_young"] >= 0 else "right",
            fontsize=9
        )

    fig.tight_layout()
    fig.savefig(figures_dir / f"{outname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{outname}.pdf", bbox_inches="tight")
    plt.close(fig)


# Compute stats
l2_stats = subtype_stats(l2, "AIFI_L2", "monocyte_L2")
l3_stats = subtype_stats(l3, "AIFI_L3", "monocyte_L3")

# Chosen panel sets
l2_subtypes = ["CD14 monocyte", "CD16 monocyte", "Intermediate monocyte"]
l3_subtypes = [
    "Core CD14 monocyte",
    "Core CD16 monocyte",
    "Intermediate monocyte",
    "ISG+ CD14 monocyte",
    "ISG+ CD16 monocyte",
]

# Plot main panels
plot_boxstrip(
    l2, l2_stats, "AIFI_L2", l2_subtypes,
    "Figure_monocyte_L2_age_boxplots",
    "Monocyte L2 composition shifts with age"
)

plot_boxstrip(
    l3, l3_stats, "AIFI_L3", l3_subtypes,
    "Figure_monocyte_L3_age_boxplots",
    "Monocyte L3 substate shifts with age"
)

# Plot effect summaries
plot_effect_summary(
    l2_stats, "AIFI_L2",
    "Figure_monocyte_L2_effect_summary",
    "Monocyte L2 donor-level age effect summary"
)

plot_effect_summary(
    l3_stats, "AIFI_L3",
    "Figure_monocyte_L3_effect_summary",
    "Monocyte L3 donor-level age effect summary"
)

print("Done.")
print(results_dir / "monocyte_L2_subtype_stats.csv")
print(results_dir / "monocyte_L3_subtype_stats.csv")
print(figures_dir / "Figure_monocyte_L2_age_boxplots.png")
print(figures_dir / "Figure_monocyte_L3_age_boxplots.png")
print(figures_dir / "Figure_monocyte_L2_effect_summary.png")
print(figures_dir / "Figure_monocyte_L3_effect_summary.png")
