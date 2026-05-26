from pathlib import Path
import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm

infile = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/DC_meta_combined_sample_level.csv")
outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")

df = pd.read_csv(infile)

endpoints = [
    "cdc2_score_mean",
    "prop_cdc2_high",
]

def hedges_g(asym, sym):
    # effect defined as asymptomatic - symptomatic
    n1 = len(asym)
    n2 = len(sym)
    m1 = np.mean(asym)
    m2 = np.mean(sym)
    s1 = np.std(asym, ddof=1)
    s2 = np.std(sym, ddof=1)

    sp = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan, np.nan

    d = (m1 - m2) / sp
    J = 1 - (3 / (4*(n1+n2) - 9))
    g = J * d

    var_g = ((n1 + n2) / (n1*n2)) + (g**2 / (2*(n1 + n2 - 2)))
    se_g = np.sqrt(var_g)

    return g, var_g, m1, m2

def random_effects_meta(effects, variances):
    effects = np.array(effects, dtype=float)
    variances = np.array(variances, dtype=float)
    w_fixed = 1 / variances
    fixed_mean = np.sum(w_fixed * effects) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (effects - fixed_mean)**2)
    df_q = len(effects) - 1
    c = np.sum(w_fixed) - (np.sum(w_fixed**2) / np.sum(w_fixed))
    tau2 = max(0, (Q - df_q) / c) if c > 0 else 0
    w_random = 1 / (variances + tau2)
    pooled = np.sum(w_random * effects) / np.sum(w_random)
    se = np.sqrt(1 / np.sum(w_random))
    z = pooled / se
    p = 2 * (1 - norm.cdf(abs(z)))
    ci_low = pooled - 1.96 * se
    ci_high = pooled + 1.96 * se
    i2 = max(0, ((Q - df_q) / Q) * 100) if Q > 0 else 0
    return {
        "pooled_effect": pooled,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z": z,
        "p_value": p,
        "Q": Q,
        "I2_percent": i2,
        "tau2": tau2,
        "n_datasets": len(effects),
    }

all_dataset_rows = []
all_meta_rows = []
all_loo_rows = []

for endpoint in endpoints:
    sub = df[["dataset", "status", endpoint]].dropna().copy()

    dataset_rows = []
    for ds, dsub in sub.groupby("dataset"):
        a = dsub.loc[dsub["status"] == "Asymptomatic", endpoint].astype(float).dropna().values
        s = dsub.loc[dsub["status"] == "Symptomatic", endpoint].astype(float).dropna().values

        if len(a) < 1 or len(s) < 1:
            continue
        if len(a) < 2 or len(s) < 2:
            # Keep for reporting but meta-analysis effect size may be unstable
            pass

        g, var_g, asym_mean, sym_mean = hedges_g(a, s)
        dataset_rows.append({
            "endpoint": endpoint,
            "dataset": ds,
            "n_asymptomatic": len(a),
            "n_symptomatic": len(s),
            "asymptomatic_mean": asym_mean,
            "symptomatic_mean": sym_mean,
            "asym_minus_symptomatic": asym_mean - sym_mean,
            "hedges_g": g,
            "variance_g": var_g,
            "se_g": np.sqrt(var_g) if pd.notna(var_g) else np.nan,
            "ci_low": g - 1.96*np.sqrt(var_g) if pd.notna(var_g) else np.nan,
            "ci_high": g + 1.96*np.sqrt(var_g) if pd.notna(var_g) else np.nan,
        })

    ddf = pd.DataFrame(dataset_rows)
    ddf.to_csv(outdir / f"DC_meta_dataset_effects_{endpoint}.csv", index=False)
    all_dataset_rows.append(ddf)

    # Primary meta-analysis: use GSE224273 + GSE260657
    primary = ddf[ddf["dataset"].isin(["GSE224273", "GSE260657"])].dropna(subset=["hedges_g", "variance_g"]).copy()
    if len(primary) >= 2:
        meta_primary = random_effects_meta(primary["hedges_g"], primary["variance_g"])
        meta_primary["endpoint"] = endpoint
        meta_primary["analysis"] = "primary_meta_GSE224273_GSE260657"
        all_meta_rows.append(meta_primary)

        # Leave-one-dataset-out on primary set
        for ds in primary["dataset"].unique():
            loo = primary[primary["dataset"] != ds]
            if len(loo) >= 1:
                # with 1 dataset left, just carry that effect through
                if len(loo) == 1:
                    r = loo.iloc[0]
                    loo_row = {
                        "endpoint": endpoint,
                        "left_out_dataset": ds,
                        "remaining_dataset": r["dataset"],
                        "pooled_effect": r["hedges_g"],
                        "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"],
                    }
                else:
                    m = random_effects_meta(loo["hedges_g"], loo["variance_g"])
                    loo_row = {
                        "endpoint": endpoint,
                        "left_out_dataset": ds,
                        "remaining_dataset": ",".join(loo["dataset"].tolist()),
                        "pooled_effect": m["pooled_effect"],
                        "ci_low": m["ci_low"],
                        "ci_high": m["ci_high"],
                    }
                all_loo_rows.append(loo_row)

    # Sensitivity meta-analysis including GSE253902
    sensitivity = ddf[ddf["dataset"].isin(["GSE224273", "GSE260657", "GSE253902"])].dropna(subset=["hedges_g", "variance_g"]).copy()
    if len(sensitivity) >= 2:
        meta_sens = random_effects_meta(sensitivity["hedges_g"], sensitivity["variance_g"])
        meta_sens["endpoint"] = endpoint
        meta_sens["analysis"] = "sensitivity_meta_plus_GSE253902"
        all_meta_rows.append(meta_sens)

dataset_effects = pd.concat(all_dataset_rows, ignore_index=True)
meta_summary = pd.DataFrame(all_meta_rows)
loo_summary = pd.DataFrame(all_loo_rows)

dataset_effects.to_csv(outdir / "DC_meta_dataset_effects_all.csv", index=False)
meta_summary.to_csv(outdir / "DC_meta_summary.csv", index=False)
loo_summary.to_csv(outdir / "DC_meta_leave_one_dataset_out.csv", index=False)

print("Dataset effects:")
print(dataset_effects.to_string(index=False))
print("\nMeta-analysis summary:")
print(meta_summary.to_string(index=False))
print("\nLeave-one-dataset-out:")
print(loo_summary.to_string(index=False))
print("\nSaved:")
print(outdir / "DC_meta_dataset_effects_all.csv")
print(outdir / "DC_meta_summary.csv")
print(outdir / "DC_meta_leave_one_dataset_out.csv")
