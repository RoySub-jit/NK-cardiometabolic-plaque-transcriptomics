from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

infile = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation/GSE224273_dc_apc_restricted_sample_summary.csv")
outdir = Path("/pub/subhajr/projects/heat_aging_human_reanalysis/results/tier2_validation")
outdir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

df = pd.read_csv(infile)

endpoints = [
    "prop_cdc2_high",
    "cdc2_score_mean",
]

def exact_permutation_p(values, labels):
    # observed = asymptomatic mean - symptomatic mean
    values = np.array(values, dtype=float)
    labels = np.array(labels)
    idx = np.arange(len(values))

    asym_idx = np.where(labels == "Asymptomatic")[0]
    sym_idx = np.where(labels == "Symptomatic")[0]
    obs = values[asym_idx].mean() - values[sym_idx].mean()

    n_asym = len(asym_idx)
    perm_diffs = []
    for comb in combinations(idx, n_asym):
        comb = np.array(comb)
        rest = np.setdiff1d(idx, comb)
        diff = values[comb].mean() - values[rest].mean()
        perm_diffs.append(diff)

    perm_diffs = np.array(perm_diffs)
    p = np.mean(np.abs(perm_diffs) >= abs(obs))
    return obs, p, perm_diffs

def bootstrap_ci(asym, sym, n_boot=5000):
    asym = np.array(asym, dtype=float)
    sym = np.array(sym, dtype=float)
    diffs = []
    for _ in range(n_boot):
        a = rng.choice(asym, size=len(asym), replace=True)
        s = rng.choice(sym, size=len(sym), replace=True)
        diffs.append(a.mean() - s.mean())
    diffs = np.array(diffs)
    return diffs.mean(), np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)

def analyze(input_df, analysis_name):
    rows = []
    for endpoint in endpoints:
        sub = input_df[["sample", "status", endpoint]].dropna().copy()
        values = sub[endpoint].astype(float).values
        labels = sub["status"].values

        asym = sub.loc[sub["status"] == "Asymptomatic", endpoint].astype(float).values
        sym = sub.loc[sub["status"] == "Symptomatic", endpoint].astype(float).values

        obs_diff, p_exact, _ = exact_permutation_p(values, labels)
        boot_mean, ci_low, ci_high = bootstrap_ci(asym, sym, n_boot=5000)

        rows.append({
            "analysis": analysis_name,
            "endpoint": endpoint,
            "n_asymptomatic": len(asym),
            "n_symptomatic": len(sym),
            "asymptomatic_mean": asym.mean(),
            "symptomatic_mean": sym.mean(),
            "asym_minus_symptomatic_mean": obs_diff,
            "exact_permutation_p": p_exact,
            "bootstrap_mean_diff": boot_mean,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        })
    return pd.DataFrame(rows)

# Full analysis
full_res = analyze(df, "full_samples")

# Sensitivity: collapse Sample1 and Sample1G into one symptomatic average
sens_df = df.copy()
sym_pair = sens_df[sens_df["sample"].isin(["Sample1", "Sample1G"])].copy()
other = sens_df[~sens_df["sample"].isin(["Sample1", "Sample1G"])].copy()

if len(sym_pair) == 2:
    collapsed = {
        "sample": "Sample1_collapsed",
        "gsm": "GSM7018579+GSM7018585",
        "status": "Symptomatic",
    }
    for endpoint in endpoints:
        collapsed[endpoint] = sym_pair[endpoint].mean()
    for extra in ["prop_apc", "n_cells", "n_apc_cells", "myeloid_score_mean", "cdc2_minus_inflammatory_mean", "prop_myeloid_high", "prop_pdc_high", "pdc_minus_cdc2_mean", "pdc_score_mean"]:
        if extra in sens_df.columns:
            collapsed[extra] = sym_pair[extra].mean()

    sens_df = pd.concat([other, pd.DataFrame([collapsed])], ignore_index=True)

collapsed_res = analyze(sens_df, "collapsed_Sample1_Sample1G")

res = pd.concat([full_res, collapsed_res], ignore_index=True)
outfile = outdir / "GSE224273_dc_exact_permutation_bootstrap_results.csv"
res.to_csv(outfile, index=False)

print(res.to_string(index=False))
print("\nSaved:", outfile)
