"""
Stage 03 - Missingness shift between groups
----------------------------------------------
Task: for both the raw feature matrix and the harmonized concept matrix,
test whether each feature's missingness (coverage) differs between the
PCOS and control groups. Uses Fisher's exact test with Benjamini-Hochberg
FDR correction and a Newcombe score confidence interval for the
coverage-difference effect size. Also exports binary presence/absence
maps (features x participants) for later heatmap figures.

Result:
  outputs_*/03_missingness/raw_coverage_fisher_fdr.csv
  outputs_*/03_missingness/harmonized_coverage_fisher_fdr.csv
  outputs_*/03_missingness/raw_missingness_map.csv
  outputs_*/03_missingness/harmonized_missingness_map.csv

Self-contained: no imports from other project scripts. Requires stage 02
(harmonized_matrix.parquet) to have already run.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests

ROOT = Path.cwd()
if ROOT.name == "scripts":
    ROOT = ROOT.parent


# ---------------------------------------------------------------------------
# Helpers (inlined, no external project modules)
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_input(cfg, root=Path(".")):
    p = root / cfg["input_file"]
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p, sheet_name=cfg.get("sheet_name", 0))
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(p.suffix)


def infer_target(df, cfg):
    for c in cfg["target_candidates"]:
        if c in df.columns:
            return c
    raise KeyError("Target not found")


def encode_target(s, positive_labels):
    pos = {str(x).strip().lower() for x in positive_labels}
    out = s.map(lambda x: 1 if str(x).strip().lower() in pos else 0).astype(int)
    if out.nunique() != 2:
        raise ValueError(out.value_counts().to_dict())
    return out


def admin_columns(df, cfg, target):
    pats = [p.lower() for p in cfg["administrative_patterns"]]
    exact = set(cfg.get("id_candidates", []))
    out = []
    for c in df.columns:
        if c == target:
            continue
        cl = str(c).lower()
        if c in exact or any(re.search(rf"(^|[_\s]){re.escape(p)}([_\s]|$)", cl) for p in pats):
            out.append(c)
    return sorted(set(out))


def numeric_matrix(df, exclude):
    out = df.drop(columns=[c for c in exclude if c in df], errors="ignore").copy()
    for c in out:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.loc[:, out.notna().any()]


def _wilson_interval(successes, n, alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def _newcombe_difference_ci(x1, n1, x0, n0, alpha=0.05):
    """Newcombe score CI for p1-p0 without continuity correction."""
    p1 = x1 / n1 if n1 else np.nan
    p0 = x0 / n0 if n0 else np.nan
    l1, u1 = _wilson_interval(x1, n1, alpha)
    l0, u0 = _wilson_interval(x0, n0, alpha)
    low = (p1 - p0) - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    high = (p1 - p0) + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return max(-1.0, low), min(1.0, high)


def coverage_audit(X, y):
    rows = []
    y = np.asarray(y)
    for c in X.columns:
        a = X.loc[y == 1, c].notna()
        b = X.loc[y == 0, c].notna()
        tab = [[int((~a).sum()), int(a.sum())], [int((~b).sum()), int(b.sum())]]
        odds, p = fisher_exact(tab)
        x1, n1 = int(a.sum()), len(a)
        x0, n0 = int(b.sum()), len(b)
        cp = x1 / n1
        cc = x0 / n0
        d = cp - cc
        lo, hi = _newcombe_difference_ci(x1, n1, x0, n0)
        rows.append({
            "feature": c, "n_pcos": n1, "n_control": n0, "observed_pcos": x1, "observed_control": x0,
            "coverage_pcos": cp, "coverage_control": cc, "delta_coverage": d,
            "delta_ci_low": lo, "delta_ci_high": hi, "ci_method": "Newcombe score",
            "abs_delta_coverage": abs(d), "missing_odds_ratio": odds, "fisher_p": p,
        })
    z = pd.DataFrame(rows)
    z["fdr_q"] = multipletests(z.fisher_p.fillna(1), method="fdr_bh")[1]
    return z.sort_values(["abs_delta_coverage", "fdr_q"], ascending=[False, True])


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUTROOT.mkdir(exist_ok=True)
SEED = CFG["random_seed"]

df = read_input(CFG, ROOT)
target = infer_target(df, CFG)
y = encode_target(df[target], CFG["positive_labels"])
admins = admin_columns(df, CFG, target)
Xr = numeric_matrix(df, [target] + admins)
Xh = pd.read_parquet(OUTROOT / "02_harmonization" / "harmonized_matrix.parquet")

coverage_audit(Xr, y).to_csv(OUTROOT / "03_missingness" / "raw_coverage_fisher_fdr.csv", index=False)
coverage_audit(Xh, y).to_csv(OUTROOT / "03_missingness" / "harmonized_coverage_fisher_fdr.csv", index=False)

# patient-feature maps, sorted by group, exported only for visualization
Xr.notna().astype(int).assign(target=y.values).sort_values("target").to_csv(
    OUTROOT / "03_missingness" / "raw_missingness_map.csv", index=False)
Xh.notna().astype(int).assign(target=y.values).sort_values("target").to_csv(
    OUTROOT / "03_missingness" / "harmonized_missingness_map.csv", index=False)
