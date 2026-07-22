"""
Stage 09 - Calibration and reliability
-----------------------------------------
Task: for every ablation spec's out-of-fold predictions (stage 05), fit a
logistic recalibration (logit(y) ~ logit(p)) to estimate the calibration
intercept and slope, compute the Brier score, Brier skill score and
expected calibration error (ECE), and bootstrap all of these together
with a reliability-curve confidence band (10 probability bins).

Result:
  outputs_*/09_calibration/calibration_summary.csv
  outputs_*/09_calibration/calibration_bootstrap_distributions.csv
  outputs_*/09_calibration/reliability_curve_bootstrap_bands.csv

Self-contained: no imports from other project scripts. Requires stage 05
(ablation_predictions.csv) to have already run.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path.cwd()
if ROOT.name == "scripts":
    ROOT = ROOT.parent


# ---------------------------------------------------------------------------
# Helpers (inlined, no external project modules)
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_predictions(preds):
    return preds.groupby(["row_index", "y_true"], as_index=False).p.mean()


def calibration_metrics(y, p, n_bins=10):
    import statsmodels.api as sm
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    try:
        fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
        intercept, slope = map(float, fit.params)
    except Exception:
        intercept = slope = np.nan
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(qs)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi if hi == bins[-1] else p < hi)
        if m.any():
            ece += float(m.mean() * abs(y[m].mean() - p[m].mean()))
    b = brier_score_loss(y, p)
    base = brier_score_loss(y, np.repeat(y.mean(), len(y)))
    return {
        "brier": b, "brier_skill": 1 - b / base if base > 0 else np.nan, "ece": ece,
        "calibration_intercept": intercept, "calibration_slope": slope,
        "calibration_model": "logistic recalibration: logit(Y)~logit(p)",
    }


def bootstrap_calibration(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    y = np.asarray(y)
    p = np.asarray(p)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        r = calibration_metrics(y[idx], p[idx])
        r["iteration"] = i
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "09_calibration"
OUT.mkdir(parents=True, exist_ok=True)
SEED = CFG["random_seed"]

preds = pd.read_csv(OUTROOT / "05_ablation" / "ablation_predictions.csv")
rows, boots, curves = [], [], []

for spec, g in preds.groupby("spec"):
    a = aggregate_predictions(g)
    cm = calibration_metrics(a.y_true, a.p, 10)
    cm["spec"] = spec
    rows.append(cm)

    b = bootstrap_calibration(a.y_true, a.p, CFG["calibration"]["bootstrap_iterations"], SEED)
    b["spec"] = spec
    boots.append(b)

    grid = np.linspace(0, 1, 11)
    rng = np.random.default_rng(SEED)
    vals = []
    for i in range(CFG["calibration"]["bootstrap_iterations"]):
        idx = rng.integers(0, len(a), len(a))
        yy = a.y_true.to_numpy()[idx]
        pp = a.p.to_numpy()[idx]
        for lo, hi in zip(grid[:-1], grid[1:]):
            m = (pp >= lo) & (pp < (hi if hi < 1 else hi + 1e-9))
            if m.any():
                vals.append({"iteration": i, "bin_mid": (lo + hi) / 2, "observed": yy[m].mean(), "predicted": pp[m].mean()})
    v = pd.DataFrame(vals)
    s = v.groupby("bin_mid").agg(
        observed_mean=("observed", "mean"),
        ci_low=("observed", lambda x: x.quantile(.025)),
        ci_high=("observed", lambda x: x.quantile(.975)),
        predicted_mean=("predicted", "mean"),
    ).reset_index()
    s["spec"] = spec
    curves.append(s)

pd.DataFrame(rows).to_csv(OUT / "calibration_summary.csv", index=False)
pd.concat(boots).to_csv(OUT / "calibration_bootstrap_distributions.csv", index=False)
pd.concat(curves).to_csv(OUT / "reliability_curve_bootstrap_bands.csv", index=False)
