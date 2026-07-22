"""
Stage 10 - Semi-synthetic factorial validation
--------------------------------------------------
Task: generate synthetic cohorts with independently controlled biological
signal, workflow-driven missingness shift and schema shift (two aliased
"source" columns per lab feature that get collapsed into one during
harmonization), sweeping all three factors over a small grid with
repeats. For each condition, evaluate a logistic-regression holdout
classifier in both the raw (aliased) and collapsed (harmonized) feature
space, under mask_only / values_only / values_plus_missingness. This is
a controlled sanity check that the framework detects workflow/schema
leakage when and only when it is actually present.

Result:
  outputs_*/10_simulation/semi_synthetic_factorial_validation.csv

Self-contained: no imports from other project scripts. Does not depend
on any previous stage or on the source workbook - synthetic data only.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                              confusion_matrix, brier_score_loss, log_loss, matthews_corrcoef)

ROOT = Path.cwd()
if ROOT.name == "scripts":
    ROOT = ROOT.parent


# ---------------------------------------------------------------------------
# Helpers (inlined, no external project modules)
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CoverageBalanceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, min_coverage=.7, max_delta=.1, exclude=None):
        self.min_coverage = min_coverage
        self.max_delta = max_delta
        self.exclude = exclude

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)
        self.columns_in_ = list(X.columns)
        cp = X.loc[y == 1].notna().mean()
        cc = X.loc[y == 0].notna().mean()
        keep = (cp >= self.min_coverage) & (cc >= self.min_coverage) & ((cp - cc).abs() <= self.max_delta)
        exc = set(self.exclude or [])
        self.selected_columns_ = [c for c in X if keep[c] and c not in exc]
        if not self.selected_columns_:
            raise ValueError("No features satisfy coverage criteria")
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=getattr(X, "columns", self.columns_in_))[self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_columns_, object)


def make_model(name, seed=42):
    import os
    if name == "logistic":
        return LogisticRegression(max_iter=150, class_weight="balanced", solver="liblinear", tol=1e-3, random_state=seed)
    if name == "random_forest":
        n = 20 if os.environ.get("PCOS_SMOKE") == "1" else 30
        return RandomForestClassifier(n_estimators=n, max_depth=6, min_samples_leaf=2,
                                       class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    raise ValueError(name)


def make_pipeline(model_name, mode, seed=42, selector=None):
    steps = []
    if selector is not None:
        steps.append(("coverage_selector", selector))
    if mode == "mask_only":
        steps.append(("model", make_model(model_name, seed)))
    elif mode in ("values_only", "values_plus_missingness"):
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=(mode == "values_plus_missingness"), keep_empty_features=True)))
        if model_name == "logistic":
            steps.append(("scale", StandardScaler(with_mean=False)))
        steps.append(("model", make_model(model_name, seed)))
    else:
        raise ValueError(mode)
    return Pipeline(steps)


def metric_row(y, p, threshold=.5):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "roc_auc": roc_auc_score(y, p), "ap_pcos": average_precision_score(y, p),
        "ap_control": average_precision_score(1 - y, 1 - p), "baseline_ap_pcos": float(y.mean()),
        "baseline_ap_control": float((1 - y).mean()), "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": sens, "specificity": spec, "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.c_[1 - p, p], labels=[0, 1]),
    }


def evaluate_holdout(pipe, X, y, seed=42, test_size=.25):
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    est = clone(pipe)
    est.fit(X.iloc[tr], y.iloc[tr])
    p = est.predict_proba(X.iloc[te])[:, 1]
    return metric_row(y.iloc[te], p)


def synthetic_schema_shift(n=600, seed=42, bio_signal=.7, workflow_shift=.5, schema_shift=.7):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = {}
    for j in range(12):
        v = rng.normal(bio_signal * y, 1, n)
        miss = rng.random(n) < (0.1 + workflow_shift * (y if j < 4 else 1 - y) * .6)
        v[miss] = np.nan
        X[f"lab_{j}"] = v
    df = pd.DataFrame(X)
    for j in range(4):
        a = df[f"lab_{j}"].copy()
        b = df[f"lab_{j}"].copy()
        df[f"sourceA_lab_{j}"] = a.where((y == 0) | (rng.random(n) > schema_shift))
        df[f"sourceB_lab_{j}"] = b.where((y == 1) | (rng.random(n) > schema_shift))
        df.drop(columns=[f"lab_{j}"], inplace=True)
    return df, pd.Series(y, name="target")


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "10_simulation"
OUT.mkdir(parents=True, exist_ok=True)
SEED = CFG["random_seed"]

rows = []
for bio in [0, .4, .8]:
    for wf in [0, .3, .6]:
        for schema in [0, .3, .7]:
            for rep in range(CFG["simulation"]["n_repeats"]):
                X, y = synthetic_schema_shift(CFG["simulation"]["n"], SEED + rep, bio, wf, schema)
                collapsed = X.copy()
                for j in range(4):
                    a = f"sourceA_lab_{j}"
                    b = f"sourceB_lab_{j}"
                    collapsed[f"lab_{j}"] = collapsed[[a, b]].bfill(axis=1).iloc[:, 0]
                    collapsed.drop(columns=[a, b], inplace=True)
                for space, Z in [("raw", X), ("collapsed", collapsed)]:
                    for mode in ["mask_only", "values_only", "values_plus_missingness"]:
                        rec = {"bio_signal": bio, "workflow_shift": wf, "schema_shift": schema, "rep": rep, "space": space, "mode": mode}
                        try:
                            Xin = Z.notna().astype(int) if mode == "mask_only" else Z
                            r = evaluate_holdout(make_pipeline("logistic", mode, SEED + rep), Xin, y, SEED + rep, .3)
                            r.update(rec, status="PASS", error="")
                            rows.append(r)
                        except Exception as e:
                            rec.update(status="FAIL", error=repr(e))
                            rows.append(rec)

out = pd.DataFrame(rows)
out.to_csv(OUT / "semi_synthetic_factorial_validation.csv", index=False)

expected = 3 * 3 * 3 * CFG["simulation"]["n_repeats"] * 2 * 3
if len(out) != expected or (out.status == "FAIL").any():
    raise RuntimeError("Semi-synthetic validation incomplete")
