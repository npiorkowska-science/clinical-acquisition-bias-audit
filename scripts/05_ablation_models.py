"""
Stage 05 - Ablation models (M1-M5, H1-H3)
---------------------------------------------
Task: systematically vary what a classifier is allowed to see, in both
the raw and harmonized feature spaces:
  M1/H1 mask_only               - only the missingness indicator
  M2/H2 values_only              - only imputed clinical values
  M3/H3 values_plus_missingness  - imputed values + missingness indicator
  M4 balanced                    - harmonized values, features filtered to
                                    comparable coverage across groups
  M5 balanced_no_age             - same as M4, with Age excluded
Each spec x model combination is evaluated with repeated stratified
cross-validation. This isolates how much of the apparent predictive
signal comes from acquisition/workflow artifacts (missingness) versus
genuine harmonized clinical values.

Result:
  outputs_*/05_ablation/ablation_metrics_by_fold.csv
  outputs_*/05_ablation/ablation_predictions.csv

Self-contained: no imports from other project scripts. Requires stage 02
(harmonized_matrix.parquet) to have already run.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                              confusion_matrix, brier_score_loss, log_loss, matthews_corrcoef)
from sklearn.inspection import permutation_importance

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


def repeated_oof(pipe, X, y, n_splits=5, n_repeats=20, seed=42, return_importance=False):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    metrics, preds, imps, selection, errors = [], [], [], [], []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        est = clone(pipe)
        try:
            est.fit(X.iloc[tr], y.iloc[tr])
            p = est.predict_proba(X.iloc[te])[:, 1]
            m = metric_row(y.iloc[te], p)
            m.update(fold=k, n_test=len(te), status="PASS")
            metrics.append(m)
            preds.append(pd.DataFrame({"row_index": X.index[te], "fold": k, "y_true": y.iloc[te].to_numpy(), "p": p}))
            if "coverage_selector" in est.named_steps:
                chosen = list(est.named_steps["coverage_selector"].selected_columns_)
                selection.extend({"fold": k, "feature": c, "selected": 1} for c in chosen)
            if return_importance and k < n_splits:
                try:
                    r = permutation_importance(est, X.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=2, random_state=seed + k, n_jobs=1)
                    names = list(X.columns)
                    if len(r.importances_mean) == len(names):
                        imps.extend({"fold": k, "feature": name, "importance_mean": float(a), "importance_sd": float(b)}
                                    for name, a, b in zip(names, r.importances_mean, r.importances_std))
                except Exception as e:
                    errors.append({"fold": k, "stage": "permutation_importance", "error": repr(e)})
        except Exception as e:
            errors.append({"fold": k, "stage": "fit_predict", "error": repr(e)})
            raise RuntimeError(f"Fold {k} failed: {e}") from e
    if not preds:
        raise RuntimeError("No OOF predictions generated")
    imp = pd.DataFrame(imps)
    if len(imp):
        sel = pd.DataFrame(selection)
        if len(sel):
            freq = sel.groupby("feature").fold.nunique() / len(metrics)
            imp["selection_frequency"] = imp.feature.map(freq).fillna(0.0)
        else:
            imp["selection_frequency"] = 1.0
    imp.attrs["errors"] = errors
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True), imp


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

MAIN = CFG["fair_thresholds"]["main"]
AGE = "Age" if "Age" in Xh.columns else None

specs = []
for model in CFG["models"]:
    specs += [
        {"name": f"M1_raw_mask_{model}", "space": "raw", "mode": "mask_only", "model": model},
        {"name": f"M2_raw_values_{model}", "space": "raw", "mode": "values_only", "model": model},
        {"name": f"M3_raw_values_mask_{model}", "space": "raw", "mode": "values_plus_missingness", "model": model},
        {"name": f"H1_harmonized_mask_{model}", "space": "harmonized", "mode": "mask_only", "model": model},
        {"name": f"H2_harmonized_values_{model}", "space": "harmonized", "mode": "values_only", "model": model},
        {"name": f"H3_harmonized_values_mask_{model}", "space": "harmonized", "mode": "values_plus_missingness", "model": model},
        {"name": f"M4_balanced_{model}", "space": "harmonized", "mode": "values_only", "model": model,
         "selector": {"min_coverage": MAIN["min_coverage"], "max_delta": MAIN["max_delta"], "exclude": []}},
        {"name": f"M5_balanced_no_age_{model}", "space": "harmonized", "mode": "values_only", "model": model,
         "selector": {"min_coverage": MAIN["min_coverage"], "max_delta": MAIN["max_delta"], "exclude": [AGE] if AGE else []}},
    ]

allm, allp = [], []
for s in specs:
    X = Xh if s["space"] == "harmonized" else Xr
    Xin = X.notna().astype(int) if s["mode"] == "mask_only" else X
    sel = CoverageBalanceSelector(**s["selector"]) if s.get("selector") else None
    try:
        m, p, _ = repeated_oof(make_pipeline(s["model"], s["mode"], SEED, sel), Xin, y,
                                CFG["cv"]["n_splits"], CFG["cv"]["n_repeats"], SEED)
        m["spec"] = s["name"]
        p["spec"] = s["name"]
        allm.append(m)
        allp.append(p)
    except Exception as e:
        print("SKIP", s["name"], e)

pd.concat(allm).to_csv(OUTROOT / "05_ablation" / "ablation_metrics_by_fold.csv", index=False)
pd.concat(allp).to_csv(OUTROOT / "05_ablation" / "ablation_predictions.csv", index=False)
