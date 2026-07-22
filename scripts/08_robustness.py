"""
Stage 08 - Robustness: full-pipeline bootstrap, paired contrasts, permutation test
--------------------------------------------------------------------------------------
Task:
  1. Full-pipeline stratified bootstrap: refit every ablation spec (M1-M5,
     H1-H3) on many stratified bootstrap resamples with a common resample
     per iteration (enabling paired comparisons), evaluate on the
     out-of-bag participants.
  2. Paired bootstrap contrasts: for each model, compute the paired
     difference in ROC-AUC / balanced accuracy / Brier between key spec
     pairs (e.g. values+mask vs values-only, raw vs harmonized mask,
     balanced vs balanced-no-age), with completeness accounting.
  3. Label-permutation test: for four prespecified key configurations,
     shuffle the outcome label, refit by stratified CV, and compare the
     observed ROC-AUC against the empirical null distribution.

Result:
  outputs_*/08_robustness/full_pipeline_bootstrap_all_models.csv
  outputs_*/08_robustness/paired_bootstrap_contrasts.csv
  outputs_*/08_robustness/label_permutation_key_models.csv
  outputs_*/08_robustness/permutation_test_summary.csv

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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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


def aggregate_predictions(preds):
    return preds.groupby(["row_index", "y_true"], as_index=False).p.mean()


def full_pipeline_bootstrap(specs, Xraw, Xharm, y, n_iter=500, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    ca = np.where(y.to_numpy() == 1)[0]
    co = np.where(y.to_numpy() == 0)[0]
    for i in range(n_iter):
        boot = np.r_[rng.choice(ca, len(ca), True), rng.choice(co, len(co), True)]
        oob = np.setdiff1d(np.arange(len(y)), np.unique(boot))
        if len(oob) < 20 or len(np.unique(y.iloc[oob])) < 2:
            for s in specs:
                rows.append({"iteration": i, "spec": s["name"], "status": "NON_ESTIMABLE", "error": "insufficient_oob"})
            continue
        for s in specs:
            X = Xharm if s["space"] == "harmonized" else Xraw
            Xin = X.notna().astype(int) if s["mode"] == "mask_only" else X
            sel = CoverageBalanceSelector(**s["selector"]) if s.get("selector") else None
            pipe = make_pipeline(s["model"], s["mode"], seed + i, sel)
            try:
                pipe.fit(Xin.iloc[boot], y.iloc[boot])
                p = pipe.predict_proba(Xin.iloc[oob])[:, 1]
                r = metric_row(y.iloc[oob], p)
                r.update(iteration=i, spec=s["name"], status="PASS", error="")
                rows.append(r)
            except Exception as e:
                rows.append({"iteration": i, "spec": s["name"], "status": "NON_ESTIMABLE", "error": repr(e)})
    out = pd.DataFrame(rows)
    expected = n_iter * len(specs)
    if len(out) != expected:
        raise RuntimeError(f"Bootstrap attempt accounting incomplete: {len(out)}/{expected}")
    return out


def paired_contrasts(boot, contrasts, metrics=("roc_auc", "balanced_accuracy", "brier"), min_success_fraction=.90, n_requested=None):
    rows = []
    for a, b in contrasts:
        aa = boot[(boot.spec == a) & (boot.status == "PASS")].set_index("iteration")
        bb = boot[(boot.spec == b) & (boot.status == "PASS")].set_index("iteration")
        req = n_requested if n_requested is not None else int(boot.iteration.nunique())
        for m in metrics:
            valid = aa.index.intersection(bb.index)
            if m in aa.columns and m in bb.columns:
                valid = valid.intersection(aa.index[aa[m].notna()]).intersection(bb.index[bb[m].notna()])
            else:
                valid = []
            d = aa.loc[valid, m] - bb.loc[valid, m] if len(valid) else pd.Series(dtype=float)
            frac = len(d) / req if req else 0.0
            if len(d) == 0:
                status = "NOT_ESTIMABLE"
            else:
                status = "PASS" if frac >= min_success_fraction else "LOW_COMPLETENESS"
            rows.append({
                "contrast": f"{a} - {b}", "metric": m, "mean_difference": d.mean() if len(d) else np.nan,
                "ci_low": d.quantile(.025) if len(d) else np.nan, "ci_high": d.quantile(.975) if len(d) else np.nan,
                "n_requested": req, "n_successful_A": int(aa.index.nunique()), "n_successful_B": int(bb.index.nunique()),
                "n_paired": len(d), "success_fraction": frac, "status": status,
            })
    return pd.DataFrame(rows)


def permutation_models(specs, Xraw, Xharm, y, n_iter=200, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_iter):
        yp = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + i)
        for s in specs:
            X = Xharm if s["space"] == "harmonized" else Xraw
            Xin = X.notna().astype(int) if s["mode"] == "mask_only" else X
            ps = np.full(len(y), np.nan)
            try:
                for tr, te in cv.split(Xin, yp):
                    sel = CoverageBalanceSelector(**s["selector"]) if s.get("selector") else None
                    pipe = make_pipeline(s["model"], s["mode"], seed + i, sel)
                    pipe.fit(Xin.iloc[tr], yp.iloc[tr])
                    ps[te] = pipe.predict_proba(Xin.iloc[te])[:, 1]
                rows.append({"iteration": i, "spec": s["name"], "roc_auc": roc_auc_score(yp, ps), "status": "PASS", "error": ""})
            except Exception as e:
                rows.append({"iteration": i, "spec": s["name"], "roc_auc": np.nan, "status": "NON_ESTIMABLE", "error": repr(e)})
    out = pd.DataFrame(rows)
    if len(out) != n_iter * len(specs):
        raise RuntimeError("Permutation attempt accounting incomplete")
    return out


def empirical_permutation_summary(observed, null_df):
    rows = []
    for spec, obs in observed.items():
        vals = null_df.loc[null_df["spec"] == spec, "roc_auc"].dropna().to_numpy()
        if len(vals) == 0:
            rows.append({"spec": spec, "observed_roc_auc": obs, "n_permutations": 0, "empirical_p": np.nan, "status": "FAIL"})
            continue
        p = (1 + np.sum(vals >= obs)) / (1 + len(vals))
        rows.append({
            "spec": spec, "observed_roc_auc": obs, "null_mean": float(vals.mean()),
            "null_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "null_q95": float(np.quantile(vals, .95)), "n_permutations": len(vals),
            "empirical_p": float(p), "status": "PASS",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "08_robustness"
OUT.mkdir(parents=True, exist_ok=True)
SEED = CFG["random_seed"]

df = read_input(CFG, ROOT)
target = infer_target(df, CFG)
y = encode_target(df[target], CFG["positive_labels"])
admins = admin_columns(df, CFG, target)
Xr = numeric_matrix(df, [target] + admins)
Xh = pd.read_parquet(OUTROOT / "02_harmonization" / "harmonized_matrix.parquet")

MAIN = CFG["fair_thresholds"]["main"]
AGE = "Age" if "Age" in Xh else None

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

nb = CFG["bootstrap"]["n_iterations"]
boot = full_pipeline_bootstrap(specs, Xr, Xh, y, nb, SEED)
boot.to_csv(OUT / "full_pipeline_bootstrap_all_models.csv", index=False)

contr = []
for model in CFG["models"]:
    contr += [
        (f"M3_raw_values_mask_{model}", f"M2_raw_values_{model}"),
        (f"M3_raw_values_mask_{model}", f"M1_raw_mask_{model}"),
        (f"M1_raw_mask_{model}", f"H1_harmonized_mask_{model}"),
        (f"H3_harmonized_values_mask_{model}", f"H2_harmonized_values_{model}"),
        (f"M4_balanced_{model}", f"M5_balanced_no_age_{model}"),
        (f"M3_raw_values_mask_{model}", f"M5_balanced_no_age_{model}"),
    ]
pc = paired_contrasts(boot, contr, n_requested=nb)
pc.to_csv(OUT / "paired_bootstrap_contrasts.csv", index=False)
if (pc.status == "LOW_COMPLETENESS").any():
    raise RuntimeError("Paired bootstrap completeness below threshold")

key = [s for s in specs if s["name"] in ["M1_raw_mask_logistic", "H1_harmonized_mask_logistic",
                                          "M3_raw_values_mask_logistic", "M5_balanced_no_age_logistic"]]
null = permutation_models(key, Xr, Xh, y, CFG["permutation"]["n_iterations"], CFG["cv"]["n_splits"], SEED)
null.to_csv(OUT / "label_permutation_key_models.csv", index=False)

obs = {}
for s in key:
    X = Xh if s["space"] == "harmonized" else Xr
    Xin = X.notna().astype(int) if s["mode"] == "mask_only" else X
    sel = CoverageBalanceSelector(**s["selector"]) if s.get("selector") else None
    m, p, _ = repeated_oof(make_pipeline(s["model"], s["mode"], SEED, sel), Xin, y, CFG["cv"]["n_splits"], 1, SEED)
    obs[s["name"]] = float(aggregate_predictions(p).pipe(lambda a: roc_auc_score(a.y_true, a.p)))

summary = empirical_permutation_summary(obs, null)
summary.to_csv(OUT / "permutation_test_summary.csv", index=False)
if (summary.status != "PASS").any():
    raise RuntimeError("Permutation test incomplete")
