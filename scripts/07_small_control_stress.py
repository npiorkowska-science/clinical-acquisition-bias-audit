"""
Stage 07 - Small-control-size stress test
------------------------------------------
Task: repeatedly resample small, stratified case/control subsets (control
sizes from config/analysis_config.json: stress_test.control_sizes) and
re-evaluate every ablation spec (M1-M5, H1-H3) with a lighter repeated CV
on each resample, tracking the estimable fraction (how often a spec could
be fit and evaluated at all at that sample size). Also repeats the C4
age-matching evaluation across many random-seed iterations for every
spec, using the raw age column directly from the source workbook.

Result:
  outputs_*/07_stress_test/small_control_stress_all_models.csv
  outputs_*/07_stress_test/error_log.csv
  outputs_*/07_stress_test/stress_test_estimability.csv
  outputs_*/07_stress_test/repeated_age_matching_all_models.csv
  outputs_*/07_stress_test/age_matching_error_log.csv

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


def aggregate_predictions(preds):
    return preds.groupby(["row_index", "y_true"], as_index=False).p.mean()


def evaluate_repeated_cv(pipe, X, y, n_splits=5, n_repeats=3, seed=42):
    n_splits = min(n_splits, int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        raise ValueError("Insufficient minority-class observations for stratified CV")
    m, p, _ = repeated_oof(pipe, X.reset_index(drop=True), pd.Series(y).reset_index(drop=True), n_splits, n_repeats, seed)
    agg = aggregate_predictions(p)
    out = metric_row(agg.y_true, agg.p)
    out.update({"cv_n_splits": n_splits, "cv_n_repeats": n_repeats, "n_evaluated": len(agg)})
    return out


def age_match_indices(age, y, seed=42, caliper=.5):
    rng = np.random.default_rng(seed)
    cases = np.where(np.asarray(y) == 1)[0].tolist()
    ctrls = np.where(np.asarray(y) == 0)[0].tolist()
    rng.shuffle(cases)
    used = set()
    idx = []
    for j in ctrls:
        cand = [i for i in cases if i not in used and pd.notna(age.iloc[i]) and pd.notna(age.iloc[j])
                and abs(float(age.iloc[i]) - float(age.iloc[j])) <= caliper]
        if cand:
            i = min(cand, key=lambda k: abs(float(age.iloc[k]) - float(age.iloc[j])))
            used.add(i)
            idx.extend([i, j])
    return np.asarray(idx, int)


def stratified_resample_indices(y, n_case, n_ctrl, rng):
    ca = np.where(np.asarray(y) == 1)[0]
    co = np.where(np.asarray(y) == 0)[0]
    return np.r_[rng.choice(ca, n_case, replace=n_case > len(ca)), rng.choice(co, n_ctrl, replace=n_ctrl > len(co))]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "07_stress_test"
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

rng = np.random.default_rng(SEED)
rows, errors = [], []
for i in range(CFG["stress_test"]["n_iterations"]):
    print("stress iteration", i, flush=True)
    for nc in CFG["stress_test"]["control_sizes"]:
        idx = stratified_resample_indices(y, nc, nc, rng)
        for s in specs:
            X = Xh if s["space"] == "harmonized" else Xr
            Xin = X.notna().astype(int) if s["mode"] == "mask_only" else X
            sel = CoverageBalanceSelector(**s["selector"]) if s.get("selector") else None
            try:
                r = evaluate_repeated_cv(make_pipeline(s["model"], s["mode"], SEED + i, sel), Xin.iloc[idx], y.iloc[idx],
                                          n_splits=2, n_repeats=1, seed=SEED + i)
                r.update(iteration=i, control_size=nc, spec=s["name"])
                rows.append(r)
            except Exception as e:
                errors.append({"stage": "small_control", "iteration": i, "control_size": nc, "spec": s["name"], "error": repr(e)})
                rows.append({"iteration": i, "control_size": nc, "spec": s["name"], "status": "NON_ESTIMABLE", "error": repr(e)})

res = pd.DataFrame(rows)
res.to_csv(OUT / "small_control_stress_all_models.csv", index=False)
pd.DataFrame(errors).to_csv(OUT / "error_log.csv", index=False)

expected = CFG["stress_test"]["n_iterations"] * len(CFG["stress_test"]["control_sizes"]) * len(specs)
if len(res) != expected:
    raise RuntimeError(f"Stress test attempt accounting incomplete: {len(res)}/{expected}")

summary = res.groupby("spec").agg(n_attempted=("iteration", "size"), n_estimable=("roc_auc", "count")).reset_index()
summary["estimable_fraction"] = summary.n_estimable / summary.n_attempted
summary.to_csv(OUT / "stress_test_estimability.csv", index=False)

# repeated age matching evaluated by CV, using the raw age column
age_col = next((c for c in CFG["age_candidates"] if c in df.columns), None)
am, amerr = [], []
if age_col:
    age = pd.to_numeric(df[age_col], errors="coerce")
    for i in range(CFG["age_matching"]["n_iterations"]):
        print("age iteration", i, flush=True)
        idx = age_match_indices(age, y, SEED + i, CFG["age_matching"]["caliper"])
        if len(idx) < 20 or pd.Series(y.iloc[idx]).value_counts().min() < 2:
            continue
        for s in specs:
            X = Xh if s["space"] == "harmonized" else Xr
            Xin = X.notna().astype(int) if s["mode"] == "mask_only" else X
            sel = CoverageBalanceSelector(**s["selector"]) if s.get("selector") else None
            try:
                r = evaluate_repeated_cv(make_pipeline(s["model"], s["mode"], SEED + i, sel), Xin.iloc[idx], y.iloc[idx],
                                          n_splits=2, n_repeats=1, seed=SEED + i)
                r.update(iteration=i, n_pairs=len(idx) // 2, spec=s["name"])
                am.append(r)
            except Exception as e:
                amerr.append({"stage": "age_matching", "iteration": i, "spec": s["name"], "error": repr(e)})

pd.DataFrame(am).to_csv(OUT / "repeated_age_matching_all_models.csv", index=False)
pd.DataFrame(amerr).to_csv(OUT / "age_matching_error_log.csv", index=False)
