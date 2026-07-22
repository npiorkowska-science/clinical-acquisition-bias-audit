"""
Stage 06 - Bias-controlled scenarios (C1-C5)
--------------------------------------------------
Task: re-evaluate the harmonized-values model under progressively
stricter acquisition-bias controls:
  C1_strict          - tight coverage-balance thresholds
  C2_age_excluded     - main thresholds, Age removed from the feature set
  C3_relaxed          - loose coverage-balance thresholds
  C4_repeated_age_matched - repeated 1:1 age-matched case/control pairs
                             (caliper matching), evaluated by CV
  C5_common_age_support   - restrict to the age range where both groups
                             overlap, then evaluate by CV
This isolates whether apparent group separability survives once
age-driven and coverage-driven acquisition artifacts are controlled for.

Result:
  outputs_*/06_bias_controlled/repeated_age_matching_metrics.csv
  outputs_*/06_bias_controlled/common_age_support.json
  outputs_*/06_bias_controlled/bias_controlled_metrics.csv
  outputs_*/06_bias_controlled/bias_controlled_predictions.csv
  outputs_*/06_bias_controlled/error_log.csv

Self-contained: no imports from other project scripts. Requires stage 02
(harmonized_matrix.parquet) to have already run.
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


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


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


def common_age_support(age, y):
    a = pd.to_numeric(age, errors="coerce")
    lo = max(a[y == 1].min(), a[y == 0].min())
    hi = min(a[y == 1].max(), a[y == 0].max())
    return a.between(lo, hi), float(lo), float(hi)


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


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "06_bias_controlled"
OUT.mkdir(parents=True, exist_ok=True)
SEED = CFG["random_seed"]

df = read_input(CFG, ROOT)
target = infer_target(df, CFG)
y = encode_target(df[target], CFG["positive_labels"])
Xh = pd.read_parquet(OUTROOT / "02_harmonization" / "harmonized_matrix.parquet")
age = Xh["Age"] if "Age" in Xh else pd.Series(np.nan, index=y.index)

rows, preds, errors = [], [], []
scenarios = {
    "C1_strict": CFG["fair_thresholds"]["C1_strict"],
    "C2_age_excluded": CFG["fair_thresholds"]["main"],
    "C3_relaxed": CFG["fair_thresholds"]["C3_relaxed"],
}
for name, t in scenarios.items():
    for model in CFG["models"]:
        exc = ["Age"] if name == "C2_age_excluded" and "Age" in Xh else []
        sel = CoverageBalanceSelector(t["min_coverage"], t["max_delta"], exc)
        try:
            m, p, _ = repeated_oof(make_pipeline(model, "values_only", SEED, sel), Xh, y,
                                    CFG["cv"]["n_splits"], CFG["cv"]["n_repeats"], SEED)
            m["scenario"] = name; m["model"] = model; m["status"] = "PASS"
            p["scenario"] = name; p["model"] = model
            rows.append(m); preds.append(p)
        except Exception as e:
            errors.append({"scenario": name, "model": model, "stage": "primary", "error": repr(e)})
            rows.append(pd.DataFrame([{"scenario": name, "model": model, "status": "NON_ESTIMABLE", "error": repr(e)}]))

# repeated age matching, evaluated by repeated stratified CV
age_attempts = []
for rep in range(CFG["age_matching"]["n_iterations"]):
    idx = age_match_indices(age, y, SEED + rep, CFG["age_matching"]["caliper"])
    for model in CFG["models"]:
        rec = {"scenario": "C4_repeated_age_matched", "model": model, "iteration": rep, "n_matched": len(idx) // 2}
        if len(idx) < 20 or len(np.unique(y.iloc[idx])) < 2:
            rec.update(status="NON_ESTIMABLE", error="insufficient matched pairs")
            age_attempts.append(rec)
            continue
        sel = CoverageBalanceSelector(**CFG["fair_thresholds"]["main"], exclude=["Age"] if "Age" in Xh else [])
        try:
            r = evaluate_repeated_cv(make_pipeline(model, "values_only", SEED + rep, sel), Xh.iloc[idx], y.iloc[idx],
                                      n_splits=min(5, len(idx) // 2), n_repeats=2, seed=SEED + rep)
            r.update(rec, status="PASS", error="")
            age_attempts.append(r)
        except Exception as e:
            rec.update(status="NON_ESTIMABLE", error=repr(e))
            age_attempts.append(rec)
            errors.append({"scenario": "C4_repeated_age_matched", "model": model, "iteration": rep, "error": repr(e)})

age_df = pd.DataFrame(age_attempts)
age_df.to_csv(OUT / "repeated_age_matching_metrics.csv", index=False)

# common age support
mask, lo, hi = common_age_support(age, y)
save_json({"lower": lo, "upper": hi, "n": int(mask.sum()), "n_pcos": int(y.loc[mask].sum()),
           "n_control": int((1 - y.loc[mask]).sum())}, OUT / "common_age_support.json")

for model in CFG["models"]:
    sel = CoverageBalanceSelector(**CFG["fair_thresholds"]["main"], exclude=["Age"] if "Age" in Xh else [])
    try:
        m, p, _ = repeated_oof(make_pipeline(model, "values_only", SEED, sel),
                                Xh.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True),
                                CFG["cv"]["n_splits"], CFG["cv"]["n_repeats"], SEED)
        m["scenario"] = "C5_common_age_support"; m["model"] = model; m["status"] = "PASS"
        p["scenario"] = "C5_common_age_support"; p["model"] = model
        rows.append(m); preds.append(p)
    except Exception as e:
        errors.append({"scenario": "C5_common_age_support", "model": model, "stage": "common_support", "error": repr(e)})
        rows.append(pd.DataFrame([{"scenario": "C5_common_age_support", "model": model, "status": "NON_ESTIMABLE", "error": repr(e)}]))

pd.concat(rows, ignore_index=True).to_csv(OUT / "bias_controlled_metrics.csv", index=False)
if preds:
    pd.concat(preds, ignore_index=True).to_csv(OUT / "bias_controlled_predictions.csv", index=False)
pd.DataFrame(errors).to_csv(OUT / "error_log.csv", index=False)

expected_primary = len(scenarios) * len(CFG["models"]) + len(CFG["models"])
if pd.concat(rows, ignore_index=True)[["scenario", "model"]].drop_duplicates().shape[0] < expected_primary:
    raise RuntimeError("Bias-controlled attempt accounting incomplete")
