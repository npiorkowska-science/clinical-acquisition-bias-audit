"""
Stage 01 - Cohort and raw-schema audit
---------------------------------------
Task: load the source workbook, infer the target column and the
administrative (non-clinical) columns, build the raw numeric feature
matrix, and record basic cohort counts (PCOS vs control) and per-column
coverage before any harmonization.

Result:
  outputs_*/01_schema_audit/cohort_manifest.json
  outputs_*/01_schema_audit/raw_schema_inventory.csv

Self-contained: no imports from other project scripts.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

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
Xraw = numeric_matrix(df, [target] + admins)

manifest = {
    "n_rows": len(df),
    "n_pcos": int(y.sum()),
    "n_control": int((1 - y).sum()),
    "n_raw_columns": df.shape[1],
    "n_numeric_features": Xraw.shape[1],
    "target": target,
    "administrative_columns": admins,
}
save_json(manifest, OUTROOT / "01_schema_audit" / "cohort_manifest.json")

pd.DataFrame({
    "column": df.columns,
    "dtype": map(str, df.dtypes),
    "coverage": df.notna().mean().values,
}).to_csv(OUTROOT / "01_schema_audit" / "raw_schema_inventory.csv", index=False)

print(manifest)
