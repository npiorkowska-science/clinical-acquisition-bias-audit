"""
Stage 02 - Harmonization and data contract
--------------------------------------------
Task: apply the clinically approved raw-to-canonical feature mapping
(config/feature_dictionary.csv) to build the harmonized concept matrix,
resolve multi-alias conflicts according to each mapping row's declared
conflict policy, and validate the mapping dictionary itself (required
columns, approval status, duplicate/ non-numeric entries, unresolved
conflicts).

Result:
  outputs_*/02_harmonization/harmonized_matrix.parquet
  outputs_*/02_harmonization/harmonization_summary.csv
  outputs_*/02_harmonization/alias_conflict_report.csv
  outputs_*/02_harmonization/harmonization_validation.json

Self-contained: no imports from other project scripts.
"""
from pathlib import Path
import json

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


def harmonize(df, mapping_path, abs_tol=1e-6, rel_tol=.05, fail_on_unapproved=False):
    mp = pd.read_csv(mapping_path)
    if "priority" not in mp.columns:
        mp["priority"] = np.arange(len(mp))
    if "conflict_action" not in mp.columns:
        mp["conflict_action"] = "FAIL"
    if "conflict_reason" not in mp.columns:
        mp["conflict_reason"] = ""
    approved = mp[mp["clinical_approval"].astype(str).str.upper().isin(["YES", "APPROVED", "TRUE", "1"])].copy()
    if fail_on_unapproved and len(approved) != len(mp):
        raise ValueError("Unapproved mapping rows exist")

    out = pd.DataFrame(index=df.index)
    summary = []
    conflicts = []
    for canon, g in approved.sort_values("priority").groupby("canonical_feature", sort=False):
        vals = []
        meta = []
        for _, r in g.sort_values("priority").iterrows():
            raw = r["raw_column"]
            if raw not in df:
                continue
            v = pd.to_numeric(df[raw], errors="coerce") * float(r.get("multiplier", 1) or 1) + float(r.get("offset", 0) or 0)
            vals.append(v.rename(raw))
            meta.append(r)
        if not vals:
            continue
        block = pd.concat(vals, axis=1)
        chosen = block.bfill(axis=1).iloc[:, 0]
        out[canon] = chosen
        multi = block.notna().sum(axis=1) > 1
        action = str(g.iloc[0].get("conflict_action", "FAIL")).upper()
        reason = str(g.iloc[0].get("conflict_reason", ""))
        for idx, row in block.loc[multi].iterrows():
            arr = row.dropna().astype(float)
            ref = arr.iloc[0]
            max_abs = float((arr - ref).abs().max())
            denom = max(abs(ref), abs(arr).max(), 1e-12)
            max_rel = max_abs / denom
            within = bool(max_abs <= abs_tol or max_rel <= rel_tol)
            resolved = within or action in ["USE_PRIORITY", "MEAN", "MEDIAN"]
            if not within and action == "MEAN":
                out.loc[idx, canon] = float(arr.mean())
            elif not within and action == "MEDIAN":
                out.loc[idx, canon] = float(arr.median())
            conflicts.append({
                "row_index": idx, "canonical_feature": canon,
                "raw_columns": " | ".join(arr.index), "values": " | ".join(map(str, arr.values)),
                "max_abs_difference": max_abs, "max_relative_difference": max_rel, "within_tolerance": within,
                "conflict_action": action, "conflict_reason": reason, "resolved": resolved,
            })
        ccanon = [x for x in conflicts if x["canonical_feature"] == canon]
        summary.append({
            "canonical_feature": canon, "raw_columns_used": " | ".join(block.columns),
            "priority_order": " | ".join(block.columns),
            "n_multi_alias_rows": int(multi.sum()),
            "n_conflicts_outside_tolerance": int(sum(not x["within_tolerance"] for x in ccanon)),
            "n_unresolved_conflicts": int(sum(not x["resolved"] for x in ccanon)),
            "coverage": float(chosen.notna().mean()),
            "conflict_action": action,
        })
    return out, pd.DataFrame(summary), pd.DataFrame(conflicts)


def validate_harmonization(mapping_path, conflicts, require_approval=True, fail_on_conflicts=True):
    mp = pd.read_csv(mapping_path)
    required = ["raw_column", "canonical_feature", "unit_raw", "unit_canonical", "multiplier",
                "offset", "source_schema", "clinical_approval", "priority", "conflict_action", "conflict_reason"]
    missing = [c for c in required if c not in mp.columns]
    if missing:
        raise ValueError(f"Mapping dictionary missing required columns: {missing}")
    approved = mp["clinical_approval"].astype(str).str.upper().isin(["YES", "APPROVED", "TRUE", "1"])
    issues = []
    if require_approval and not approved.all():
        issues.append(f"{int((~approved).sum())} mapping rows are not clinically approved")
    if mp[["raw_column", "canonical_feature", "unit_raw", "unit_canonical"]].isna().any().any():
        issues.append("Missing mapping identity or unit fields")
    if pd.to_numeric(mp["multiplier"], errors="coerce").isna().any() or pd.to_numeric(mp["offset"], errors="coerce").isna().any():
        issues.append("Non-numeric multiplier or offset values")
    if mp.duplicated(["raw_column"]).any():
        issues.append("Duplicated raw-column mappings")
    unresolved = 0
    if conflicts is not None and len(conflicts):
        unresolved = int((~conflicts["resolved"].astype(bool)).sum()) if "resolved" in conflicts else int((~conflicts["within_tolerance"].astype(bool)).sum())
        if fail_on_conflicts and unresolved:
            issues.append(f"{unresolved} unresolved alias conflicts")
    report = {
        "n_mapping_rows": len(mp), "n_approved": int(approved.sum()), "n_unapproved": int((~approved).sum()),
        "n_unresolved_conflicts": unresolved, "status": "PASS" if not issues else "FAIL", "issues": " | ".join(issues),
    }
    if issues:
        raise ValueError("Harmonization validation failed: " + "; ".join(issues))
    return report


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "02_harmonization"
OUT.mkdir(parents=True, exist_ok=True)

df = read_input(CFG, ROOT)

Xh, summ, conf = harmonize(
    df, ROOT / "config" / "feature_dictionary.csv",
    abs_tol=CFG["harmonization"]["absolute_tolerance"],
    rel_tol=CFG["harmonization"]["relative_tolerance"],
    fail_on_unapproved=CFG["harmonization"]["require_clinical_approval"],
)
report = validate_harmonization(
    ROOT / "config" / "feature_dictionary.csv", conf,
    CFG["harmonization"]["require_clinical_approval"], True,
)

Xh.to_parquet(OUT / "harmonized_matrix.parquet")
summ.to_csv(OUT / "harmonization_summary.csv", index=False)
conf.to_csv(OUT / "alias_conflict_report.csv", index=False)
save_json(report, OUT / "harmonization_validation.json")

print(report)
