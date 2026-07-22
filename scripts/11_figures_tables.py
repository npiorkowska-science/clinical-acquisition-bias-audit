"""
Stage 11 - Manuscript figures and tables
--------------------------------------------
Task: collect the CSV outputs of stages 03, 05-10 into a single
multi-sheet Excel workbook, then render the summary figures used to
report the results: model performance forest plot (ROC-AUC with 95%
empirical fold interval), coverage-difference bar charts (raw and
harmonized), missingness heatmaps, paired-bootstrap-contrast forest
plot, calibration reliability curves, and a simulation summary table.

Result:
  outputs_*/11_figures_tables/Manuscript_Tables.xlsx
  outputs_*/11_figures_tables/table_model_performance_summary.csv
  outputs_*/11_figures_tables/table_simulation_summary.csv
  outputs_*/11_figures_tables/fig_model_performance.png
  outputs_*/11_figures_tables/fig_raw_coverage_difference.png
  outputs_*/11_figures_tables/fig_harmonized_coverage_difference.png
  outputs_*/11_figures_tables/fig_raw_missingness_heatmap.png
  outputs_*/11_figures_tables/fig_harmonized_missingness_heatmap.png
  outputs_*/11_figures_tables/fig_paired_auc_contrasts.png
  outputs_*/11_figures_tables/fig_calibration_curves.png

Self-contained: no imports from other project scripts. Requires stages
03, 05, 06, 07, 08, 09 and 10 to have already run.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == "scripts":
    ROOT = ROOT.parent


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUT = OUTROOT / "11_figures_tables"
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "Ablation": OUTROOT / "05_ablation" / "ablation_metrics_by_fold.csv",
    "BiasControlled": OUTROOT / "06_bias_controlled" / "bias_controlled_metrics.csv",
    "AgeMatching": OUTROOT / "06_bias_controlled" / "repeated_age_matching_metrics.csv",
    "Stress": OUTROOT / "07_stress_test" / "small_control_stress_all_models.csv",
    "Bootstrap": OUTROOT / "08_robustness" / "full_pipeline_bootstrap_all_models.csv",
    "Contrasts": OUTROOT / "08_robustness" / "paired_bootstrap_contrasts.csv",
    "Permutation": OUTROOT / "08_robustness" / "permutation_test_summary.csv",
    "Calibration": OUTROOT / "09_calibration" / "calibration_summary.csv",
    "CalibrationBootstrap": OUTROOT / "09_calibration" / "calibration_bootstrap_distributions.csv",
    "Simulation": OUTROOT / "10_simulation" / "semi_synthetic_factorial_validation.csv",
    "RawCoverage": OUTROOT / "03_missingness" / "raw_coverage_fisher_fdr.csv",
    "HarmonizedCoverage": OUTROOT / "03_missingness" / "harmonized_coverage_fisher_fdr.csv",
}

with pd.ExcelWriter(OUT / "Manuscript_Tables.xlsx", engine="openpyxl") as w:
    for name, p in files.items():
        if p.exists():
            pd.read_csv(p).to_excel(w, sheet_name=name[:31], index=False)

# --- model performance summary + forest plot --------------------------------
a = pd.read_csv(files["Ablation"])
summ = a.groupby("spec").agg(
    mean_auc=("roc_auc", "mean"), sd_auc=("roc_auc", "std"), median_auc=("roc_auc", "median"),
    auc_q025=("roc_auc", lambda x: x.quantile(.025)), auc_q975=("roc_auc", lambda x: x.quantile(.975)),
    mean_bacc=("balanced_accuracy", "mean"), mean_brier=("brier", "mean"), n_folds=("fold", "nunique"),
).reset_index()
summ.to_csv(OUT / "table_model_performance_summary.csv", index=False)

s = summ.sort_values("mean_auc")
plt.figure(figsize=(10, max(6, len(s) * .28)))
plt.errorbar(s.mean_auc, range(len(s)), xerr=np.vstack([s.mean_auc - s.auc_q025, s.auc_q975 - s.mean_auc]), fmt="o")
plt.yticks(range(len(s)), s.spec)
plt.xlabel("ROC-AUC with empirical 95% fold interval")
plt.tight_layout()
plt.savefig(OUT / "fig_model_performance.png", dpi=300)
plt.close()

# --- coverage-difference bar charts ------------------------------------------
for label, key in [("raw", "RawCoverage"), ("harmonized", "HarmonizedCoverage")]:
    d = pd.read_csv(files[key]).head(30).sort_values("delta_coverage")
    plt.figure(figsize=(9, 8))
    plt.barh(d.feature, d.delta_coverage)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Coverage difference: PCOS - control")
    plt.tight_layout()
    plt.savefig(OUT / f"fig_{label}_coverage_difference.png", dpi=300)
    plt.close()

# --- missingness heatmaps -----------------------------------------------------
for label, path in [("raw", OUTROOT / "01_schema_audit" / "raw_numeric_matrix.parquet"),
                     ("harmonized", OUTROOT / "02_harmonization" / "harmonized_matrix.parquet")]:
    if path.exists():
        x = pd.read_parquet(path)
        order = np.argsort(x.isna().mean(axis=1).to_numpy())
        cols = x.isna().mean().sort_values(ascending=False).head(60).index
        plt.figure(figsize=(12, 8))
        plt.imshow(x.loc[x.index[order], cols].isna(), aspect="auto", interpolation="nearest")
        plt.xlabel("Features")
        plt.ylabel("Participants ordered by missingness")
        plt.tight_layout()
        plt.savefig(OUT / f"fig_{label}_missingness_heatmap.png", dpi=300)
        plt.close()

# --- paired bootstrap contrasts forest plot -----------------------------------
d = pd.read_csv(files["Contrasts"])
d = d[(d.metric == "roc_auc") & (d.status == "PASS")].copy()
if len(d):
    d = d.sort_values("mean_difference")
    xerr = np.vstack([d.mean_difference - d.ci_low, d.ci_high - d.mean_difference])
    plt.figure(figsize=(10, max(5, len(d) * .35)))
    plt.errorbar(d.mean_difference, range(len(d)), xerr=xerr, fmt="o")
    plt.axvline(0, linewidth=1)
    plt.yticks(range(len(d)), d.contrast)
    plt.xlabel("Paired bootstrap difference in ROC-AUC")
    plt.tight_layout()
    plt.savefig(OUT / "fig_paired_auc_contrasts.png", dpi=300)
    plt.close()

# --- reliability curves -------------------------------------------------------
rp = OUTROOT / "09_calibration" / "reliability_curve_bootstrap_bands.csv"
if rp.exists():
    r = pd.read_csv(rp)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "--")
    for spec, g in r.groupby("spec"):
        if spec.endswith("logistic"):
            plt.plot(g.predicted_mean, g.observed_mean, label=spec)
            plt.fill_between(g.predicted_mean, g.ci_low, g.ci_high, alpha=.08)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT / "fig_calibration_curves.png", dpi=300)
    plt.close()

# --- simulation summary table -------------------------------------------------
sim = pd.read_csv(files["Simulation"])
ss = sim.groupby(["space", "mode", "bio_signal", "workflow_shift", "schema_shift"]).roc_auc.mean().reset_index()
ss.to_csv(OUT / "table_simulation_summary.csv", index=False)
