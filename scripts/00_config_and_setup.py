"""
Stage 00 - Configuration and output setup
------------------------------------------
Task: load the frozen analysis configuration, resolve the output root
(outputs_final for a publication run, outputs_smoke_test when
config/analysis_config.json has "fast_mode": true), create the per-stage
output folders, and persist the effective configuration that was actually
used for this run.

Result: outputs_*/00_config/effective_config.json

Self-contained: this script does not import any other project script or
module. It only reads plain data/config files (JSON). Run it first, in a
fresh Colab runtime or locally, before running stage 01 onward.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colab / local environment
# ---------------------------------------------------------------------------
# In Google Colab: clone the repository and change into its directory first,
# e.g.
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/PCOS_AIM_Acquisition_Bias_Framework
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


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUTROOT.mkdir(exist_ok=True)
SEED = CFG["random_seed"]

STAGE_FOLDERS = [
    "00_config", "01_schema_audit", "02_harmonization", "03_missingness",
    "04_adversarial", "05_ablation", "06_bias_controlled", "07_stress_test",
    "08_robustness", "09_calibration", "10_simulation", "11_figures_tables",
    "12_reproducibility",
]
for d in STAGE_FOLDERS:
    (OUTROOT / d).mkdir(parents=True, exist_ok=True)

save_json(CFG, OUTROOT / "00_config" / "effective_config.json")
print("Output root:", OUTROOT)
