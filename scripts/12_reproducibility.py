"""
Stage 12 - Reproducibility manifest
---------------------------------------
Task: record the runtime environment (Python version, platform) and the
SHA-256 checksum of the configuration file, the feature mapping
dictionary, the source workbook (if present), and every file produced
under the output root so far. This manifest lets anyone verify that a
later re-run reproduced byte-identical inputs and outputs.

Result:
  outputs_*/12_reproducibility/reproducibility_manifest.json

Self-contained: no imports from other project scripts. Run this last,
after stages 00-11 have completed.
"""
from pathlib import Path
import json
import hashlib
import platform
import sys

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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
CFG = load_config(ROOT / "config" / "analysis_config.json")
OUTROOT = ROOT / ("outputs_smoke_test" if CFG.get("fast_mode") else "outputs_final")
OUTROOT.mkdir(exist_ok=True)
SEED = CFG["random_seed"]

manifest = {
    "python": sys.version,
    "platform": platform.platform(),
    "config_sha256": file_sha256(ROOT / "config" / "analysis_config.json"),
    "mapping_sha256": file_sha256(ROOT / "config" / "feature_dictionary.csv"),
    "input_sha256": file_sha256(ROOT / CFG["input_file"]) if (ROOT / CFG["input_file"]).exists() else None,
    "outputs": [],
}
for p in OUTROOT.rglob("*"):
    if p.is_file():
        manifest["outputs"].append({"path": str(p.relative_to(ROOT)), "sha256": file_sha256(p), "size": p.stat().st_size})

save_json(manifest, OUTROOT / "12_reproducibility" / "reproducibility_manifest.json")
print("Reproducibility manifest written for", len(manifest["outputs"]), "files")
