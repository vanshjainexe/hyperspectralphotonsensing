"""Check reproducibility artifacts and assemble the bounded computational study."""

from pathlib import Path
import hashlib, json, platform, shutil, subprocess, sys, zipfile
import importlib.metadata
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path.cwd()
OUT = ROOT / "VNS_research_package"
CODE = OUT / "code"
PROTO = OUT / "protocol"
PILOT = OUT / "pilot_v1"
for p in [CODE, PROTO, PILOT]:
    p.mkdir(exist_ok=True)
scripts = [
    "vns_budget_sweep.py",
    "aggregate_loso.py",
    "vns_final_figures.py",
    "vns_robustness.py",
    "vns_robust_analysis.py",
    "vns_image_examples.py",
    "vns_system_diagram.py",
    "test_vns.py",
    "build_research_report.py",
    "package_and_validate.py",
]
for name in scripts:
    shutil.copy2(ROOT / name, CODE / name)
for name in ["protocol.json", "folds.json", "sample_cache.npz"]:
    shutil.copy2(ROOT / "vns_robust_outputs" / name, PROTO / name)
shutil.copy2(ROOT / "robustness_run.log", PROTO / "runtime_log.txt")
shutil.copytree(
    ROOT / "vns_final_outputs", PILOT / "figures_and_summaries", dirs_exist_ok=True
)
shutil.copytree(
    ROOT / "cave_final_84", PILOT / "main_84pct_scene_runs", dirs_exist_ok=True
)
shutil.copytree(
    ROOT / "cave_efficiency",
    PILOT / "efficiency_sensitivity_scene_runs",
    dirs_exist_ok=True,
)
shutil.copytree(
    ROOT / "cave_loso", PILOT / "ideal_router_scene_runs", dirs_exist_ok=True
)
for png in OUT.rglob("*.png"):
    with Image.open(png) as im:
        im.verify()

raw = pd.read_csv(OUT / "data/all_trial_metrics.csv")
keys = [
    "scene",
    "config",
    "budget",
    "optics",
    "photons",
    "channels",
    "bits",
    "decoder",
    "trial",
]
assert len(raw) == 72090, len(raw)
assert not raw.duplicated(keys).any()
assert (raw.channels * raw.bits == raw.budget).all()
assert np.isfinite(raw[["rmse", "sam", "psnr", "validation_rmse"]].to_numpy()).all()
assert raw.scene.nunique() == 15 and raw.config.nunique() == 12
assert (
    raw.groupby(
        ["scene", "config", "optics", "photons", "channels", "bits", "decoder"]
    ).size()
    == 3
).all()
folds = json.loads((PROTO / "folds.json").read_text())
for f in folds:
    tr = set(f["training_scenes"])
    va = set(f["validation_scenes"])
    te = {f["test_scene"]}
    assert not (tr & va or tr & te or va & te)
    assert len(tr) == 11 and len(va) == 3
metrics = pd.read_csv(OUT / "data/scene_condition_metrics.csv")
choices = pd.read_csv(OUT / "data/validation_selected_designs.csv")
for _, r in choices.iterrows():
    pool = metrics[
        (metrics.scene == r.scene)
        & (metrics.config == r.config)
        & (metrics.optics == r.optics)
        & (metrics.photons == r.photons)
        & (metrics.decoder == r.decoder)
    ]
    chosen = pool.loc[pool.validation_rmse.idxmin()]
    assert int(chosen.channels) == int(r.selected_k)
    assert abs(chosen.rmse - r.selected_rmse) < 1e-12

env = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {
        x: importlib.metadata.version(x)
        for x in ["numpy", "scipy", "pandas", "matplotlib", "Pillow", "reportlab"]
    },
}
(PROTO / "environment.json").write_text(json.dumps(env, indent=2))
(CODE / "requirements.txt").write_text(
    "\n".join(f'{x}=={env["packages"][x]}' for x in env["packages"]) + "\n"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


sources = []
for p in sorted((ROOT / "cave_extracted").rglob("*_ms_*.png")):
    sources.append(
        {
            "path": str(p.relative_to(ROOT / "cave_extracted")),
            "bytes": p.stat().st_size,
            "sha256": sha(p),
        }
    )
assert len(sources) == 465
pd.DataFrame(sources).to_csv(PROTO / "source_image_checksums.csv", index=False)
archive = ROOT / "upload/archive.zip"
if archive.exists():
    (PROTO / "source_archive.json").write_text(
        json.dumps(
            {
                "filename": "archive.zip",
                "bytes": archive.stat().st_size,
                "sha256": sha(archive),
            },
            indent=2,
        )
    )
checks = subprocess.run(
    [sys.executable, str(CODE / "test_vns.py")], capture_output=True, text=True
)
assert checks.returncode == 0, checks.stderr
(PROTO / "unit_test_results.txt").write_text(checks.stdout + checks.stderr)
qa = {
    "status": "passed",
    "trial_rows": len(raw),
    "scene_configuration_files": 180,
    "source_spectral_pngs": 465,
    "scenes": 15,
    "configs": 12,
    "fold_disjointness": "passed",
    "validation_only_selection_recomputed": "passed",
    "unique_trial_keys": "passed",
    "fixed_budget_conservation": "passed",
    "finite_metrics": "passed",
    "all_png_integrity": "passed",
    "physics_and_software_unit_tests": 10,
    "figures_png": len(list((OUT / "figures").glob("*.png"))),
    "legacy_png": len(list((PILOT / "figures_and_summaries").glob("figure_*.png"))),
    "scope": "15-scene computational study; not hardware or external-dataset validation",
}
(PROTO / "validation_report.json").write_text(json.dumps(qa, indent=2))
files = [
    p
    for p in sorted(OUT.rglob("*"))
    if p.is_file()
    and "__pycache__" not in p.parts
    and p.name != "package_checksums.csv"
]
pd.DataFrame(
    [
        {"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in files
    ]
).to_csv(PROTO / "package_checksums.csv", index=False)
dest = ROOT / "VNS_complete_computational_study.zip"
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(ROOT))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
print(json.dumps(qa, indent=2))
print("ZIP bytes", dest.stat().st_size)
