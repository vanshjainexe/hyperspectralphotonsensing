"""Recompute representative conditions and audit the paper's data path."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from vns_robustness import *
from vns_quantizer_extension import fit_quantile_adc, apply_adc

root = Path("vns_robust_outputs")
spec = json.loads((root / "protocol.json").read_text())
folds = json.loads((root / "folds.json").read_text())
cache = np.load(root / "sample_cache.npz")
fold = folds[0]
name = fold["test_scene"]
xtr = np.concatenate([cache[s][:600] for s in fold["training_scenes"]])
xv = np.concatenate([cache[s][:1024] for s in fold["validation_scenes"]])
xt = cache[name]
xx = np.tile(xtr, (2, 1))
seg = [(j * 1024, (j + 1) * 1024) for j in range(3)]
allold = pd.read_csv("VNS_research_package/data/all_trial_metrics.csv")
new = pd.read_csv("VNS_paper/data/quantile_trial_metrics.csv")
reports = []
for config, k, kind in [
    ("main", 6, "uniform"),
    ("sqrt_adc", 6, "sqrt"),
    ("quantile_adc", 24, "quantile"),
]:
    n = 100000
    b = 24 // k
    p = operator(k)
    cfg = Config(config, quantizer="sqrt" if kind == "sqrt" else "uniform")
    rng = lambda tag: np.random.default_rng(
        seed(spec["seed"], name, "router", n, k, tag)
    )

    def measurement(x, tag):
        return sense(x, p, n, k, b, "router", cfg, rng(tag), adc=kind != "quantile")[0]

    yt = np.concatenate([measurement(xtr, "train" + str(j)) for j in range(2)])
    yv = measurement(xv, "validation")
    if kind == "quantile":
        adc = fit_quantile_adc(yt, b)
        yt, _ = apply_adc(yt, adc)
        yv, _ = apply_adc(yv, adc)
    verr, alpha, dec = select_decoder(yt, xx, yv, xv, seg, "matched")
    source = new if kind == "quantile" else allold
    errors = []
    for trial in range(3):
        yy = measurement(xt, "test" + str(trial))
        if kind == "quantile":
            yy, _ = apply_adc(yy, adc)
        m = measure(xt, predict(yy, dec))
        row = source[
            (source.scene == name)
            & (source.config == config)
            & (source.optics == "router")
            & (source.photons == n)
            & (source.channels == k)
            & (source.decoder == "matched")
            & (source.trial == trial)
        ].iloc[0]
        for metric in ["rmse", "sam", "psnr"]:
            np.testing.assert_allclose(m[metric], row[metric], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(verr, row.validation_rmse, atol=1e-12)
        assert alpha == row.alpha
        errors.append(abs(m["rmse"] - row.rmse))
    reports.append(
        {
            "config": config,
            "scene": name,
            "photons": n,
            "channels": k,
            "trials": 3,
            "max_rmse_reproduction_error": max(errors),
        }
    )

# Recompute every joint selection from validation errors, with no test selection.
combined = pd.read_csv("VNS_paper/data/main_and_quantizer_scene_metrics.csv")
chosen = pd.read_csv("VNS_paper/data/paper_validation_choices.csv")
for _, r in chosen[chosen.strategy == "joint_selection"].iterrows():
    pool = combined[
        (combined.scene == r.scene)
        & (combined.optics == r.optics)
        & (combined.photons == r.photons)
        & (combined.decoder == "matched")
    ].sort_values(["config", "channels"])
    expected = pool.loc[pool.validation_rmse.idxmin()]
    assert expected.config == r.selected_config and expected.channels == r.selected_k
    np.testing.assert_allclose(expected.rmse, r.rmse, atol=1e-12)
assert len(chosen[chosen.strategy == "joint_selection"]) == 270
for f in folds:
    tr = set(f["training_scenes"])
    va = set(f["validation_scenes"])
    te = {f["test_scene"]}
    assert not (tr & va or tr & te or va & te)
report = {
    "status": "passed",
    "representative_recomputed_conditions": reports,
    "joint_decisions_recomputed": 270,
    "disjoint_scene_roles": "passed",
    "original_rows": len(allold),
    "extension_rows": len(new),
}
Path("VNS_paper/data/final_numerical_audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
