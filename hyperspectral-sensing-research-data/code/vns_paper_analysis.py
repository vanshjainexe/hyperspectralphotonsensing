"""Auditable comparisons for the manuscript; no test-driven design selection."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("VNS_paper")
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)
old = pd.read_csv("VNS_research_package/data/scene_condition_metrics.csv")
paths = sorted(Path("vns_quantile_outputs").glob("*.csv"))
assert len(paths) == 15
raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
assert (
    len(raw) == 9720
    and not raw.duplicated(
        ["scene", "optics", "photons", "channels", "decoder", "trial"]
    ).any()
)
assert (raw.channels * raw.bits == 24).all()
assert np.isfinite(raw[["rmse", "sam", "psnr", "validation_rmse"]]).all().all()
keys = ["scene", "config", "budget", "optics", "photons", "channels", "bits", "decoder"]
q = raw.groupby(keys, as_index=False).mean(numeric_only=True)
raw.to_csv(DATA / "quantile_trial_metrics.csv", index=False)
q.to_csv(DATA / "quantile_scene_metrics.csv", index=False)
base = old[old.config.isin(["main", "sqrt_adc"])].copy()
combined = pd.concat([base, q], ignore_index=True)
combined.to_csv(DATA / "main_and_quantizer_scene_metrics.csv", index=False)
names = sorted(combined.scene.unique())
ids = np.random.default_rng(20261014).integers(0, 15, (10000, 15))


def stats(x):
    x = np.asarray(x)
    boot = x[ids].mean(1)
    return [float(x.mean()), *map(float, np.percentile(boot, [2.5, 97.5]))]


agg = combined.groupby(keys[1:], as_index=False)[["rmse", "sam", "psnr"]].mean()
agg.to_csv(DATA / "quantizer_cross_scene_metrics.csv", index=False)
win = agg.loc[agg.groupby(["config", "optics", "photons", "decoder"]).rmse.idxmin()]
win.to_csv(DATA / "quantizer_descriptive_winners.csv", index=False)
strategies = {
    "fixed3_uniform": lambda g: g[(g.config == "main") & (g.channels == 3)],
    "selected_uniform": lambda g: g[g.config == "main"],
    "selected_sqrt": lambda g: g[g.config == "sqrt_adc"],
    "selected_quantile": lambda g: g[g.config == "quantile_adc"],
    "fixed3_calibrated": lambda g: g[g.channels == 3],
    "joint_selection": lambda g: g,
}
selected = []
for (scene, mode, n), g in combined[combined.decoder == "matched"].groupby(
    ["scene", "optics", "photons"]
):
    for strategy, restrict in strategies.items():
        pool = restrict(g).sort_values(["config", "channels"])
        row = pool.loc[pool.validation_rmse.idxmin()]
        selected.append(
            {
                "scene": scene,
                "optics": mode,
                "photons": n,
                "strategy": strategy,
                "selected_config": row.config,
                "selected_k": int(row.channels),
                "selected_b": int(row.bits),
                "validation_rmse": row.validation_rmse,
                "rmse": row.rmse,
                "sam": row.sam,
                "psnr": row.psnr,
            }
        )
sel = pd.DataFrame(selected)
sel.to_csv(DATA / "paper_validation_choices.csv", index=False)
summary = []
for (mode, n, strategy), g in sel.groupby(["optics", "photons", "strategy"]):
    g = g.sort_values("scene")
    mean, lo, hi = stats(g.rmse)
    summary.append(
        {
            "optics": mode,
            "photons": n,
            "strategy": strategy,
            "rmse": mean,
            "ci_low": lo,
            "ci_high": hi,
            "sam": float(g.sam.mean()),
            "psnr": float(g.psnr.mean()),
        }
    )
summary = pd.DataFrame(summary)
summary.to_csv(DATA / "paper_strategy_summary.csv", index=False)
effects = []
contrasts = [
    ("selected_uniform", "fixed3_uniform"),
    ("selected_quantile", "selected_uniform"),
    ("joint_selection", "fixed3_uniform"),
    ("joint_selection", "selected_uniform"),
    ("joint_selection", "fixed3_calibrated"),
]
for (mode, n), g in sel.groupby(["optics", "photons"]):
    piv = g.pivot(index="scene", columns="strategy", values="rmse").reindex(names)
    for a, b in contrasts:
        av = piv[a].to_numpy()
        bv = piv[b].to_numpy()
        m, lo, hi = stats(bv - av)
        gain = 100 * (bv.mean() - av.mean()) / bv.mean()
        boot = 100 * (bv[ids].mean(1) - av[ids].mean(1)) / bv[ids].mean(1)
        effects.append(
            {
                "optics": mode,
                "photons": n,
                "candidate": a,
                "reference": b,
                "absolute_rmse_gain": m,
                "abs_ci_low": lo,
                "abs_ci_high": hi,
                "relative_gain_pct": gain,
                "relative_ci_low": float(np.percentile(boot, 2.5)),
                "relative_ci_high": float(np.percentile(boot, 97.5)),
                "scenes_improved": int(np.sum(av < bv)),
            }
        )
effects = pd.DataFrame(effects)
effects.to_csv(DATA / "paper_paired_effects.csv", index=False)
occup = (
    q[q.decoder == "matched"]
    .groupby(["optics", "photons", "channels", "bits"], as_index=False)[
        [
            "mean_code_entropy_bits",
            "mean_dominant_code_fraction",
            "uniform_mean_code_entropy_bits",
            "uniform_mean_dominant_code_fraction",
        ]
    ]
    .mean()
)
occup.to_csv(DATA / "paper_code_occupancy.csv", index=False)

checks = {
    "extension_rows": len(raw),
    "unique_trial_keys": True,
    "finite_metrics": True,
    "budget": 24,
    "scenes": 15,
    "bootstrap_resamples": 10000,
    "scope": "exploratory extension on same 15 scenes",
    "selection": "minimum validation scene-mean RMSE, never test RMSE",
}
(DATA / "paper_analysis_checks.json").write_text(json.dumps(checks, indent=2))
print("HIGH-LIGHT STRATEGIES")
print(summary[summary.photons == 100000].to_string(index=False))
print("HIGH-LIGHT EFFECTS")
print(effects[effects.photons == 100000].to_string(index=False))
print("QUANTILE MATCHED WINNERS")
print(
    win[(win.config == "quantile_adc") & (win.decoder == "matched")][
        ["optics", "photons", "channels", "bits", "rmse"]
    ].to_string(index=False)
)
print("ONE-BIT OCCUPANCY")
print(
    occup[(occup.channels == 24) & occup.photons.isin([30, 1000, 100000])].to_string(
        index=False
    )
)
