"""Aggregate v2 experiments, validate design selection, make all final figures."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr, rankdata
from scipy.optimize import nnls
from vns_robustness import PAIRS, PHOTONS, intrinsic_dimension

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="vns_robust_outputs")
parser.add_argument("--out", default="VNS_research_package")
args = parser.parse_args()
ROOT = Path(args.root)
OUT = Path(args.out)
OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"
DATA = OUT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)
COLORS = ["#482878", "#365c8d", "#277f8e", "#1fa187", "#6ece58", "#b5de2b", "#e6ab02"]
frames = [pd.read_csv(p) for p in sorted(ROOT.glob("*/*.csv"))]
if len(frames) == 15 * 12:
    raw = pd.concat(frames, ignore_index=True)
elif not frames and (DATA / "all_trial_metrics.csv").exists():
    raw = pd.read_csv(DATA / "all_trial_metrics.csv")
else:
    raise RuntimeError(f"Incomplete sweep: {len(frames)} of 180 scene/config files")
assert raw[["rmse", "psnr", "sam", "validation_rmse"]].notna().all().all()
keys = ["scene", "config", "budget", "optics", "photons", "channels", "bits", "decoder"]
sc = raw.groupby(keys, as_index=False).mean(numeric_only=True)
raw.to_csv(DATA / "all_trial_metrics.csv", index=False)
sc.to_csv(DATA / "scene_condition_metrics.csv", index=False)
names = sorted(sc.scene.unique())
rng = np.random.default_rng(20261013)
indices = rng.integers(0, len(names), size=(10000, len(names)))


def ci(values):
    a = np.asarray(values)
    b = a[indices].mean(axis=1)
    return float(a.mean()), *map(float, np.percentile(b, [2.5, 97.5]))


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=190, bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


agg = []
for key, g in sc.groupby(keys[1:]):
    row = dict(zip(keys[1:], key))
    for metric in ["rmse", "sam", "psnr"]:
        m, lo, hi = ci(g.sort_values("scene")[metric].to_numpy())
        row.update({metric: m, metric + "_ci_low": lo, metric + "_ci_high": hi})
    agg.append(row)
agg = pd.DataFrame(agg)
agg.to_csv(DATA / "cross_scene_metrics.csv", index=False)
group = ["config", "budget", "optics", "photons", "decoder"]
winners = agg.loc[agg.groupby(group).rmse.idxmin()].sort_values(group)
winners.to_csv(DATA / "descriptive_global_winners.csv", index=False)

selected = []
for key, g in sc.groupby(["scene", "config", "budget", "optics", "photons", "decoder"]):
    row = dict(zip(["scene", "config", "budget", "optics", "photons", "decoder"], key))
    win = g.loc[g.validation_rmse.idxmin()]
    fixed = g[g.channels == 3].iloc[0]
    oracle = g.loc[g.rmse.idxmin()]
    row.update(
        {
            "selected_k": int(win.channels),
            "selected_b": int(win.bits),
            "selected_rmse": win.rmse,
            "fixed3_rmse": fixed.rmse,
            "test_oracle_rmse": oracle.rmse,
            "oracle_k": int(oracle.channels),
            "null_rmse": win.null_rmse,
            "selected_sam": win.sam,
            "selected_psnr": win.psnr,
            "selected_saturation": win.saturation_fraction,
            "selected_detected_photons": win.expected_detected_photons,
        }
    )
    selected.append(row)
sel = pd.DataFrame(selected)
sel.to_csv(DATA / "validation_selected_designs.csv", index=False)

benefits = []
for (config, budget, mode, n, decoder), g in sel.groupby(group):
    g = g.sort_values("scene")
    m, lo, hi = ci((g.fixed3_rmse - g.selected_rmse).to_numpy())
    benefits.append(
        {
            "config": config,
            "budget": budget,
            "optics": mode,
            "photons": int(n),
            "decoder": decoder,
            "fixed3_minus_selected_rmse": m,
            "ci_low": lo,
            "ci_high": hi,
            "relative_gain_pct": 100
            * (g.fixed3_rmse.mean() - g.selected_rmse.mean())
            / g.fixed3_rmse.mean(),
            "selected_rmse_mean": g.selected_rmse.mean(),
            "fixed3_rmse_mean": g.fixed3_rmse.mean(),
            "oracle_rmse_mean": g.test_oracle_rmse.mean(),
            "null_rmse_mean": g.null_rmse.mean(),
            "selected_sam_mean": g.selected_sam.mean(),
            "selected_psnr_mean": g.selected_psnr.mean(),
        }
    )
benefits = pd.DataFrame(benefits)
benefits.to_csv(DATA / "heldout_selection_benefits.csv", index=False)

main = agg[(agg.config == "main") & (agg.decoder == "matched")]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
for ax, mode in zip(axes, ["router", "filter"]):
    for color, (k, b) in zip(COLORS, PAIRS[24]):
        d = main[(main.optics == mode) & (main.channels == k)].sort_values("photons")
        ax.plot(d.photons, d.rmse, "o-", label=f"{k} x {b}", color=color, ms=4)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="White-reference photon exposure N",
        title="84% router" if mode == "router" else "Equal-area absorptive filter",
    )
    ax.grid(alpha=0.2, which="both")
axes[0].set_ylabel("Mean held-out spectral RMSE")
axes[1].legend(title="Channels x bits", fontsize=9)
fig.suptitle("Matched-noise affine decoder | 24 bits | 15 held-out scenes")
fig.tight_layout()
save(fig, "01_matched_decoder_error_curves")


def phase_plot(configs, decoders, rowlabels, title, name):
    fig, axes = plt.subplots(
        1, 2, figsize=(13, max(3.8, len(configs) * 0.49 + 1.8)), sharey=True
    )
    possible = [3, 4, 6, 8, 12, 16, 24]
    mapping = {k: i for i, k in enumerate(possible)}
    for ax, mode in zip(axes, ["router", "filter"]):
        z = np.zeros((len(configs), len(PHOTONS)))
        texts = []
        for r, (config, decoder) in enumerate(zip(configs, decoders)):
            cells = []
            for j, n in enumerate(PHOTONS):
                a = winners[
                    (winners.config == config)
                    & (winners.decoder == decoder)
                    & (winners.optics == mode)
                    & (winners.photons == n)
                ].iloc[0]
                z[r, j] = mapping[int(a.channels)]
                cells.append(f"{int(a.channels)}x{int(a.bits)}")
            texts.append(cells)
        ax.imshow(z, aspect="auto", cmap=ListedColormap(COLORS), vmin=-0.5, vmax=6.5)
        for r in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(
                    j,
                    r,
                    texts[r][j],
                    ha="center",
                    va="center",
                    color="white" if z[r, j] < 3.5 else "#17202a",
                    fontsize=9,
                )
        ax.set_xticks(
            range(len(PHOTONS)),
            ["10", "30", "100", "300", "1k", "3k", "10k", "30k", "100k"],
        )
        ax.set_yticks(range(len(rowlabels)), rowlabels)
        ax.set(
            xlabel="White-reference photons N",
            title="Router" if mode == "router" else "Absorptive filter",
        )
    fig.suptitle(
        title + "\nLowest mean test RMSE; descriptive winners, not a deployed selector",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, name)


phase_plot(
    ["main"] * 4,
    ["clean", "matched", "quadratic", "interpolation"],
    [
        "Clean-trained affine",
        "Matched-noise affine",
        "Matched quadratic",
        "Interpolation",
    ],
    "Does the decoder change the preferred allocation?",
    "02_decoder_sensitivity",
)
configs = [
    "main",
    "headroom_1",
    "headroom_2",
    "sqrt_adc",
    "read_2e",
    "sensor_stress",
    "crosstalk_15pct",
    "declining_throughput",
    "filter_transmission_80pct",
    "fractional_bands",
]
phase_plot(
    configs,
    ["matched"] * len(configs),
    [
        "Baseline",
        "ADC headroom 1.0",
        "ADC headroom 2.0",
        "Square-root ADC",
        "Read noise 2 e-",
        "Combined stress",
        "Router leakage 15%",
        "Declining efficiency",
        "Filter transmission 80%",
        "Equal-width overlap bands",
    ],
    "Sensitivity to calibration and hardware assumptions",
    "03_sensor_quantizer_sensitivity",
)
phase_plot(
    ["budget_12", "main", "budget_48"],
    ["matched"] * 3,
    ["12 bits", "24 bits", "48 bits"],
    "Changing the total output-bit budget",
    "04_total_budget_sensitivity",
)

msel = sel[(sel.config == "main") & (sel.decoder == "matched")]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), sharey=True)
for ax, mode in zip(axes, ["router", "filter"]):
    d = msel[msel.optics == mode]
    for metric, label, color in [
        ("selected_rmse", "Validation-selected", "#287d8e"),
        ("fixed3_rmse", "Fixed 3 x 8", "#713f93"),
        ("null_rmse", "Training-mean spectrum", "#8a8a8a"),
    ]:
        stats = [
            ci(d[d.photons == n].sort_values("scene")[metric].to_numpy())
            for n in PHOTONS
        ]
        a = np.array(stats)
        ax.plot(PHOTONS, a[:, 0], "o-", label=label, color=color, ms=4)
        ax.fill_between(PHOTONS, a[:, 1], a[:, 2], alpha=0.12, color=color)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="White-reference photons N",
        title=mode.title(),
    )
    ax.grid(alpha=0.2, which="both")
axes[0].set_ylabel("Mean held-out spectral RMSE")
axes[1].legend(fontsize=8)
fig.suptitle(
    "Choose using validation scenes; evaluate on a separate test scene\nShading: descriptive 95% scene-bootstrap intervals",
    fontsize=12,
)
fig.tight_layout()
save(fig, "05_honest_design_selection")

fig, ax = plt.subplots(figsize=(9, 4.7))
for mode, marker in [("router", "o"), ("filter", "s")]:
    d = benefits[
        (benefits.config == "main")
        & (benefits.decoder == "matched")
        & (benefits.optics == mode)
    ].sort_values("photons")
    ax.errorbar(
        d.photons,
        d.fixed3_minus_selected_rmse,
        yerr=[
            d.fixed3_minus_selected_rmse - d.ci_low,
            d.ci_high - d.fixed3_minus_selected_rmse,
        ],
        fmt=marker + "-",
        capsize=3,
        label=mode,
    )
ax.axhline(0, color="black", lw=1)
ax.set_xscale("log")
ax.set_xlabel("White-reference photons N")
ax.set_ylabel("Fixed 3 x 8 RMSE minus selected RMSE")
ax.set_title(
    "Does choosing an allocation improve on fixed 3 x 8?\nPositive values favor the validation-selected design"
)
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
save(fig, "06_selection_gain_uncertainty")

# Paired router/filter comparison using their independently validation-selected designs.
gains = []
for n in PHOTONS:
    d = (
        msel[msel.photons == n]
        .pivot(index="scene", columns="optics", values="selected_rmse")
        .reindex(names)
    )
    rb = d.router.to_numpy()
    fb = d["filter"].to_numpy()
    boot = 100 * (fb[indices].mean(1) - rb[indices].mean(1)) / fb[indices].mean(1)
    gains.append(
        {
            "photons": n,
            "router_gain_percent": 100 * (fb.mean() - rb.mean()) / fb.mean(),
            "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5),
        }
    )
gains = pd.DataFrame(gains)
gains.to_csv(DATA / "router_advantage_validation_selected.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(gains.photons, gains.router_gain_percent, "o-", color="#267f8e")
ax.fill_between(gains.photons, gains.ci_low, gains.ci_high, alpha=0.17, color="#267f8e")
ax.axhline(0, color="black", lw=1)
ax.set(
    xscale="log",
    xlabel="White-reference photons N",
    ylabel="Reduction in mean RMSE (%)",
    title="Router advantage after validation-only design selection",
)
ax.grid(alpha=0.2)
fig.tight_layout()
save(fig, "07_router_advantage")

# Exploratory complexity associations, evaluated honestly as associations, not predictors.
samples = np.load(ROOT / "sample_cache.npz")
complexity = []
correlations = []
for name in names:
    x = samples[name]
    rawrank = intrinsic_dimension(x)["effective_rank"]
    bright = x.mean(1) > 1e-3
    shape = x[bright] / x[bright].sum(1, keepdims=True)
    sr = intrinsic_dimension(shape)["effective_rank"]
    for decoder in ["clean", "matched", "quadratic", "interpolation"]:
        sub = sc[
            (sc.scene == name)
            & (sc.config == "main")
            & (sc.decoder == decoder)
            & (sc.optics == "router")
            & (sc.photons == 100000)
        ]
        k = int(sub.loc[sub.rmse.idxmin()].channels)
        complexity.append(
            {
                "scene": name,
                "decoder": decoder,
                "raw_effective_rank": rawrank,
                "shape_effective_rank": sr,
                "mean_brightness": float(x.mean()),
                "oracle_k_100k": k,
            }
        )
complexity = pd.DataFrame(complexity)
complexity.to_csv(DATA / "exploratory_complexity.csv", index=False)
for decoder, g in complexity.groupby("decoder"):
    for rank in ["raw_effective_rank", "shape_effective_rank", "mean_brightness"]:
        x = rankdata(g[rank])
        y = rankdata(g.oracle_k_100k)
        if y.std() == 0:
            rho = 0.0
            p = 1.0
        else:
            rho = float(np.corrcoef(x, y)[0, 1])
            xc = (x - x.mean()) / np.linalg.norm(x - x.mean())
            yc = (y - y.mean()) / np.linalg.norm(y - y.mean())
            null = np.array([xc @ rng.permutation(yc) for _ in range(20000)])
            p = float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / 20001)
        correlations.append(
            {
                "decoder": decoder,
                "rank_statistic": rank,
                "spearman_rho": rho,
                "permutation_p_unadjusted": p,
                "permutations": 20000,
                "exploratory": True,
            }
        )
correlations = pd.DataFrame(correlations)
correlations.to_csv(DATA / "complexity_correlations.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for ax, decoder in zip(axes, ["clean", "matched"]):
    d = complexity[complexity.decoder == decoder]
    c = correlations[
        (correlations.decoder == decoder)
        & (correlations.rank_statistic == "raw_effective_rank")
    ].iloc[0]
    ax.scatter(d.raw_effective_rank, d.oracle_k_100k, s=55, color="#277f8e", alpha=0.75)
    ax.set(
        xlabel="Effective rank (includes brightness variation)",
        title=f"{decoder} affine: rho={c.spearman_rho:.2f}, p={c.permutation_p_unadjusted:.3f}",
    )
    ax.set_yticks([3, 4, 6])
    ax.set_ylim(2.7, 6.4)
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Test-oracle K at N = 100,000")
fig.suptitle(
    "Exploratory complexity association | 15 scenes | unadjusted permutation p"
)
fig.tight_layout()
save(fig, "08_complexity_association")

# A compact, testable risk surrogate: C + A/N fitted to validation errors at 3 anchors.
surrogates = []
coeff = []
for (scene, mode), g in sc[(sc.config == "main") & (sc.decoder == "matched")].groupby(
    ["scene", "optics"]
):
    models = {}
    for k, kg in g.groupby("channels"):
        anchors = kg[kg.photons.isin([30, 1000, 30000])].sort_values("photons")
        a = np.c_[np.ones(3), 1000 / anchors.photons.to_numpy()]
        c = nnls(a, anchors.validation_rmse.to_numpy() ** 2)[0]
        models[int(k)] = c
        coeff.append(
            {
                "scene": scene,
                "optics": mode,
                "channels": int(k),
                "C": c[0],
                "A": c[1] * 1000,
            }
        )
    for n in PHOTONS:
        k = min(models, key=lambda k: models[k] @ [1, 1000 / n])
        chosen = g[(g.photons == n) & (g.channels == k)].iloc[0]
        fixed = g[(g.photons == n) & (g.channels == 3)].iloc[0]
        surrogates.append(
            {
                "scene": scene,
                "optics": mode,
                "photons": n,
                "predicted_k": k,
                "rmse": chosen.rmse,
                "fixed3_rmse": fixed.rmse,
                "anchor": n in [30, 1000, 30000],
            }
        )
surrogates = pd.DataFrame(surrogates)
surrogates.to_csv(DATA / "surrogate_rule_validation.csv", index=False)
pd.DataFrame(coeff).to_csv(DATA / "surrogate_rule_coefficients.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, mode in zip(axes, ["router", "filter"]):
    d = surrogates[surrogates.optics == mode]
    for col, label in [
        ("rmse", "Three-anchor C + A/N rule"),
        ("fixed3_rmse", "Fixed 3 x 8"),
    ]:
        z = d.groupby("photons")[col].mean()
        ax.plot(z.index, z.values, "o-", label=label)
    z = msel[msel.optics == mode].groupby("photons").selected_rmse.mean()
    ax.plot(z.index, z.values, "s--", label="Full validation sweep")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="White-reference photons N",
        title=mode.title(),
    )
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Mean test RMSE")
axes[1].legend(fontsize=8)
fig.suptitle(
    "Can three validation exposures predict a useful allocation?\nFitted at N=30, 1,000, 30,000; other exposures test interpolation/extrapolation"
)
fig.tight_layout()
save(fig, "09_compact_design_rule")

# Bootstrap winner support, explicitly descriptive rather than posterior probabilities.
supports = []
paired = []
for mode in ["router", "filter"]:
    for n in PHOTONS:
        d = sc[
            (sc.config == "main")
            & (sc.decoder == "matched")
            & (sc.optics == mode)
            & (sc.photons == n)
        ]
        p = d.pivot(index="scene", columns="channels", values="rmse").reindex(names)
        v = p.to_numpy()
        ww = np.argmin(v[indices[:3000]].mean(1), axis=1)
        for j, k in enumerate(p.columns):
            supports.append(
                {
                    "optics": mode,
                    "photons": n,
                    "channels": int(k),
                    "bootstrap_win_fraction": float(np.mean(ww == j)),
                }
            )
        m, lo, hi = ci((p[3] - p[4]).to_numpy())
        paired.append(
            {
                "optics": mode,
                "photons": n,
                "rmse_3_minus_4": m,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
pd.DataFrame(supports).to_csv(
    DATA / "descriptive_bootstrap_winner_support.csv", index=False
)
pd.DataFrame(paired).to_csv(DATA / "paired_3_vs_4.csv", index=False)

summary = {
    "scenes": len(names),
    "configurations": len(sc.config.unique()),
    "trial_condition_decoder_rows": len(raw),
    "main_matched_winners": winners[
        (winners.config == "main") & (winners.decoder == "matched")
    ][["optics", "photons", "channels", "bits", "rmse"]].to_dict("records"),
    "heldout_main_benefits": benefits[
        (benefits.config == "main") & (benefits.decoder == "matched")
    ].to_dict("records"),
    "complexity_correlations": correlations.to_dict("records"),
    "router_advantage": gains.to_dict("records"),
    "uncertainty": "10,000 paired scene bootstrap resamples, descriptive; overlapping CV training sets and selected dataset limit population inference",
    "surrogate_nonanchor_gain": {
        mode: float(100 * (d.fixed3_rmse.mean() - d.rmse.mean()) / d.fixed3_rmse.mean())
        for mode, d in surrogates[~surrogates.anchor].groupby("optics")
    },
}
(DATA / "results_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
