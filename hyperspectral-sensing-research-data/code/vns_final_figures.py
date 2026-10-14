from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from scipy.stats import spearmanr

MAIN = Path("cave_final_84")
OUT = Path("vns_final_outputs")
OUT.mkdir(exist_ok=True)
PAIRS = [(3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (24, 1)]
PHOTONS = [30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Figure 0: concept and forward model.
fig, axis = plt.subplots(figsize=(13.5, 5.2))
axis.set_xlim(0, 13.5)
axis.set_ylim(0, 5.2)
axis.axis("off")


def box(x, y, width, height, title, body, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.70,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        x + width / 2, y + height * 0.35, body, ha="center", va="center", fontsize=9
    )


def arrow(x1, y1, x2, y2, label=""):
    axis.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "0.25"},
    )
    if label:
        axis.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.18,
            label,
            ha="center",
            fontsize=8,
            color="0.3",
        )


box(0.3, 1.75, 2.0, 1.5, "Scene spectrum", "31 CAVE bands\n400–700 nm", "#3b82f6")
box(3.0, 3.15, 2.2, 1.35, "Spectral router", "K outputs\nthroughput η", "#16a085")
box(3.0, 0.55, 2.2, 1.35, "Absorptive filters", "K sites\narea penalty 1/K", "#e67e22")
box(
    6.0,
    1.75,
    2.1,
    1.5,
    "Photon detection",
    "Poisson shot noise\nset by light N",
    "#8e44ad",
)
box(8.85, 1.75, 1.75, 1.5, "ADC", "B bits/channel\nK × B = 24", "#c0392b")
box(11.35, 1.75, 1.8, 1.5, "Decoder", "Reconstruct\n31-band spectrum", "#2c3e50")
arrow(2.3, 2.55, 3.0, 3.75)
arrow(2.3, 2.45, 3.0, 1.20)
arrow(5.2, 3.75, 6.0, 2.75, "more photons")
arrow(5.2, 1.20, 6.0, 2.25, "fewer photons")
arrow(8.1, 2.5, 8.85, 2.5)
arrow(10.6, 2.5, 11.35, 2.5)
axis.text(
    6.75,
    4.85,
    "Fixed-budget routed snapshot hyperspectral sensing",
    ha="center",
    fontsize=16,
    fontweight="bold",
)
axis.text(
    6.75,
    0.12,
    "Optimization: choose K and B that minimize held-out spectral error at each N and optical throughput",
    ha="center",
    fontsize=10.5,
)
fig.tight_layout()
fig.savefig(OUT / "figure_0_system_model.png", dpi=220, bbox_inches="tight")
plt.close(fig)


scene_dirs = sorted(path for path in MAIN.iterdir() if path.is_dir())
scenes = [path.name for path in scene_dirs]

summary_rows = []
null_rmse = {}
dimensions = {}
decomp_rows = []
clean_rows = []

for scene_dir in scene_dirs:
    scene = scene_dir.name
    for row in read_csv(scene_dir / "summary_metrics.csv"):
        summary_rows.append(
            {
                "scene": scene,
                "optics": row["optics"],
                "photons": int(row["photons"]),
                "channels": int(row["channels"]),
                "bits": int(row["bits"]),
                "rmse": float(row["spectral_rmse_mean"]),
                "sam": float(row["sam_deg_mean"]),
                "psnr": float(row["psnr_db_mean"]),
            }
        )
    for row in read_csv(scene_dir / "error_decomposition.csv"):
        decomp_rows.append(
            {
                "scene": scene,
                "optics": row["optics"],
                "photons": int(row["photons"]),
                "channels": int(row["channels"]),
                "bits": int(row["bits"]),
                "trial": int(row["trial"]),
                "component": row["component"],
                "rmse": float(row["spectral_rmse"]),
            }
        )
    for row in read_csv(scene_dir / "clean_reconstruction_ceiling.csv"):
        if row["optics"] == "ideal_router":
            clean_rows.append(
                {
                    "scene": scene,
                    "channels": int(row["channels"]),
                    "bits": int(row["bits"]),
                    "rmse": float(row["spectral_rmse"]),
                }
            )
    null_rmse[scene] = float(
        read_csv(scene_dir / "null_baseline.csv")[0]["spectral_rmse"]
    )
    manifest = json.loads((scene_dir / "run_manifest.json").read_text())
    dimensions[scene] = manifest["test_intrinsic_dimension"]


grouped = defaultdict(list)
for row in summary_rows:
    grouped[(row["optics"], row["photons"], row["channels"], row["bits"])].append(row)

aggregate_rows = []
for (optics, photons, channels, bits), rows in sorted(grouped.items()):
    for metric in ("rmse", "sam", "psnr"):
        values = np.array([row[metric] for row in rows], dtype=float)
        aggregate_rows.append(
            {
                "optics": optics,
                "photons": photons,
                "channels": channels,
                "bits": bits,
                "metric": metric,
                "scene_mean": float(values.mean()),
                "scene_std": float(values.std(ddof=1)),
                "ci95_halfwidth": float(
                    1.96 * values.std(ddof=1) / np.sqrt(len(values))
                ),
                "scenes": len(values),
            }
        )
write_csv(OUT / "all_metrics_cross_scene.csv", aggregate_rows)


def metric_row(optics: str, photons: int, channels: int, metric: str = "rmse") -> dict:
    return next(
        row
        for row in aggregate_rows
        if row["optics"] == optics
        and row["photons"] == photons
        and row["channels"] == channels
        and row["metric"] == metric
    )


# Figure 1: all allocation curves with scene-level confidence intervals.
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), sharey=True)
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(PAIRS)))
for axis, optics in zip(axes, ["ideal_router", "absorptive_filter"]):
    for color, (channels, bits) in zip(colors, PAIRS):
        rows = [metric_row(optics, n, channels) for n in PHOTONS]
        y = np.array([row["scene_mean"] for row in rows])
        ci = np.array([row["ci95_halfwidth"] for row in rows])
        axis.plot(PHOTONS, y, marker="o", color=color, label=f"{channels}×{bits}")
        axis.fill_between(
            PHOTONS, np.maximum(y - ci, 1e-5), y + ci, color=color, alpha=0.10
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.22)
    axis.set_xlabel("Incident photons per macro-pixel")
    axis.set_title("84% Router" if optics == "ideal_router" else "Absorptive Filter")
axes[0].set_ylabel("Spectral RMSE (scene mean ± 95% CI)")
axes[0].legend(title="Channels × bits", fontsize=8)
fig.suptitle("Fixed 24-bit allocation across 15 held-out CAVE scenes")
fig.tight_layout()
fig.savefig(OUT / "figure_1_rmse_curves.png", dpi=220)
plt.close(fig)


# Winners and phase transition.
winners = []
for optics in ["ideal_router", "absorptive_filter"]:
    for photons in PHOTONS:
        candidates = [metric_row(optics, photons, channels) for channels, _ in PAIRS]
        winner = min(candidates, key=lambda row: row["scene_mean"])
        winners.append(dict(winner))
write_csv(OUT / "cross_scene_winners.csv", winners)

fig, axis = plt.subplots(figsize=(7.8, 4.8))
for optics, marker in [("ideal_router", "o"), ("absorptive_filter", "s")]:
    selected = [row for row in winners if row["optics"] == optics]
    axis.plot(
        [row["photons"] for row in selected],
        [row["channels"] for row in selected],
        marker=marker,
        linewidth=2.5,
        label="84% router" if optics == "ideal_router" else "absorptive filter",
    )
axis.set_xscale("log")
axis.set_yticks([3, 4, 6])
axis.set_ylim(2.8, 6.2)
axis.set_xlabel("Incident photons per macro-pixel")
axis.set_ylabel("Lowest-mean-RMSE channel count")
axis.set_title("Routing moves the 3×8 → 4×6 transition earlier")
axis.grid(True, which="both", alpha=0.25)
axis.legend()
fig.tight_layout()
fig.savefig(OUT / "figure_2_phase_transition.png", dpi=220)
plt.close(fig)


# Figure 3: per-scene winner prevalence.
winner_counts = defaultdict(Counter)
scene_winners = []
for scene in scenes:
    for optics in ["ideal_router", "absorptive_filter"]:
        for photons in PHOTONS:
            candidates = [
                row
                for row in summary_rows
                if row["scene"] == scene
                and row["optics"] == optics
                and row["photons"] == photons
            ]
            winner = min(candidates, key=lambda row: row["rmse"])
            key = (winner["channels"], winner["bits"])
            winner_counts[(optics, photons)][key] += 1
            scene_winners.append(
                {
                    "scene": scene,
                    "optics": optics,
                    "photons": photons,
                    "channels": winner["channels"],
                    "bits": winner["bits"],
                    "rmse": winner["rmse"],
                }
            )
write_csv(OUT / "per_scene_winners.csv", scene_winners)

fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
for axis, optics in zip(axes, ["ideal_router", "absorptive_filter"]):
    bottom = np.zeros(len(PHOTONS))
    for color, pair in zip(colors, PAIRS):
        values = np.array(
            [winner_counts[(optics, n)][pair] / len(scenes) * 100 for n in PHOTONS]
        )
        axis.bar(
            np.arange(len(PHOTONS)),
            values,
            bottom=bottom,
            color=color,
            label=f"{pair[0]}×{pair[1]}",
        )
        bottom += values
    axis.set_ylabel("Held-out scenes (%)")
    axis.set_title("84% Router" if optics == "ideal_router" else "Absorptive Filter")
    axis.set_ylim(0, 100)
axes[-1].set_xticks(np.arange(len(PHOTONS)), [f"{n:,}" for n in PHOTONS], rotation=30)
axes[-1].set_xlabel("Incident photons per macro-pixel")
axes[0].legend(ncol=3, fontsize=8, loc="upper right")
fig.suptitle("The best allocation varies with light level and scene")
fig.tight_layout()
fig.savefig(OUT / "figure_3_winner_prevalence.png", dpi=220)
plt.close(fig)


# Figure 4: router advantage using each mode's own optimal allocation.
advantage_rows = []
for photons in PHOTONS:
    router = next(
        row
        for row in winners
        if row["optics"] == "ideal_router" and row["photons"] == photons
    )
    filt = next(
        row
        for row in winners
        if row["optics"] == "absorptive_filter" and row["photons"] == photons
    )
    advantage = 100 * (filt["scene_mean"] - router["scene_mean"]) / filt["scene_mean"]
    advantage_rows.append(
        {"photons": photons, "router_rmse_reduction_percent": advantage}
    )
write_csv(OUT / "router_advantage.csv", advantage_rows)

fig, axis = plt.subplots(figsize=(7.5, 4.6))
axis.plot(
    PHOTONS,
    [row["router_rmse_reduction_percent"] for row in advantage_rows],
    marker="o",
    linewidth=2.5,
)
axis.set_xscale("log")
axis.set_xlabel("Incident photons per macro-pixel")
axis.set_ylabel("RMSE reduction from routing (%)")
axis.set_title("Routing advantage is largest in photon-starved conditions")
axis.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "figure_4_router_advantage.png", dpi=220)
plt.close(fig)


# Figure 5: error-source decomposition for the plausible frontier designs.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for axis, (channels, bits) in zip(axes, PAIRS[:3]):
    clean = np.mean([row["rmse"] for row in clean_rows if row["channels"] == channels])
    full_values, shot_values, quant_values = [], [], []
    for photons in PHOTONS:
        full_values.append(metric_row("ideal_router", photons, channels)["scene_mean"])
        shot_scene = []
        quant_scene = []
        for scene in scenes:
            shot = [
                row["rmse"]
                for row in decomp_rows
                if row["scene"] == scene
                and row["optics"] == "ideal_router"
                and row["photons"] == photons
                and row["channels"] == channels
                and row["component"] == "shot_noise_only"
            ]
            quant = [
                row["rmse"]
                for row in decomp_rows
                if row["scene"] == scene
                and row["optics"] == "ideal_router"
                and row["photons"] == photons
                and row["channels"] == channels
                and row["component"] == "quantization_only"
            ]
            shot_scene.append(np.mean(shot))
            quant_scene.append(np.mean(quant))
        shot_values.append(np.mean(shot_scene))
        quant_values.append(np.mean(quant_scene))
    axis.plot(PHOTONS, full_values, marker="o", label="Full model")
    axis.plot(PHOTONS, shot_values, marker="s", label="Shot noise only")
    axis.plot(PHOTONS, quant_values, marker="^", label="Quantization only")
    axis.axhline(clean, color="black", linestyle="--", label="Representation floor")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.22)
    axis.set_title(f"{channels} channels × {bits} bits")
    axis.set_xlabel("Incident photons")
axes[0].set_ylabel("Spectral RMSE")
axes[0].legend(fontsize=8)
fig.suptitle("Why each 24-bit allocation succeeds or fails (84% router)")
fig.tight_layout()
fig.savefig(OUT / "figure_5_error_decomposition.png", dpi=220)
plt.close(fig)


# Figure 6: performance relative to a mean-spectrum/no-information predictor.
information_rows = []
fig, axis = plt.subplots(figsize=(7.7, 4.7))
for optics, marker in [("ideal_router", "o"), ("absorptive_filter", "s")]:
    means, cis = [], []
    for photons in PHOTONS:
        ratios = []
        for scene in scenes:
            candidate = min(
                (
                    row
                    for row in summary_rows
                    if row["scene"] == scene
                    and row["optics"] == optics
                    and row["photons"] == photons
                ),
                key=lambda row: row["rmse"],
            )
            ratios.append(candidate["rmse"] / null_rmse[scene])
        means.append(float(np.mean(ratios)))
        cis.append(float(1.96 * np.std(ratios, ddof=1) / np.sqrt(len(ratios))))
        information_rows.append(
            {
                "optics": optics,
                "photons": photons,
                "best_rmse_over_null_mean": means[-1],
                "ci95_halfwidth": cis[-1],
            }
        )
    axis.errorbar(
        PHOTONS,
        means,
        yerr=cis,
        marker=marker,
        capsize=3,
        linewidth=2,
        label=optics.replace("_", " "),
    )
axis.axhline(1.0, color="black", linestyle="--", label="No-information baseline")
axis.set_xscale("log")
axis.set_xlabel("Incident photons per macro-pixel")
axis.set_ylabel("Best RMSE / mean-spectrum RMSE")
axis.set_title("Measurements become informative as the ratio falls below 1")
axis.grid(True, which="both", alpha=0.25)
axis.legend()
fig.tight_layout()
fig.savefig(OUT / "figure_6_information_gain.png", dpi=220)
plt.close(fig)
write_csv(OUT / "information_gain.csv", information_rows)


# Figure 7: scene spectral complexity versus the high-light winner.
complexity_rows = []
for scene in scenes:
    winner = next(
        row
        for row in scene_winners
        if row["scene"] == scene
        and row["optics"] == "ideal_router"
        and row["photons"] == 100_000
    )
    complexity_rows.append(
        {
            "scene": scene,
            "effective_rank": float(dimensions[scene]["effective_rank"]),
            "pca_components_95pct": int(dimensions[scene]["pca_components_95pct"]),
            "pca_components_99pct": int(dimensions[scene]["pca_components_99pct"]),
            "winning_channels_100k": winner["channels"],
        }
    )
rho, pvalue = spearmanr(
    [row["effective_rank"] for row in complexity_rows],
    [row["winning_channels_100k"] for row in complexity_rows],
)
write_csv(OUT / "scene_complexity.csv", complexity_rows)

rng = np.random.default_rng(20261013)
fig, axis = plt.subplots(figsize=(7.3, 4.7))
axis.scatter(
    [row["effective_rank"] for row in complexity_rows],
    [row["winning_channels_100k"] + rng.normal(0, 0.035) for row in complexity_rows],
    s=55,
)
axis.set_xlabel("Effective spectral rank of held-out scene")
axis.set_ylabel("Winning channels at 100,000 photons")
axis.set_yticks([3, 4, 6])
axis.set_title(f"Spectral complexity hypothesis: Spearman ρ={rho:.2f}, p={pvalue:.3f}")
axis.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "figure_7_complexity_vs_channels.png", dpi=220)
plt.close(fig)


# Figure 8: router-efficiency/channel-count phase map.
eff_roots = {
    0.40: Path("cave_efficiency/0.4"),
    0.60: Path("cave_efficiency/0.6"),
    0.84: Path("cave_final_84"),
    1.00: Path("cave_loso"),
}
phase_rows = []
phase = np.zeros((len(eff_roots), len(PHOTONS)), dtype=int)
for ei, (efficiency, root) in enumerate(eff_roots.items()):
    bucket = defaultdict(list)
    for path in root.glob("*/summary_metrics.csv"):
        for row in read_csv(path):
            if row["optics"] != "ideal_router":
                continue
            key = (int(row["photons"]), int(row["channels"]), int(row["bits"]))
            bucket[key].append(float(row["spectral_rmse_mean"]))
    for ni, photons in enumerate(PHOTONS):
        candidates = []
        for channels, bits in PAIRS:
            values = bucket[(photons, channels, bits)]
            candidates.append((float(np.mean(values)), channels, bits))
        mean_rmse, channels, bits = min(candidates)
        phase[ei, ni] = channels
        phase_rows.append(
            {
                "router_efficiency": efficiency,
                "photons": photons,
                "channels": channels,
                "bits": bits,
                "cross_scene_mean_rmse": mean_rmse,
            }
        )
write_csv(OUT / "router_efficiency_phase_map.csv", phase_rows)

fig, axis = plt.subplots(figsize=(10.2, 4.5))
shown_channels = sorted(set(phase.ravel()))
channel_to_index = {channel: index for index, channel in enumerate(shown_channels)}
phase_indices = np.vectorize(channel_to_index.get)(phase)
palette = ["#472c7a", "#2a788e", "#7ad151"][: len(shown_channels)]
image = axis.imshow(
    phase_indices,
    aspect="auto",
    cmap=ListedColormap(palette),
    vmin=-0.5,
    vmax=len(shown_channels) - 0.5,
)
axis.set_xticks(np.arange(len(PHOTONS)), [f"{n:,}" for n in PHOTONS], rotation=30)
axis.set_yticks(np.arange(len(eff_roots)), [f"{eff:.0%}" for eff in eff_roots])
axis.set_xlabel("Incident photons per macro-pixel")
axis.set_ylabel("Router throughput")
axis.set_title("Design rule: best 24-bit allocation versus light and router efficiency")
for ei in range(phase.shape[0]):
    for ni in range(phase.shape[1]):
        channels = phase[ei, ni]
        bits = dict(PAIRS)[channels]
        axis.text(
            ni,
            ei,
            f"{channels}×{bits}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
axis.legend(
    handles=[
        Patch(facecolor=palette[index], label=f"{channel} channels")
        for index, channel in enumerate(shown_channels)
    ],
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    framealpha=0.92,
)
fig.tight_layout()
fig.savefig(OUT / "figure_8_efficiency_phase_map.png", dpi=220)
plt.close(fig)


report = {
    "scenes": len(scenes),
    "router_efficiency_main": 0.84,
    "spearman_effective_rank_vs_winning_channels_100k": float(rho),
    "spearman_p_value": float(pvalue),
    "router_crossover_photons": min(
        row["photons"]
        for row in winners
        if row["optics"] == "ideal_router" and row["channels"] > 3
    ),
    "filter_crossover_photons": min(
        row["photons"]
        for row in winners
        if row["optics"] == "absorptive_filter" and row["channels"] > 3
    ),
    "router_advantage_percent": advantage_rows,
}
(OUT / "analysis_summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
