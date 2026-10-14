"""Publication-sized vector figures generated directly from saved results."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from vns_robustness import PHOTONS, PAIRS

OUT = Path("VNS_paper")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
D = OUT / "data"
OLD = Path("VNS_research_package/data")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 150,
    }
)
colors = ["#345995", "#E59B26", "#20867D", "#965A9C", "#BA4A47", "#5C6B73"]


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=240)
    plt.close(fig)


def style(ax, ylabel=None):
    ax.set_xscale("log")
    ax.set_xlabel("White-reference exposure, N (photons)")
    ax.grid(axis="y", alpha=0.18)
    if ylabel:
        ax.set_ylabel(ylabel)


def label(ax, s):
    ax.text(0, 1.075, s, transform=ax.transAxes, fontweight="bold", fontsize=10)


# Physical path plus calibration/evaluation separation.
fig, ax = plt.subplots(figsize=(7.2, 2.65))
ax.set(xlim=(0, 10), ylim=(0, 4))
ax.axis("off")


def box(x, y, w, h, title, body, color="#EAF0F5"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=color,
            edgecolor="#BAC8D2",
            lw=0.65,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.73,
        title,
        ha="center",
        va="center",
        weight="bold",
        fontsize=7,
    )
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=6.2)


def arrow(a, b):
    ax.annotate(
        "", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#516273", "lw": 1}
    )


box(0.02, 2.05, 1.5, 1.15, "Spectrum", "31 reference bands")
box(2, 2.82, 1.5, 0.85, "Router", r"$N\eta P_Kx$", "#E6F1EF")
box(2, 1.5, 1.5, 0.85, "Filter", r"$N\tau P_Kx/K$", "#F8EDDC")
box(4, 2.05, 1.4, 1.15, "Detection", "Poisson counts")
box(5.87, 2.05, 1.55, 1.15, "Digitization", "$K$ channels\n$B$ bits each")
box(7.9, 2.05, 1.95, 1.15, "Reconstruction", "Estimate all 31 bands")
arrow((1.52, 2.8), (2, 3.24))
arrow((1.52, 2.43), (2, 1.93))
arrow((3.5, 3.24), (4, 2.8))
arrow((3.5, 1.93), (4, 2.4))
arrow((5.4, 2.62), (5.87, 2.62))
arrow((7.42, 2.62), (7.9, 2.62))
box(0.02, 0.12, 3.05, 0.94, "11 training scenes", "Fit decoder and quantile thresholds")
box(3.5, 0.12, 3.0, 0.94, "3 validation scenes", "Choose regularization, K and ADC")
box(6.94, 0.12, 2.91, 0.94, "1 held-out scene", "Evaluate; rotate through 15 scenes")
arrow((3.07, 0.59), (3.5, 0.59))
arrow((6.5, 0.59), (6.94, 0.59))
save(fig, "fig01_model")

agg = pd.read_csv(D / "quantizer_cross_scene_metrics.csv")
main = agg[(agg.config == "main") & (agg.decoder == "matched")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
for j, (ax, mode) in enumerate(zip(axes, ["router", "filter"])):
    for color, (k, b) in zip(colors, PAIRS[24]):
        t = main[(main.optics == mode) & (main.channels == k)].sort_values("photons")
        ax.plot(t.photons, t.rmse, "o-", color=color, label=f"{k} × {b}")
    ax.set_yscale("log")
    ax.set_title(
        "Router (η = 0.84)" if mode == "router" else "Equal-area filter (τ = 1)"
    )
    style(ax, "Mean spectral RMSE" if j == 0 else None)
    label(ax, "ab"[j])
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.53, 1.09),
    ncol=6,
    frameon=False,
)
fig.tight_layout()
save(fig, "fig02_allocation_curves")

win = pd.read_csv(D / "quantizer_descriptive_winners.csv")
rows = [
    ("main", "clean", "Clean affine / uniform"),
    ("main", "matched", "Matched affine / uniform"),
    ("main", "quadratic", "Quadratic / uniform"),
    ("sqrt_adc", "matched", "Matched affine / square-root"),
    ("quantile_adc", "matched", "Matched affine / quantile"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
cmap = ListedColormap(
    ["#E7EDF5", "#C8DFE8", "#95C4C0", "#70AA9C", "#407F78", "#24535E"]
)
for j, (ax, mode) in enumerate(zip(axes, ["router", "filter"])):
    z = np.zeros((len(rows), len(PHOTONS)))
    texts = []
    for r, (config, dec, name) in enumerate(rows):
        for col, n in enumerate(PHOTONS):
            w = win[
                (win.config == config)
                & (win.decoder == dec)
                & (win.optics == mode)
                & (win.photons == n)
            ].iloc[0]
            z[r, col] = [3, 4, 6, 8, 12, 24].index(int(w.channels))
            texts.append((col, r, int(w.channels), int(w.bits)))
    ax.imshow(z, aspect="auto", cmap=cmap, vmin=-0.5, vmax=5.5)
    for col, r, k, b in texts:
        ax.text(
            col,
            r,
            f"{k}×{b}",
            ha="center",
            va="center",
            fontsize=6.6,
            color="white" if k >= 12 else "#172A3A",
        )
    ax.set_xticks(
        range(9), ["10", "30", "100", "300", "1k", "3k", "10k", "30k", "100k"]
    )
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=7)
    ax.set_xlabel("White-reference exposure, N")
    ax.set_title(mode.title())
    label(ax, "ab"[j])
fig.tight_layout()
save(fig, "fig03_allocation_map")

summary = pd.read_csv(D / "paper_strategy_summary.csv")
strategies = [
    ("fixed3_uniform", "Fixed 3 × 8 / uniform", "#7F8790", "--"),
    ("selected_uniform", "Choose K / uniform", "#345995", "-"),
    ("selected_sqrt", "Choose K / square-root", "#20867D", "-"),
    ("selected_quantile", "Choose K / quantile", "#BA4A47", ":"),
    ("joint_selection", "Choose K + ADC", "#151D27", "-"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
for j, (ax, mode) in enumerate(zip(axes, ["router", "filter"])):
    for key, title, c, ls in strategies:
        t = summary[(summary.optics == mode) & (summary.strategy == key)].sort_values(
            "photons"
        )
        ax.plot(
            t.photons,
            t.rmse,
            ls,
            color=c,
            marker="o" if key == "joint_selection" else None,
            label=title,
        )
    ax.set_yscale("log")
    style(ax, "Mean held-out RMSE" if j == 0 else None)
    ax.set_title(mode.title())
    label(ax, "ab"[j])
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.53, 1.12),
    ncol=3,
    frameon=False,
)
fig.tight_layout()
save(fig, "fig04_selection")

occ = pd.read_csv(D / "paper_code_occupancy.csv")
q = occ[(occ.optics == "router") & (occ.channels == 24)]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
axes[0].plot(
    q.photons, q.uniform_mean_code_entropy_bits, "o-", label="Uniform", color="#345995"
)
axes[0].plot(
    q.photons, q.mean_code_entropy_bits, "s-", label="Quantile", color="#BA4A47"
)
axes[0].set(ylim=(-0.03, 1.03), title="Code use: 24 channels × 1 bit")
style(axes[0], "Mean marginal code entropy (bits)")
axes[0].legend(frameon=False)
for config, c, lab in [
    ("main", "#345995", "Uniform"),
    ("quantile_adc", "#BA4A47", "Quantile"),
]:
    t = agg[
        (agg.optics == "router")
        & (agg.channels == 24)
        & (agg.decoder == "matched")
        & (agg.config == config)
    ].sort_values("photons")
    axes[1].plot(t.photons, t.rmse, "o-", color=c, label=lab)
axes[1].set_title("Reconstruction: same one-bit allocations")
style(axes[1], "Mean held-out RMSE")
axes[1].legend(frameon=False)
for ax, l in zip(axes, "ab"):
    label(ax, l)
fig.tight_layout()
save(fig, "fig05_one_bit")

effects = pd.read_csv(D / "paper_paired_effects.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for j, (ax, mode) in enumerate(zip(axes, ["router", "filter"])):
    t = effects[
        (effects.optics == mode)
        & (effects.candidate == "joint_selection")
        & (effects.reference == "selected_uniform")
    ].sort_values("photons")
    ax.errorbar(
        t.photons,
        t.relative_gain_pct,
        yerr=[
            t.relative_gain_pct - t.relative_ci_low,
            t.relative_ci_high - t.relative_gain_pct,
        ],
        fmt="o-",
        capsize=2,
        color="#20867D",
    )
    ax.axhline(0, color="#485565", lw=0.7)
    style(ax, "RMSE reduction (%)" if j == 0 else None)
    ax.set_title(mode.title())
    label(ax, "ab"[j])
fig.tight_layout()
save(fig, "fig06_paired_gains")

# Main sweep robustness, regenerated at print size.
oldwin = pd.read_csv(OLD / "descriptive_global_winners.csv")
configs = [
    ("main", "Baseline"),
    ("headroom_1", "Headroom 1.0"),
    ("headroom_2", "Headroom 2.0"),
    ("read_2e", "Read noise 2 e⁻"),
    ("sensor_stress", "Combined stress"),
    ("crosstalk_15pct", "15% crosstalk"),
    ("declining_throughput", "Declining throughput"),
    ("filter_transmission_80pct", "Filter transmission 0.8"),
    ("fractional_bands", "Equal-width bands"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, mode in zip(axes, ["router", "filter"]):
    z = np.zeros((len(configs), 9))
    texts = []
    for r, (config, _) in enumerate(configs):
        for col, n in enumerate(PHOTONS):
            w = oldwin[
                (oldwin.config == config)
                & (oldwin.decoder == "matched")
                & (oldwin.optics == mode)
                & (oldwin.photons == n)
            ].iloc[0]
            z[r, col] = [3, 4, 6, 8, 12, 24].index(int(w.channels))
            texts.append((col, r, int(w.channels), int(w.bits)))
    ax.imshow(z, aspect="auto", cmap=cmap, vmin=-0.5, vmax=5.5)
    for col, r, k, b in texts:
        ax.text(
            col,
            r,
            f"{k}×{b}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if k >= 12 else "#172A3A",
        )
    ax.set_xticks(
        range(9), ["10", "30", "100", "300", "1k", "3k", "10k", "30k", "100k"]
    )
    ax.set_yticks(range(len(configs)), [r[1] for r in configs], fontsize=7)
    ax.set_xlabel("White-reference exposure, N")
    ax.set_title(mode.title())
fig.tight_layout()
save(fig, "figS01_sensitivity")

sur = pd.read_csv(OLD / "surrogate_rule_validation.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
for j, (ax, mode) in enumerate(zip(axes, ["router", "filter"])):
    t = sur[sur.optics == mode].groupby("photons").mean(numeric_only=True)
    ax.plot(t.index, t.rmse, "o-", color="#345995", label="Three-anchor rule")
    ax.plot(t.index, t.fixed3_rmse, "--", color="#7F8790", label="Fixed 3 × 8")
    d = summary[
        (summary.optics == mode) & (summary.strategy == "selected_uniform")
    ].sort_values("photons")
    ax.plot(d.photons, d.rmse, "s-", color="#20867D", label="Full validation sweep")
    ax.set_yscale("log")
    style(ax, "Mean held-out RMSE" if j == 0 else None)
    ax.set_title(mode.title())
    for n in [30, 1000, 30000]:
        ax.axvline(n, ls=":", color="#7F8790", lw=0.6, alpha=0.5)
axes[1].legend(frameon=False)
fig.tight_layout()
save(fig, "figS02_surrogate")
print("Created vector and PNG figures.")
