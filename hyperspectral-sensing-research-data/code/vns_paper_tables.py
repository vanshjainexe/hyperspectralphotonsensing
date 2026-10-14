"""Generate every key manuscript number from the final CSV records."""

from pathlib import Path
import pandas as pd

OUT = Path("VNS_paper")
D = OUT / "data"
s = pd.read_csv(D / "paper_strategy_summary.csv")
e = pd.read_csv(D / "paper_paired_effects.csv")
o = pd.read_csv(D / "paper_code_occupancy.csv")
a = pd.read_csv(D / "quantizer_cross_scene_metrics.csv")


def strategy(key, mode="router", n=100000):
    return s[(s.strategy == key) & (s.optics == mode) & (s.photons == n)].iloc[0]


def effect(candidate, reference, mode="router", n=100000):
    return e[
        (e.candidate == candidate)
        & (e.reference == reference)
        & (e.optics == mode)
        & (e.photons == n)
    ].iloc[0]


def onebit(config):
    return a[
        (a.config == config)
        & (a.optics == "router")
        & (a.channels == 24)
        & (a.photons == 100000)
        & (a.decoder == "matched")
    ].iloc[0]


high = effect("joint_selection", "fixed3_uniform")
inc = effect("joint_selection", "selected_uniform")
occ = o[(o.optics == "router") & (o.photons == 100000) & (o.channels == 24)].iloc[0]
values = {
    "JointGainHigh": (high.relative_gain_pct, 1),
    "JointGainLowCI": (high.relative_ci_low, 1),
    "JointGainHighCI": (high.relative_ci_high, 1),
    "IncrementalGainHigh": (inc.relative_gain_pct, 1),
    "IncrementalLowCI": (inc.relative_ci_low, 1),
    "IncrementalHighCI": (inc.relative_ci_high, 1),
    "CalibratedBaselineGain": (
        effect("joint_selection", "fixed3_calibrated").relative_gain_pct,
        1,
    ),
    "FixedHigh": (strategy("fixed3_uniform").rmse, 4),
    "UniformHigh": (strategy("selected_uniform").rmse, 4),
    "JointHigh": (strategy("joint_selection").rmse, 4),
    "FilterJointHigh": (strategy("joint_selection", "filter").rmse, 4),
    "FilterTotalGain": (
        effect("joint_selection", "fixed3_uniform", "filter").relative_gain_pct,
        1,
    ),
    "QuantileEntropy": (occ.mean_code_entropy_bits, 3),
    "UniformEntropy": (occ.uniform_mean_code_entropy_bits, 3),
    "OneBitUniformRMSE": (onebit("main").rmse, 4),
    "OneBitQuantileRMSE": (onebit("quantile_adc").rmse, 4),
    "QuantileSelectedHigh": (strategy("selected_quantile").rmse, 4),
}
(OUT / "numbers.tex").write_text(
    "% Generated from saved results; do not edit numbers manually.\n"
    + "".join(
        "\\newcommand{\\" + k + "}{" + f"{v:.{digits}f}" + "}\n"
        for k, (v, digits) in values.items()
    )
)
table = [
    r"\begin{table*}[t]",
    r"\centering\small",
    r"\caption{Held-out results after validation-only selection. RMSE is in normalized spectral units; SAM is in degrees and PSNR in dB. Entries average three noise trials within each scene and then weight the 15 scenes equally. The joint selector chooses channel count and ADC type.}",
    r"\label{tab:main}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{rlrrrrrr}",
    r"\toprule",
    r"& & \multicolumn{3}{c}{Router} & \multicolumn{3}{c}{Filter} \\",
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
    r"$N$ & Selection procedure & RMSE $\downarrow$ & SAM $\downarrow$ & PSNR $\uparrow$ & RMSE $\downarrow$ & SAM $\downarrow$ & PSNR $\uparrow$ \\",
    r"\midrule",
]
for n in [100, 1000, 10000, 100000]:
    for i, (key, label) in enumerate(
        [
            ("fixed3_uniform", r"Fixed $3\times8$, uniform"),
            ("selected_uniform", r"Choose $K$, uniform"),
            ("joint_selection", r"Choose $K$ and ADC"),
        ]
    ):
        cols = []
        for mode in ["router", "filter"]:
            row = strategy(key, mode, n)
            cols.extend([f"{row.rmse:.4f}", f"{row.sam:.2f}", f"{row.psnr:.2f}"])
        table.append(
            (f"{n:,}" if i == 0 else "")
            + " & "
            + label
            + " & "
            + " & ".join(cols)
            + r" \\"
        )
    if n != 100000:
        table.append(r"\addlinespace[4pt]")
table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
(OUT / "table_main.tex").write_text("\n".join(table) + "\n")

# Complete condition-by-condition effects, including unfavorable results.
tab = [
    r"\begin{table}[htbp]",
    r"\centering\small",
    r"\caption{Incremental gain from joint channel/ADC selection over channel selection with a uniform ADC. Positive values favor joint selection. Paired scene-bootstrap intervals are descriptive.}",
    r"\label{tab:allgains}",
    r"\begin{tabular}{rrrrrrr}",
    r"\toprule",
    r"& \multicolumn{3}{c}{Router} & \multicolumn{3}{c}{Filter} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"$N$ & Gain (\%) & 95\% interval & Scenes improved & Gain (\%) & 95\% interval & Scenes improved \\",
    r"\midrule",
]
for n in [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]:
    cs = []
    for mode in ["router", "filter"]:
        r = effect("joint_selection", "selected_uniform", mode, n)
        cs.extend(
            [
                f"{r.relative_gain_pct:.2f}",
                f"[{r.relative_ci_low:.2f}, {r.relative_ci_high:.2f}]",
                f"{r.scenes_improved}/15",
            ]
        )
    tab.append(f"{n:,}" + " & " + " & ".join(cs) + r" \\")
tab.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
(OUT / "table_gains.tex").write_text("\n".join(tab) + "\n")
print("Manuscript numbers:", {k: round(v, digits) for k, (v, digits) in values.items()})
