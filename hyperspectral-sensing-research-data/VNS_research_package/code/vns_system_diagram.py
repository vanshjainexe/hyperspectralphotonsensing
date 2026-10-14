from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path("VNS_research_package/figures")
out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6))
ax.set(xlim=(0, 12), ylim=(0, 6))
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=.035",
            facecolor=color,
            edgecolor="none",
            alpha=0.13,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#163e50",
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=9,
        color="#253746",
    )


def arrow(x, y, xx, yy):
    ax.annotate(
        "",
        (xx, yy),
        (x, y),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#536b76"},
    )


box(
    0.2, 3.6, 2, 1.5, "Spectrum", "31 CAVE bands\nWhite-reference exposure N", "#3182bd"
)
box(2.8, 4.35, 2.0, 1.2, "Router", "Wavelength separation\nThroughput 84%", "#169c92")
box(
    2.8,
    2.5,
    2.0,
    1.2,
    "Filter array",
    "Spectral selection\nArea factor 1 / K",
    "#e78a39",
)
box(
    5.5,
    3.6,
    1.8,
    1.5,
    "Detection",
    "Random photon counts\nOptional read noise",
    "#8265a6",
)
box(7.95, 3.6, 1.6, 1.5, "ADC", "B bits per output\nK x B = budget", "#bd5c68")
box(10.2, 3.6, 1.6, 1.5, "Decoder", "Estimated\n31-band spectrum", "#336775")
arrow(2.2, 4.4, 2.8, 4.95)
arrow(2.2, 4.1, 2.8, 3.15)
arrow(4.8, 4.95, 5.5, 4.4)
arrow(4.8, 3.15, 5.5, 4.1)
arrow(7.3, 4.35, 7.95, 4.35)
arrow(9.55, 4.35, 10.2, 4.35)
box(
    0.8,
    0.25,
    3,
    1.25,
    "11 training scenes",
    "Fit the spectral decoder\nusing simulated measurements",
    "#2d8fa6",
)
box(
    4.5,
    0.25,
    3,
    1.25,
    "3 validation scenes",
    "Choose regularization\nand channel/bit allocation",
    "#3e9b78",
)
box(
    8.2,
    0.25,
    3,
    1.25,
    "1 separate test scene",
    "Measure reconstruction error\nRepeat across all 15 scenes",
    "#a27844",
)
arrow(3.8, 0.9, 4.5, 0.9)
arrow(7.5, 0.9, 8.2, 0.9)
ax.text(
    6,
    5.7,
    "Joint optical, digital and reconstruction design",
    ha="center",
    fontsize=16,
    fontweight="bold",
    color="#133b4f",
)
ax.text(
    6,
    2.03,
    "Keep exposure and macro-pixel area fixed; do not choose the design using its test result.",
    ha="center",
    fontsize=10,
    color="#4b6572",
)
fig.tight_layout()
for ext in ["png", "svg"]:
    fig.savefig(out / ("00_forward_model." + ext), dpi=200, bbox_inches="tight")
plt.close(fig)
