from __future__ import annotations

import csv
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="cave_loso")
parser.add_argument("--out", default="cave_loso_summary")
args = parser.parse_args()

ROOT = Path(args.root)
OUT = Path(args.out)
OUT.mkdir(exist_ok=True)

rows: list[dict] = []
for path in sorted(ROOT.glob("*/summary_metrics.csv")):
    scene = path.parent.name
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "scene": scene,
                    "optics": row["optics"],
                    "photons": int(row["photons"]),
                    "channels": int(row["channels"]),
                    "bits": int(row["bits"]),
                    "rmse": float(row["spectral_rmse_mean"]),
                }
            )

grouped: dict[tuple, list[float]] = defaultdict(list)
for row in rows:
    key = (row["optics"], row["photons"], row["channels"], row["bits"])
    grouped[key].append(row["rmse"])

aggregate = []
for (optics, photons, channels, bits), values in sorted(grouped.items()):
    aggregate.append(
        {
            "optics": optics,
            "photons": photons,
            "channels": channels,
            "bits": bits,
            "scenes": len(values),
            "rmse_scene_mean": float(np.mean(values)),
            "rmse_scene_std": float(np.std(values, ddof=1)),
            "rmse_scene_median": float(np.median(values)),
        }
    )

with (OUT / "cross_scene_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
    writer.writeheader()
    writer.writerows(aggregate)

best = []
for optics in sorted({row["optics"] for row in aggregate}):
    for photons in sorted({row["photons"] for row in aggregate}):
        candidates = [
            row
            for row in aggregate
            if row["optics"] == optics and row["photons"] == photons
        ]
        best.append(min(candidates, key=lambda row: row["rmse_scene_mean"]))

with (OUT / "cross_scene_best.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(best[0]))
    writer.writeheader()
    writer.writerows(best)

winner_counts: dict[tuple[str, int], Counter] = defaultdict(Counter)
for path in sorted(ROOT.glob("*/best_allocation.csv")):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            winner_counts[(row["optics"], int(row["photons"]))][
                (int(row["channels"]), int(row["bits"]))
            ] += 1

count_rows = []
for (optics, photons), counts in sorted(winner_counts.items()):
    for (channels, bits), count in sorted(counts.items()):
        count_rows.append(
            {
                "optics": optics,
                "photons": photons,
                "channels": channels,
                "bits": bits,
                "winning_scenes": count,
                "total_scenes": sum(counts.values()),
            }
        )

with (OUT / "per_scene_winner_counts.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(count_rows[0]))
    writer.writeheader()
    writer.writerows(count_rows)

fig, axis = plt.subplots(figsize=(7.6, 4.8))
markers = {"ideal_router": "o", "absorptive_filter": "s"}
for optics in ("ideal_router", "absorptive_filter"):
    selected = sorted(
        (row for row in best if row["optics"] == optics and row["photons"] >= 30),
        key=lambda row: row["photons"],
    )
    axis.plot(
        [row["photons"] for row in selected],
        [row["channels"] for row in selected],
        marker=markers[optics],
        linewidth=2,
        label=optics.replace("_", " ").title(),
    )
axis.set_xscale("log")
axis.set_yticks([3, 4, 6, 8, 12, 24])
axis.set_xlabel("Incident photons per spatial macro-pixel")
axis.set_ylabel("Lowest-mean-RMSE channel count")
axis.set_title("Cross-scene best 24-bit allocation (15 CAVE scenes, ≥30 photons)")
axis.grid(True, which="both", alpha=0.25)
axis.legend()
fig.tight_layout()
fig.savefig(OUT / "cross_scene_best.png", dpi=200)
