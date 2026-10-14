"""Make numerical (not generative) image/spectrum examples on held-out balloons."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vns_robustness import *

root = Path("vns_robust_outputs")
out = Path("VNS_research_package")
(out / "figures").mkdir(parents=True, exist_ok=True)
(out / "data").mkdir(exist_ok=True)
protocol = json.loads((root / "protocol.json").read_text())
names = protocol["scenes"]
i = names.index("balloons")
cache = np.load(root / "sample_cache.npz")
vals = [(i + j) % 15 for j in [1, 2, 3]]
trains = [j for j in range(15) if j != i and j not in vals]
xtrain = np.concatenate([cache[names[j]][:600] for j in trains])
xval = np.concatenate([cache[names[j]][:1024] for j in vals])
segments = [(j * 1024, (j + 1) * 1024) for j in range(3)]
path = next(
    p
    for p in discover_cave_scenes(Path(protocol["data_root"]))
    if p.name == "balloons_ms"
)
cube = load_cube(path)[::4, ::4]
x = cube.reshape(-1, 31).astype(float)
config = Config("main")
wavelength = np.arange(400, 701, 10)
arrays = {"reference": cube, "wavelength_nm": wavelength}
rows = []
estimates = {}
fig, axes = plt.subplots(3, 5, figsize=(12, 7.8))
rgb = lambda a: np.clip(a[..., [25, 15, 5]] / 0.7, 0, 1) ** (1 / 2.2)
for row, n in enumerate([30, 300, 10000]):
    axes[row, 0].imshow(rgb(cube))
    axes[row, 0].set_title(f"Reference | N={n:,}")
    for col, k in enumerate([3, 4, 6, 24], 1):
        b = 24 // k
        p = operator(k)
        rnd = lambda tag: np.random.default_rng(
            seed(protocol["seed"], "balloons", "router", n, k, tag)
        )
        yt = np.concatenate(
            [
                sense(xtrain, p, n, k, b, "router", config, rnd("train" + str(j)))[0]
                for j in range(2)
            ]
        )
        yv = sense(xval, p, n, k, b, "router", config, rnd("validation"))[0]
        ve, alpha, dec = select_decoder(
            yt, np.tile(xtrain, (2, 1)), yv, xval, segments, "matched"
        )
        y = sense(x, p, n, k, b, "router", config, rnd("full_image"))[0]
        z = predict(y, dec).reshape(cube.shape)
        arrays[f"N{n}_K{k}_B{b}"] = z.astype(np.float32)
        estimates[(n, k)] = z
        m = measure(x, z.reshape(-1, 31))
        rows.append(
            {
                "scene": "balloons",
                "photons": n,
                "channels": k,
                "bits": b,
                "validation_rmse": ve,
                "alpha": alpha,
                **m,
            }
        )
        axes[row, col].imshow(rgb(z))
        axes[row, col].set_title(f'{k} x {b} | RMSE {m["rmse"]:.3f}')
    for ax in axes[row]:
        ax.axis("off")
fig.suptitle(
    "Held-out scene reconstruction | 84% router | matched-noise affine decoder\n650/550/450 nm pseudo-color, fixed display gain/gamma; not colorimetric RGB",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(out / "figures/10_image_reconstructions.png", dpi=180, bbox_inches="tight")
plt.close(fig)
np.savez_compressed(out / "data/image_reconstruction_arrays.npz", **arrays)
pd.DataFrame(rows).to_csv(out / "data/image_example_metrics.csv", index=False)

coords = [(32, 32), (32, 96), (96, 32), (96, 96)]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=True, sharey=True)
spectra = []
for ax, (y, xp) in zip(axes.ravel(), coords):
    ax.plot(wavelength, cube[y, xp], color="black", lw=2, label="Reference")
    for k in [3, 4, 6, 24]:
        ax.plot(
            wavelength, estimates[(10000, k)][y, xp], label=f"{k} x {24//k}", lw=1.3
        )
    for l, wl in enumerate(wavelength):
        spectra.append(
            {
                "image_y": y * 4,
                "image_x": xp * 4,
                "wavelength_nm": int(wl),
                "reference": float(cube[y, xp, l]),
                **{
                    f"K{k}": float(estimates[(10000, k)][y, xp, l])
                    for k in [3, 4, 6, 24]
                },
            }
        )
    ax.set_title(f"Original pixel (row={y*4}, column={xp*4})")
    ax.grid(alpha=0.2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized reflectance")
axes[0, 0].legend(fontsize=8)
fig.suptitle(
    "Four fixed-grid pixels, not selected by reconstruction quality | N=10,000"
)
fig.tight_layout()
fig.savefig(out / "figures/11_example_spectra.png", dpi=190, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(spectra).to_csv(out / "data/example_spectra.csv", index=False)
print("Saved image examples and numerical reconstruction arrays")
