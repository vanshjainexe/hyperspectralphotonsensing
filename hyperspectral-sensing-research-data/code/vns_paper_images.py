"""Numerical held-out reconstruction panels for the manuscript appendix."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vns_robustness import *
from vns_quantizer_extension import fit_quantile_adc, apply_adc

out = Path("VNS_paper")
src = Path("vns_robust_outputs")
spec = json.loads((src / "protocol.json").read_text())
fold = next(
    f
    for f in json.loads((src / "folds.json").read_text())
    if f["test_scene"] == "balloons"
)
cache = np.load(src / "sample_cache.npz")
xtr = np.concatenate([cache[n][:600] for n in fold["training_scenes"]])
xv = np.concatenate([cache[n][:1024] for n in fold["validation_scenes"]])
xx = np.tile(xtr, (2, 1))
segments = [(j * 1024, (j + 1) * 1024) for j in range(3)]
path = next(
    p for p in discover_cave_scenes(Path(spec["data_root"])) if p.name == "balloons_ms"
)
cube = load_cube(path)[::4, ::4]
xt = cube.reshape(-1, 31).astype(float)
arrays = {"reference": cube}
rows = []
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.5), layout="constrained")
rgb = lambda x: np.clip(x[..., [25, 15, 5]] / 0.7, 0, 1) ** (1 / 2.2)
designs = [(3, "uniform"), (6, "uniform"), (6, "sqrt"), (24, "quantile")]
for ri, n in enumerate([300, 10000]):
    axes[ri, 0].imshow(rgb(cube))
    axes[ri, 0].set_title(f"Reference\nN = {n:,}", fontsize=7)
    for ci, (k, kind) in enumerate(designs, 1):
        b = 24 // k
        p = operator(k)
        cfg = Config("main", quantizer="sqrt" if kind == "sqrt" else "uniform")
        rng = lambda tag: np.random.default_rng(
            seed(spec["seed"], "balloons", "router", n, k, tag)
        )
        if kind == "quantile":
            rawtr = np.concatenate(
                [
                    sense(
                        xtr, p, n, k, b, "router", cfg, rng("train" + str(j)), adc=False
                    )[0]
                    for j in range(2)
                ]
            )
            adc = fit_quantile_adc(rawtr, b)
            yt, _ = apply_adc(rawtr, adc)
            yv, _ = apply_adc(
                sense(xv, p, n, k, b, "router", cfg, rng("validation"), adc=False)[0],
                adc,
            )
            test, _ = apply_adc(
                sense(xt, p, n, k, b, "router", cfg, rng("full_image"), adc=False)[0],
                adc,
            )
        else:
            yt = np.concatenate(
                [
                    sense(xtr, p, n, k, b, "router", cfg, rng("train" + str(j)))[0]
                    for j in range(2)
                ]
            )
            yv = sense(xv, p, n, k, b, "router", cfg, rng("validation"))[0]
            test = sense(xt, p, n, k, b, "router", cfg, rng("full_image"))[0]
        verr, alpha, dec = select_decoder(yt, xx, yv, xv, segments, "matched")
        z = predict(test, dec)
        m = measure(xt, z)
        zz = z.reshape(cube.shape)
        arrays[f"N{n}_K{k}_{kind}"] = zz.astype(np.float32)
        axes[ri, ci].imshow(rgb(zz))
        label = {"uniform": "Uniform", "sqrt": "Square-root", "quantile": "Quantile"}[
            kind
        ]
        axes[ri, ci].set_title(f'{k} × {b} · {label}\nRMSE {m["rmse"]:.3f}', fontsize=7)
        rows.append(
            {
                "photons": n,
                "channels": k,
                "bits": b,
                "adc": kind,
                "alpha": alpha,
                "validation_rmse": verr,
                **m,
            }
        )
    for ax in axes[ri]:
        ax.axis("off")
fig.savefig(out / "figures/figS03_reconstructions.pdf")
fig.savefig(out / "figures/figS03_reconstructions.png", dpi=240)
np.savez_compressed(out / "data/paper_image_cubes.npz", **arrays)
pd.DataFrame(rows).to_csv(out / "data/paper_image_metrics.csv", index=False)
print("Saved held-out numerical image examples.")
