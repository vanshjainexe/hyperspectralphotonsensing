"""Reproducible, nested scene-held-out robustness study (v2).

Run: OPENBLAS_NUM_THREADS=1 python vns_robustness.py --data-root PATH
No test scene is used for decoder fitting, ridge selection or allocation choice.
N means white-reference photon exposure, not identical detected photons at
every scene pixel. ADC gain is calibrated separately at every N/optics/K.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from vns_budget_sweep import (
    discover_cave_scenes,
    load_cube,
    spectral_operators,
    intrinsic_dimension,
)

PAIRS = {
    12: [(3, 4), (4, 3), (6, 2), (12, 1)],
    24: [(3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (24, 1)],
    48: [(3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2)],
}
PHOTONS = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
ALPHAS = [1e-4, 1e-2, 1.0, 100.0]


@dataclass(frozen=True)
class Config:
    name: str
    budget: int = 24
    headroom: float = 1.2
    quantizer: str = "uniform"
    read_noise: float = 0.0
    dark: float = 0.0
    crosstalk: float = 0.0
    efficiency: float = 0.84
    declining_efficiency: bool = False
    filter_transmission: float = 1.0
    fractional_bands: bool = False


CONFIGS = [
    Config("main"),
    Config("headroom_1", headroom=1.0),
    Config("headroom_2", headroom=2.0),
    Config("sqrt_adc", quantizer="sqrt"),
    Config("read_2e", read_noise=2.0),
    Config("sensor_stress", read_noise=5.0, dark=0.2, crosstalk=0.15),
    Config("crosstalk_15pct", crosstalk=0.15),
    Config("declining_throughput", declining_efficiency=True),
    Config("filter_transmission_80pct", filter_transmission=0.8),
    Config("fractional_bands", fractional_bands=True),
    Config("budget_12", budget=12),
    Config("budget_48", budget=48),
]


def seed(*values):
    # Stable across Python processes and machines (unlike built-in hash).
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, values)).encode()).digest()[:8], "little"
    )


def write_rows(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def operator(k, fractional=False):
    if not fractional:
        return spectral_operators(k)[1]
    # 31 equal-width source bins, fractional overlap with K equal-width outputs.
    source = np.arange(32, dtype=float) / 31
    dest = np.arange(k + 1, dtype=float) / k
    return np.maximum(
        0.0,
        np.minimum(dest[1:, None], source[None, 1:])
        - np.maximum(dest[:-1, None], source[None, :-1]),
    )


def mixing(k, fraction):
    m = np.eye(k) * (1 - fraction)
    for j in range(k):
        neighbors = [i for i in [j - 1, j + 1] if 0 <= i < k]
        for i in neighbors:
            m[i, j] += fraction / len(neighbors)
    return m


def quantize(counts, fullscale, bits, kind="uniform"):
    z = np.clip(counts / fullscale, 0, 1)
    encoded = np.sqrt(z) if kind == "sqrt" else z
    encoded = np.rint(encoded * (2**bits - 1)) / (2**bits - 1)
    return (encoded**2 if kind == "sqrt" else encoded) * fullscale


def sense(x, p, n, k, b, mode, config, rng, poisson=True, adc=True):
    eta = config.efficiency
    if config.declining_efficiency:
        eta = (
            0.84 - 0.24 * (k - 3) / 21
        )  # declared hypothetical 0.84 -> 0.60, not measured
    throughput = eta if mode == "router" else config.filter_transmission / k
    q = p if mode == "filter" else mixing(k, config.crosstalk) @ p
    scale = n * throughput
    white = scale * q.sum(axis=1)
    means = scale * (x @ q.T)
    counts = (
        rng.poisson(means + config.dark).astype(float) - config.dark
        if poisson
        else means.copy()
    )
    if poisson and config.read_noise:
        counts += rng.normal(0, config.read_noise, counts.shape)
    saturation = float(np.mean(counts > config.headroom * white))
    zero_fraction = float(np.mean(counts <= 0))
    counts = np.maximum(counts, 0.0)
    if adc:
        counts = quantize(counts, config.headroom * white, b, config.quantizer)
    return counts / white, saturation, zero_fraction, float(means.mean() * k)


def features(y, kind):
    if kind == "quadratic":
        return np.concatenate([y, y * y, y[:, :-1] * y[:, 1:]], axis=1)
    return y


def fit(y, x, alpha, kind="matched"):
    z = features(y, kind)
    mu = z.mean(0)
    sd = z.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    z = (z - mu) / sd
    xm = x.mean(0)
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (x - xm))
    return (mu, sd, xm, w, kind)


def predict(y, decoder):
    mu, sd, xm, w, kind = decoder
    return np.clip((features(y, kind) - mu) / sd @ w + xm, 0, 1)


def measure(x, z):
    mse = np.mean((x - z) ** 2)
    rmse = float(np.sqrt(mse))
    xn = np.linalg.norm(x, axis=1)
    zn = np.linalg.norm(z, axis=1)
    nonblack = xn > 1e-10
    angle = np.full(len(x), 90.0)
    valid = nonblack & (zn > 1e-10)
    angle[valid] = np.degrees(
        np.arccos(
            np.clip(
                np.sum(x[valid] * z[valid], axis=1) / (xn[valid] * zn[valid]), -1, 1
            )
        )
    )
    # Zero predictions of nonzero references count as 90 deg, not silently omitted.
    return {
        "rmse": rmse,
        "psnr": float(-20 * np.log10(max(rmse, 1e-12))),
        "sam": float(angle[nonblack].mean()),
        "sam_pixels": int(nonblack.sum()),
        "zero_prediction_fraction": float(np.mean(zn < 1e-10)),
    }


def val_rmse(x, z, segments):
    return float(
        np.mean([np.sqrt(np.mean((x[a:b] - z[a:b]) ** 2)) for a, b in segments])
    )


def select_decoder(ytrain, xtrain, yval, xval, segments, kind):
    best = None
    for alpha in ALPHAS:
        dec = fit(ytrain, xtrain, alpha, kind)
        err = val_rmse(xval, predict(yval, dec), segments)
        if best is None or err < best[0]:
            best = (err, alpha, dec)
    return best


def interpolation(y, p):
    centers = (p @ np.arange(31)) / p.sum(1)
    weights = np.stack(
        [np.interp(np.arange(31), centers, np.eye(len(p))[i]) for i in range(len(p))]
    )
    return np.clip(y @ weights, 0, 1)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "sample_cache.npz"
    paths = discover_cave_scenes(Path(args.data_root))
    names = [p.name.removesuffix("_ms") for p in paths]
    if cache.exists():
        packed = np.load(cache)
        samples = [packed[n] for n in names]
    else:
        samples = []
        for i, p in enumerate(paths):
            cube = load_cube(p)
            flat = cube.reshape(-1, 31)
            ids = np.random.default_rng(seed(args.seed, names[i], "pixels")).choice(
                len(flat), args.test_pixels, replace=False
            )
            samples.append(flat[ids].astype(float))
            print("Loaded", names[i], cube.shape, flush=True)
        np.savez_compressed(cache, **dict(zip(names, samples)))
    selected = [
        c for c in CONFIGS if not args.configs or c.name in args.configs.split(",")
    ]
    specs = {
        "version": 2,
        "seed": args.seed,
        "configs": [asdict(c) for c in selected],
        "photons": PHOTONS,
        "pairs": PAIRS,
        "trials": args.trials,
        "train_pixels_per_scene": args.train_per_scene,
        "validation_pixels_per_scene": args.val_per_scene,
        "test_pixels": args.test_pixels,
        "train_noise_copies": 2,
        "alphas": ALPHAS,
        "data_root": str(Path(args.data_root).resolve()),
        "scenes": names,
        "N_definition": "white-reference photon exposure; actual per-pixel photon counts vary with normalized spectrum",
        "ADC": "white-reference fullscale recalibrated for every exposure, optics and K",
        "validation": "outer held-out scene; next 3 scenes validate alpha and K; remaining 11 train",
        "SAM": "exclude zero reference; assign 90deg when nonzero reference has zero prediction",
    }
    (out / "protocol.json").write_text(json.dumps(specs, indent=2))
    folds = []
    t0 = time.time()
    for i, name in enumerate(names):
        vals = [(i + j) % len(names) for j in [1, 2, 3]]
        trains = [j for j in range(len(names)) if j != i and j not in vals]
        folds.append(
            {
                "test_scene": name,
                "validation_scenes": [names[j] for j in vals],
                "training_scenes": [names[j] for j in trains],
            }
        )
        xtrain = np.concatenate([samples[j][: args.train_per_scene] for j in trains])
        xval = np.concatenate([samples[j][: args.val_per_scene] for j in vals])
        segments = [
            (j * args.val_per_scene, (j + 1) * args.val_per_scene) for j in range(3)
        ]
        xtest = samples[i]
        null = measure(xtest, np.broadcast_to(xtrain.mean(0), xtest.shape))
        sceneout = out / name
        sceneout.mkdir(exist_ok=True)
        (sceneout / "scene.json").write_text(
            json.dumps(
                {"null": null, "dimension": intrinsic_dimension(xtest)}, indent=2
            )
        )
        for config in selected:
            dest = sceneout / (config.name + ".csv")
            if dest.exists():
                print("Reuse", name, config.name, flush=True)
                continue
            rows = []
            for mode in ["router", "filter"]:
                for n in PHOTONS:
                    for k, b in PAIRS[config.budget]:
                        p = operator(k, config.fractional_bands)
                        rnd = lambda tag: np.random.default_rng(
                            seed(args.seed, name, mode, n, k, tag)
                        )
                        # Same streams across configurations where dimensions match.
                        ytrain = np.concatenate(
                            [
                                sense(
                                    xtrain,
                                    p,
                                    n,
                                    k,
                                    b,
                                    mode,
                                    config,
                                    rnd("train" + str(j)),
                                )[0]
                                for j in range(2)
                            ]
                        )
                        xxtrain = np.tile(xtrain, (2, 1))
                        yval = sense(xval, p, n, k, b, mode, config, rnd("validation"))[
                            0
                        ]
                        kinds = (
                            ["matched", "clean", "interpolation", "quadratic"]
                            if config.name == "main"
                            else ["matched"]
                        )
                        decoders = {}
                        for kind in kinds:
                            if kind in ["matched", "quadratic"]:
                                verr, alpha, dec = select_decoder(
                                    ytrain, xxtrain, yval, xval, segments, kind
                                )
                            elif kind == "clean":
                                yclean = sense(
                                    xtrain,
                                    p,
                                    n,
                                    k,
                                    b,
                                    mode,
                                    config,
                                    rnd("clean"),
                                    False,
                                    False,
                                )[0]
                                dec = fit(yclean, xtrain, 1e-5)
                                alpha = 1e-5
                                verr = val_rmse(xval, predict(yval, dec), segments)
                            else:
                                dec = None
                                alpha = 0.0
                                verr = val_rmse(xval, interpolation(yval, p), segments)
                            decoders[kind] = (verr, alpha, dec)
                        for trial in range(args.trials):
                            y, sat, zero, expected = sense(
                                xtest,
                                p,
                                n,
                                k,
                                b,
                                mode,
                                config,
                                rnd("test" + str(trial)),
                            )
                            for kind, (verr, alpha, dec) in decoders.items():
                                estimate = (
                                    interpolation(y, p)
                                    if kind == "interpolation"
                                    else predict(y, dec)
                                )
                                rows.append(
                                    {
                                        "scene": name,
                                        "config": config.name,
                                        "budget": config.budget,
                                        "optics": mode,
                                        "photons": n,
                                        "channels": k,
                                        "bits": b,
                                        "decoder": kind,
                                        "trial": trial,
                                        "validation_rmse": verr,
                                        "alpha": alpha,
                                        **measure(xtest, estimate),
                                        "saturation_fraction": sat,
                                        "pre_adc_zero_fraction": zero,
                                        "expected_detected_photons": expected,
                                        "null_rmse": null["rmse"],
                                    }
                                )
            write_rows(dest, rows)
            print(
                f"{name}: {config.name}: {len(rows)} rows; elapsed {time.time()-t0:.1f}s",
                flush=True,
            )
    (out / "folds.json").write_text(json.dumps(folds, indent=2))
    print("Finished robustness sweep", flush=True)


def parser():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--data-root", required=True)
    a.add_argument("--out", default="vns_robust_outputs")
    a.add_argument("--seed", type=int, default=20261013)
    a.add_argument("--trials", type=int, default=3)
    a.add_argument("--train-per-scene", type=int, default=600)
    a.add_argument("--val-per-scene", type=int, default=1024)
    a.add_argument("--test-pixels", type=int, default=4096)
    a.add_argument("--configs", default="")
    return a


if __name__ == "__main__":
    run(parser().parse_args())
