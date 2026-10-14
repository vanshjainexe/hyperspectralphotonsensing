"""Fixed photon/bit-budget sweep for routed snapshot spectral sensing.

Research question
-----------------
For a fixed 24-bit output budget per *spatial macro-pixel*, should the sensor
use many low-bit spectral channels or fewer high-bit channels?

The code compares:
  1. ideal_router: the full macro-pixel aperture is used and wavelengths are
     routed to K detector outputs (no absorptive area penalty);
  2. absorptive_filter: K equal-area detector sites each transmit only its own
     passband, giving an additional 1/K area penalty.

For a flat unit-reflectance spectrum and N photons incident on the macro-pixel,
the router therefore receives approximately N/K photons per channel, while the
filter receives approximately N/K**2. This is a super-pixel comparison. It is
not a claim that a conventional CFA loses 1/K at each same-size physical pixel.

Pipeline: CAVE reflectance cube -> K boxcar spectral measurements -> Poisson
shot noise -> B-bit uniform ADC -> affine ridge reconstruction to 31 bands.

For a defensible final paper, use scene-held-out training. A single-cube run is
labelled as a within-scene pilot and must not be reported as final validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_PAIRS = ((3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (24, 1))
DEFAULT_PHOTONS = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)


@dataclass(frozen=True)
class Result:
    optics: str
    photons: int
    channels: int
    bits: int
    trial: int
    spectral_rmse: float
    sam_deg: float
    psnr_db: float
    saturation_fraction: float


def synthetic_cube(height: int = 128, width: int = 128, bands: int = 31) -> np.ndarray:
    """Low-rank visible-reflectance-like cube for a dependency/sanity check."""
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    x, y = xx / width, yy / height
    lam = np.linspace(400.0, 700.0, bands)
    basis = np.stack(
        [
            np.ones(bands),
            np.exp(-0.5 * ((lam - 455.0) / 34.0) ** 2),
            np.exp(-0.5 * ((lam - 545.0) / 48.0) ** 2),
            np.exp(-0.5 * ((lam - 635.0) / 43.0) ** 2),
            (lam - 400.0) / 300.0,
        ],
        axis=0,
    )
    coeff = np.stack(
        [
            0.10 + 0.18 * x,
            0.06 + 0.55 * (((x - 0.25) ** 2 + (y - 0.30) ** 2) < 0.12),
            0.08 + 0.48 * (((x - 0.70) ** 2 + (y - 0.31) ** 2) < 0.17),
            0.05 + 0.60 * (((x - 0.53) ** 2 + (y - 0.70) ** 2) < 0.20),
            0.04 + 0.12 * np.sin(2 * np.pi * (3 * x + y)),
        ],
        axis=-1,
    )
    cube = np.einsum("hwq,ql->hwl", coeff, basis)
    texture = 0.82 + 0.18 * np.sin(2 * np.pi * (11 * x + 4 * y))
    return np.clip(cube * texture[..., None], 0.0, 1.0).astype(np.float32)


def _normalize_cube(cube: np.ndarray) -> np.ndarray:
    original_dtype = cube.dtype
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected HxWx31 cube, got shape {cube.shape}.")
    if cube.shape[-1] != 31 and cube.shape[0] == 31:
        cube = np.moveaxis(cube, 0, -1)
    if cube.shape[-1] != 31:
        raise ValueError(f"Expected 31 spectral bands, got shape {cube.shape}.")
    if np.issubdtype(original_dtype, np.integer):
        cube = cube.astype(np.float64) / np.iinfo(original_dtype).max
    else:
        cube = cube.astype(np.float64)
        finite_max = float(np.nanmax(cube))
        if finite_max > 1.0 + 1e-6:
            if finite_max <= 65535:
                cube /= 65535.0
            else:
                raise ValueError("Floating cube is not normalized to [0,1].")
    if not np.all(np.isfinite(cube)):
        raise ValueError("Cube contains NaN or infinite values.")
    return np.clip(cube, 0.0, 1.0).astype(np.float32)


def load_cube(path: Path) -> np.ndarray:
    """Load .npy/.npz/.mat or one extracted CAVE scene directory."""
    path = path.expanduser().resolve()
    if path.is_dir():
        pngs = sorted(
            path.rglob("*_ms_*.png"),
            key=lambda p: (
                int(re.search(r"_ms_(\d+)\.png$", p.name).group(1))
                if re.search(r"_ms_(\d+)\.png$", p.name)
                else 10_000
            ),
        )
        if len(pngs) != 31:
            raise ValueError(
                f"Expected 31 CAVE band PNGs in {path}; found {len(pngs)}."
            )
        cube = np.stack([np.asarray(Image.open(p)) for p in pngs], axis=-1)
        return _normalize_cube(cube)
    if path.suffix.lower() == ".npy":
        return _normalize_cube(np.load(path))
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        arrays = [archive[k] for k in archive.files if np.asarray(archive[k]).ndim == 3]
        if not arrays:
            raise ValueError(f"No 3-D array found in {path}.")
        return _normalize_cube(arrays[0])
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        data = loadmat(path)
        arrays = [
            v
            for k, v in data.items()
            if not k.startswith("__") and np.asarray(v).ndim == 3
        ]
        if not arrays:
            raise ValueError(f"No 3-D array found in {path}.")
        arrays.sort(key=lambda a: np.prod(a.shape), reverse=True)
        return _normalize_cube(arrays[0])
    raise ValueError(f"Unsupported cube path: {path}")


def discover_cave_scenes(root: Path) -> list[Path]:
    candidates: dict[Path, int] = {}
    for png in root.expanduser().resolve().rglob("*_ms_*.png"):
        candidates[png.parent] = candidates.get(png.parent, 0) + 1
    scenes = sorted(path for path, count in candidates.items() if count == 31)
    if not scenes:
        raise ValueError(f"No extracted CAVE scene directories found below {root}.")
    return scenes


def sample_rows(cube: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1])
    if count <= 0 or count >= len(flat):
        return flat.astype(np.float64)
    idx = rng.choice(len(flat), size=count, replace=False)
    return flat[idx].astype(np.float64)


def spectral_operators(k: int, bands: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Return channel-average matrix S and photon-fraction matrix P.

    S @ spectrum gives a mean reflectance in each contiguous wavelength bin.
    P @ spectrum gives the fraction of full-spectrum incident photons entering
    that bin under a uniform per-band illumination convention.
    """
    groups = np.array_split(np.arange(bands), k)
    s = np.zeros((k, bands), dtype=np.float64)
    p = np.zeros((k, bands), dtype=np.float64)
    for row, group in enumerate(groups):
        s[row, group] = 1.0 / len(group)
        p[row, group] = 1.0 / bands
    return s, p


def fit_affine_ridge(
    measurements: np.ndarray, spectra: np.ndarray, ridge: float
) -> np.ndarray:
    design = np.concatenate([measurements, np.ones((len(measurements), 1))], axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ spectra)


def reconstruct(measurements: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    design = np.concatenate([measurements, np.ones((len(measurements), 1))], axis=1)
    return np.clip(design @ decoder, 0.0, 1.0)


def adc_quantize(
    counts: np.ndarray, full_scale: np.ndarray, bits: int
) -> tuple[np.ndarray, float]:
    """Zero-preserving uniform ADC with 2**bits levels from 0 to full scale."""
    levels = 2**bits
    full_scale = np.maximum(full_scale, 1e-12)
    clipped = np.minimum(counts, full_scale)
    saturation = float(np.mean(counts > full_scale))
    codes = np.rint(clipped / full_scale * (levels - 1))
    return codes / (levels - 1) * full_scale, saturation


def metrics(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float, float]:
    error = reference - estimate
    rmse = float(np.sqrt(np.mean(error**2)))
    psnr = float(-20.0 * np.log10(max(rmse, 1e-12)))
    dot = np.sum(reference * estimate, axis=1)
    denom = np.linalg.norm(reference, axis=1) * np.linalg.norm(estimate, axis=1)
    valid = denom > 1e-12
    if np.any(valid):
        cosine = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
        sam = float(np.degrees(np.arccos(cosine)).mean())
    else:
        sam = float("nan")
    return rmse, sam, psnr


def intrinsic_dimension(spectra: np.ndarray) -> dict[str, float]:
    centered = spectra - spectra.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    fractions = eigenvalues / max(eigenvalues.sum(), 1e-15)
    cumulative = np.cumsum(fractions)
    positive = fractions[fractions > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return {
        "pca_components_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca_components_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
        "effective_rank": effective_rank,
    }


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def prepare_data(args: argparse.Namespace, rng: np.random.Generator):
    if args.data_root:
        scenes = discover_cave_scenes(Path(args.data_root))
        if args.test_scene:
            matches = [p for p in scenes if args.test_scene.lower() in str(p).lower()]
            if len(matches) != 1:
                raise ValueError(
                    f"--test-scene matched {len(matches)} scenes; expected exactly one."
                )
            test_path = matches[0]
        else:
            test_path = scenes[0]
        train_paths = [p for p in scenes if p != test_path]
        if not train_paths:
            raise ValueError("Scene-held-out mode requires at least two CAVE scenes.")
        per_scene = max(1, math.ceil(args.train_pixels / len(train_paths)))
        chunks = [sample_rows(load_cube(p), per_scene, rng) for p in train_paths]
        train = np.concatenate(chunks, axis=0)
        if len(train) > args.train_pixels:
            train = train[rng.choice(len(train), args.train_pixels, replace=False)]
        test_cube = load_cube(test_path)
        test = sample_rows(test_cube, args.eval_pixels, rng)
        protocol = "scene-held-out"
        scene_name = (
            test_path.parent.name if test_path.name.endswith(".png") else test_path.name
        )
        return train, test, test_cube, protocol, scene_name

    cube = load_cube(Path(args.cube)) if args.cube else synthetic_cube()
    flat = cube.reshape(-1, 31).astype(np.float64)
    order = rng.permutation(len(flat))
    split = max(1, min(len(flat) - 1, int(0.35 * len(flat))))
    train_pool, test_pool = flat[order[:split]], flat[order[split:]]
    train = train_pool[: args.train_pixels]
    test = test_pool[: args.eval_pixels]
    return (
        train,
        test,
        cube,
        "within-scene-pixel-split-pilot",
        Path(args.cube).stem if args.cube else "synthetic",
    )


def aggregate(results: Sequence[Result]) -> list[dict]:
    buckets: dict[tuple, list[Result]] = {}
    for row in results:
        key = (row.optics, row.photons, row.channels, row.bits)
        buckets.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(buckets.items()):
        optics, photons, channels, bits = key
        summary.append(
            {
                "optics": optics,
                "photons": photons,
                "channels": channels,
                "bits": bits,
                "trials": len(rows),
                "spectral_rmse_mean": float(np.mean([r.spectral_rmse for r in rows])),
                "spectral_rmse_std": (
                    float(np.std([r.spectral_rmse for r in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0
                ),
                "sam_deg_mean": float(np.mean([r.sam_deg for r in rows])),
                "psnr_db_mean": float(np.mean([r.psnr_db for r in rows])),
                "saturation_fraction_mean": float(
                    np.mean([r.saturation_fraction for r in rows])
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    summary: Sequence[dict], clean: Sequence[dict], outdir: Path
) -> list[dict]:
    optics_modes = sorted({row["optics"] for row in summary})
    pairs = sorted({(row["channels"], row["bits"]) for row in summary})
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(pairs)))
    fig, axes = plt.subplots(
        1, len(optics_modes), figsize=(6.2 * len(optics_modes), 4.8), squeeze=False
    )
    for axis, optics in zip(axes[0], optics_modes):
        for color, (k, b) in zip(colors, pairs):
            selected = sorted(
                (
                    row
                    for row in summary
                    if row["optics"] == optics and row["channels"] == k
                ),
                key=lambda row: row["photons"],
            )
            axis.plot(
                [row["photons"] for row in selected],
                [row["spectral_rmse_mean"] for row in selected],
                marker="o",
                color=color,
                label=f"{k}×{b}",
            )
        for row in clean:
            if row["optics"] == optics:
                axis.axhline(
                    row["spectral_rmse"], color="0.7", linewidth=0.6, alpha=0.25
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Incident photons per spatial macro-pixel")
        axis.set_ylabel("Spectral RMSE")
        axis.set_title(optics.replace("_", " ").title())
        axis.grid(True, which="both", alpha=0.22)
        axis.legend(title="K channels × B bits", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "rmse_sweep.png", dpi=190)
    plt.close(fig)

    best_rows = []
    fig, axis = plt.subplots(figsize=(7.2, 4.7))
    markers = ["o", "s", "^", "D"]
    for marker, optics in zip(markers, optics_modes):
        xs, ys = [], []
        photons_values = sorted(
            {row["photons"] for row in summary if row["optics"] == optics}
        )
        for photons in photons_values:
            candidates = [
                row
                for row in summary
                if row["optics"] == optics and row["photons"] == photons
            ]
            winner = min(candidates, key=lambda row: row["spectral_rmse_mean"])
            best_rows.append(winner.copy())
            xs.append(photons)
            ys.append(winner["channels"])
        axis.plot(xs, ys, marker=marker, linewidth=2, label=optics.replace("_", " "))
    axis.set_xscale("log")
    axis.set_yticks([3, 4, 6, 8, 12, 24])
    axis.set_xlabel("Incident photons per spatial macro-pixel")
    axis.set_ylabel("RMSE-optimal spectral channel count K")
    axis.set_title("Best 24-bit allocation by light level")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(outdir / "best_allocation.png", dpi=190)
    plt.close(fig)
    return best_rows


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    base_rng = np.random.default_rng(args.seed)
    train, test, _test_cube, protocol, scene_name = prepare_data(args, base_rng)
    photons_values = parse_ints(args.photons)
    pairs = DEFAULT_PAIRS
    if any(k * b != args.fixed_bits for k, b in pairs):
        raise AssertionError("Candidate pairs do not match fixed bit budget.")

    train_id_stats = intrinsic_dimension(train)
    test_id_stats = intrinsic_dimension(test)
    null_estimate = np.broadcast_to(train.mean(axis=0, keepdims=True), test.shape)
    null_rmse, null_sam, null_psnr = metrics(test, null_estimate)
    decoders: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    clean_rows = []
    for k, b in pairs:
        spectral_average, photon_fraction = spectral_operators(k)
        train_measurements = train @ spectral_average.T
        decoder = fit_affine_ridge(train_measurements, train, args.ridge)
        clean_estimate = reconstruct(test @ spectral_average.T, decoder)
        rmse, sam, psnr = metrics(test, clean_estimate)
        decoders[k] = (spectral_average, photon_fraction, decoder)
        for optics in ("ideal_router", "absorptive_filter"):
            clean_rows.append(
                {
                    "optics": optics,
                    "channels": k,
                    "bits": b,
                    "spectral_rmse": rmse,
                    "sam_deg": sam,
                    "psnr_db": psnr,
                }
            )

    results: list[Result] = []
    decomposition_rows: list[dict] = []
    for optics in ("ideal_router", "absorptive_filter"):
        for photons in photons_values:
            for k, bits in pairs:
                spectral_average, photon_fraction, decoder = decoders[k]
                area_factor = 1.0 if optics == "ideal_router" else 1.0 / k
                throughput = (
                    args.router_efficiency
                    if optics == "ideal_router"
                    else args.filter_transmission
                )
                scale = photons * area_factor * throughput
                clean_count_mean = scale * (test @ photon_fraction.T)
                channel_full_scale = (
                    args.adc_headroom * scale * photon_fraction.sum(axis=1)[None, :]
                )
                inverse_scale = np.maximum(
                    scale * photon_fraction.sum(axis=1)[None, :], 1e-12
                )
                if args.decompose:
                    quantized_clean, saturation = adc_quantize(
                        clean_count_mean, channel_full_scale, bits
                    )
                    quant_only_estimate = reconstruct(
                        quantized_clean / inverse_scale, decoder
                    )
                    rmse, sam, psnr = metrics(test, quant_only_estimate)
                    decomposition_rows.append(
                        {
                            "optics": optics,
                            "photons": photons,
                            "channels": k,
                            "bits": bits,
                            "trial": -1,
                            "component": "quantization_only",
                            "spectral_rmse": rmse,
                            "sam_deg": sam,
                            "psnr_db": psnr,
                            "saturation_fraction": saturation,
                        }
                    )
                for trial in range(args.trials):
                    trial_seed = np.random.SeedSequence(
                        [
                            args.seed,
                            photons,
                            k,
                            trial,
                            0 if optics == "ideal_router" else 1,
                        ]
                    )
                    rng = np.random.default_rng(trial_seed)
                    noisy_counts = rng.poisson(np.maximum(clean_count_mean, 0.0))
                    if args.decompose:
                        shot_only_estimate = reconstruct(
                            noisy_counts / inverse_scale, decoder
                        )
                        shot_rmse, shot_sam, shot_psnr = metrics(
                            test, shot_only_estimate
                        )
                        decomposition_rows.append(
                            {
                                "optics": optics,
                                "photons": photons,
                                "channels": k,
                                "bits": bits,
                                "trial": trial,
                                "component": "shot_noise_only",
                                "spectral_rmse": shot_rmse,
                                "sam_deg": shot_sam,
                                "psnr_db": shot_psnr,
                                "saturation_fraction": 0.0,
                            }
                        )
                    quantized_counts, saturation = adc_quantize(
                        noisy_counts, channel_full_scale, bits
                    )
                    measured_average = quantized_counts / inverse_scale
                    estimate = reconstruct(measured_average, decoder)
                    rmse, sam, psnr = metrics(test, estimate)
                    results.append(
                        Result(
                            optics=optics,
                            photons=photons,
                            channels=k,
                            bits=bits,
                            trial=trial,
                            spectral_rmse=rmse,
                            sam_deg=sam,
                            psnr_db=psnr,
                            saturation_fraction=saturation,
                        )
                    )

    trial_rows = [asdict(row) for row in results]
    summary = aggregate(results)
    best_rows = make_plots(summary, clean_rows, outdir)
    write_csv(outdir / "trial_metrics.csv", trial_rows)
    write_csv(outdir / "summary_metrics.csv", summary)
    write_csv(outdir / "clean_reconstruction_ceiling.csv", clean_rows)
    write_csv(outdir / "best_allocation.csv", best_rows)
    if decomposition_rows:
        write_csv(outdir / "error_decomposition.csv", decomposition_rows)
    write_csv(
        outdir / "null_baseline.csv",
        [
            {
                "scene": scene_name,
                "spectral_rmse": null_rmse,
                "sam_deg": null_sam,
                "psnr_db": null_psnr,
            }
        ],
    )

    assumptions = {
        "scene": scene_name,
        "validation_protocol": protocol,
        "fixed_output_bits_per_spatial_macro_pixel": args.fixed_bits,
        "candidate_allocations": [list(pair) for pair in pairs],
        "incident_photons": list(photons_values),
        "trials": args.trials,
        "train_pixels": len(train),
        "eval_pixels": len(test),
        "adc": "zero-preserving uniform quantizer with per-channel white full-scale",
        "adc_headroom": args.adc_headroom,
        "router_efficiency": args.router_efficiency,
        "filter_passband_transmission": args.filter_transmission,
        "filter_area_factor": "1/K under fixed macro-pixel area",
        "spectral_channels": "contiguous boxcar partitions of 400-700 nm",
        "reconstruction": (
            "scene-held-out affine ridge"
            if protocol == "scene-held-out"
            else "within-scene pilot affine ridge"
        ),
        "ridge": args.ridge,
        "train_intrinsic_dimension": train_id_stats,
        "test_intrinsic_dimension": test_id_stats,
        "null_baseline": {
            "spectral_rmse": null_rmse,
            "sam_deg": null_sam,
            "psnr_db": null_psnr,
        },
        "warning": (
            "Within-scene pilot results are for debugging only; use --data-root for final claims."
            if protocol != "scene-held-out"
            else None
        ),
    }
    (outdir / "run_manifest.json").write_text(json.dumps(assumptions, indent=2))
    print(json.dumps(assumptions, indent=2))
    print(f"Saved results to {outdir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data-root",
        help="Root of extracted CAVE dataset (final scene-held-out mode).",
    )
    source.add_argument(
        "--cube",
        help="One HxWx31 .npy/.npz/.mat cube or extracted CAVE scene directory.",
    )
    parser.add_argument(
        "--test-scene",
        help="Unique substring selecting held-out scene under --data-root.",
    )
    parser.add_argument("--photons", default=",".join(map(str, DEFAULT_PHOTONS)))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--train-pixels", type=int, default=40_000)
    parser.add_argument("--eval-pixels", type=int, default=20_000)
    parser.add_argument("--fixed-bits", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--adc-headroom", type=float, default=1.20)
    parser.add_argument("--router-efficiency", type=float, default=1.0)
    parser.add_argument("--filter-transmission", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Also save shot-only and quantization-only error runs.",
    )
    parser.add_argument("--outdir", default="vns_budget_results")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
