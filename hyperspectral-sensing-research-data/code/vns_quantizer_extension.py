"""Training-only empirical quantile ADC control, specified after the v2 sweep.

Keeps v2 scene folds, pixels, noise streams, ridge grid and 24-bit allocations.
Thresholds and reconstruction levels are fitted only to noisy training counts.
No test or validation spectrum is used to fit an ADC. This is an exploratory
control, not a claim of a newly invented scalar quantizer.
"""

from pathlib import Path
import argparse, json, time
import numpy as np
from vns_robustness import (
    Config,
    PHOTONS,
    PAIRS,
    seed,
    operator,
    sense,
    select_decoder,
    predict,
    measure,
    write_rows,
)


def fit_quantile_adc(y, bits):
    levels = 2**bits
    result = []
    for col in y.T:
        edges = np.unique(np.quantile(col, np.arange(1, levels) / levels))
        # Ties go to the lower cell: a zero median must distinguish zero
        # counts from positive counts, not collapse both into the upper cell.
        ids = np.searchsorted(edges, col, side="left")
        count = np.bincount(ids, minlength=len(edges) + 1)
        sums = np.bincount(ids, weights=col, minlength=len(edges) + 1)
        # Empty cells occur at atoms (especially zero Poisson counts).
        fallback = (np.r_[col.min(), edges] + np.r_[edges, col.max()]) / 2
        centers = np.divide(sums, count, out=fallback, where=count > 0)
        assert len(centers) <= levels and np.all(np.diff(centers) >= -1e-12)
        result.append((edges, centers))
    return result


def apply_adc(y, adc):
    ids = np.column_stack(
        [np.searchsorted(e, y[:, j], side="left") for j, (e, c) in enumerate(adc)]
    )
    out = np.column_stack([c[ids[:, j]] for j, (e, c) in enumerate(adc)])
    return out, ids


def occupancy(ids):
    ent = []
    dominance = []
    used = []
    for col in ids.T:
        _, counts = np.unique(col, return_counts=True)
        p = counts / len(col)
        ent.append(-np.sum(p * np.log2(p)))
        dominance.append(p.max())
        used.append(len(counts))
    return {
        "mean_code_entropy_bits": float(np.mean(ent)),
        "mean_dominant_code_fraction": float(np.mean(dominance)),
        "mean_used_codes": float(np.mean(used)),
    }


def run(args):
    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = json.loads((source / "protocol.json").read_text())
    folds = json.loads((source / "folds.json").read_text())
    samples = np.load(source / "sample_cache.npz")
    base = Config("main")
    t0 = time.time()
    protocol = {
        "purpose": "post-v2 exploratory quantizer-calibration control",
        "source_protocol": specs,
        "ADC": "empirical quantiles of two noisy training copies; conditional sample-mean levels; duplicate thresholds merged; ties assigned lower cell; outer bins unbounded",
        "decoders": ["matched", "quadratic"],
        "test_or_validation_ADC_fit": False,
        "budget": 24,
        "photons": PHOTONS,
        "common_random_numbers": "same seeds as v2 main",
        "calibration_storage": "shared per configuration; excluded from per-pixel output budget",
        "physical_status": "numerical control; nonlinear ADC implementation and circuit costs not modeled",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for fold in folds:
        name = fold["test_scene"]
        dest = out / (name + ".csv")
        if dest.exists():
            continue
        xtr = np.concatenate([samples[n][:600] for n in fold["training_scenes"]])
        xv = np.concatenate([samples[n][:1024] for n in fold["validation_scenes"]])
        xt = samples[name]
        xx = np.tile(xtr, (2, 1))
        segments = [(j * 1024, (j + 1) * 1024) for j in range(3)]
        rows = []
        calibrations = {}
        for mode in ["router", "filter"]:
            for n in PHOTONS:
                for k, b in PAIRS[24]:
                    p = operator(k)
                    rng = lambda tag: np.random.default_rng(
                        seed(specs["seed"], name, mode, n, k, tag)
                    )
                    rawtr = np.concatenate(
                        [
                            sense(
                                xtr,
                                p,
                                n,
                                k,
                                b,
                                mode,
                                base,
                                rng("train" + str(j)),
                                adc=False,
                            )[0]
                            for j in range(2)
                        ]
                    )
                    rawv = sense(
                        xv, p, n, k, b, mode, base, rng("validation"), adc=False
                    )[0]
                    adc = fit_quantile_adc(rawtr, b)
                    yt, _ = apply_adc(rawtr, adc)
                    yv, _ = apply_adc(rawv, adc)
                    for j, (edges, centers) in enumerate(adc):
                        tag = f"{mode}_{n}_{k}_{j}"
                        calibrations[tag + "_thresholds"] = edges
                        calibrations[tag + "_levels"] = centers
                    decs = {
                        kind: select_decoder(yt, xx, yv, xv, segments, kind)
                        for kind in ["matched", "quadratic"]
                    }
                    for trial in range(3):
                        rawtest, _, _, expected = sense(
                            xt,
                            p,
                            n,
                            k,
                            b,
                            mode,
                            base,
                            rng("test" + str(trial)),
                            adc=False,
                        )
                        y, ids = apply_adc(rawtest, adc)
                        occ = occupancy(ids)
                        # White-reference one-bit control, independent of decoder.
                        uniform_ids = np.rint(
                            np.clip(rawtest / 1.2, 0, 1) * (2**b - 1)
                        ).astype(int)
                        occ_u = {
                            "uniform_" + a: v for a, v in occupancy(uniform_ids).items()
                        }
                        for kind, (verr, alpha, dec) in decs.items():
                            rows.append(
                                {
                                    "scene": name,
                                    "config": "quantile_adc",
                                    "budget": 24,
                                    "optics": mode,
                                    "photons": n,
                                    "channels": k,
                                    "bits": b,
                                    "decoder": kind,
                                    "trial": trial,
                                    "validation_rmse": verr,
                                    "alpha": alpha,
                                    **measure(xt, predict(y, dec)),
                                    "expected_detected_photons": expected,
                                    **occ,
                                    **occ_u,
                                }
                            )
        write_rows(dest, rows)
        np.savez_compressed(out / (name + "_calibration.npz"), **calibrations)
        print(f"{name}: {len(rows)} rows; {time.time()-t0:.1f}s", flush=True)
    print("Extension complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="vns_robust_outputs")
    p.add_argument("--out", default="vns_quantile_outputs")
    run(p.parse_args())
