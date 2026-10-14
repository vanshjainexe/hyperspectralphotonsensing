# Methods and scope

This is a computational sensor-design study on the 15 complete CAVE scenes in the supplied archive. It is not a fabricated camera, an experimentally measured optical transfer function, or an exhaustive optimum over all possible spectral sensors. Version 2 was specified after inspecting the initial pilot; it is a robustness study, not preregistered confirmatory research.

## Question

At fixed spatial macro-pixel area and digital output budget, which combination of spectral-channel count K and intensity-code depth B produces the smallest held-out spectral reconstruction error? How does this depend on photon exposure, routing versus absorptive filtering, reconstruction, ADC calibration and sensor imperfections?

## Photon accounting

The 31-dimensional normalized CAVE spectrum is x. Source bands are treated as equally weighted photon-illumination bins. A nonnegative K by 31 matrix P partitions these bins, with each column summing to 1/31. The baseline uses contiguous integer groups from numpy.array_split. Groups can differ in size by one source bin; they are only approximately equal in width. A separate fractional-overlap sensitivity test makes all output bins equal in width without losing energy.

N is the expected photon count for a unit-reflectance white macro-pixel before routing/filter losses. Actual scene pixels do not all receive N photons: their brightness and spectral content modulate their expected counts. This convention fixes exposure/aperture across candidate designs for the same scene. It does not normalize each pixel to the same brightness.

- Router: mean counts = N eta M P x.
- Filter: mean counts = N tau P x / K.
- eta is router throughput, baseline 0.84. M is the identity except in the router-crosstalk tests.
- tau is filter passband transmission, baseline 1.0.
- The 1/K in the filter model is a fixed-macro-pixel-area penalty, separate from spectral selection in P. Equal-white bins yield approximately N eta/K versus N tau/K^2 per channel.

All detector outputs are assumed to sample the same local spectrum within the macro-pixel. No spatial blur, demosaicing, angular response or scene motion is simulated. Sensor quantum efficiency is one (or absorbed into N). The comparison is not a measured commercial Bayer camera benchmark.

Passive linear optics separates wavelengths; it cannot classify a mixed beam by a semantic color such as purple. Spectral mixtures are recovered by the decoder from channel measurements.

## Sensor and quantizer

Counts are Poisson(signal + dark) minus the calibrated expected dark count, with optional additive Gaussian read noise. Negative values are clipped to zero before ADC. Dark is measured in expected photoelectrons per detector per exposure. Read noise is standard deviation in photoelectrons. This is a simplified dark-offset correction model, not a full analog front end.

Each channel's full scale equals H times its white-reference expected count. H = 1.2 in the primary model. Exposure-dependent gain is assumed: full scale changes with N, optics and K, but never uses the test scene. This is not a fixed-full-well sensor operated across the entire exposure sweep. Saturation is recorded before clipping.

The uniform ADC has 2^B reconstruction levels from zero to full scale, with nearest-level rounding. The square-root ADC sensitivity case uniformly quantizes sqrt(count/fullscale), then squares the reconstructed code. It also has exactly 2^B codes. No compander parameter is optimized on test spectra.

The assumed 84% throughput is motivated by an RGB-routing publication, not a claim that a 24-channel device has been built at 84% throughput. The declining-throughput sensitivity is deliberately hypothetical: eta(K) = 0.84 - 0.24(K-3)/21. Its endpoints are assumptions, not measurements copied from any paper. Crosstalk redistributes 15% of each source channel into its adjacent channel(s), with columns of M summing to one. It preserves photons and is known during decoder calibration.

## Budgets and conditions

- 12 bits: 3x4, 4x3, 6x2, 12x1.
- 24 bits: 3x8, 4x6, 6x4, 8x3, 12x2, 24x1.
- 48 bits: 3x16, 4x12, 6x8, 8x6, 12x4, 16x3, 24x2.
- N: 10, 30, 100, 300, 1,000, 3,000, 10,000, 30,000, 100,000.

The sweep includes baseline, H=1.0, H=2.0, square-root ADC, read noise 2 e-, combined read noise 5 e-/dark 0.2 e-/router crosstalk 15%, crosstalk alone, declining throughput, filter transmission 80%, fractional-overlap bands, and the 12-/48-bit cases. These are one-factor-at-a-time tests except for the explicit combined stress case. They are not an exhaustive factorial experiment.

## Scene-held-out validation

Scenes are ordered alphabetically. For each of 15 outer folds, one entire scene is the test set; the next three cyclic scenes are validation; the remaining eleven are training. Each training scene contributes 600 pixels (6,600 total), each validation scene contributes 1,024 (3,072 total), and the test scene contributes 4,096. Sampling uses a fixed SHA256-derived seed without replacement. The same sampled pixels are reused across candidate conditions, not treated as independent scene replicates.

The matched decoder trains on two independently corrupted copies of the training spectra. It selects ridge regularization from 0.0001, 0.01, 1, 100 using average per-scene validation RMSE. The design selector then chooses the candidate K,B with smallest average validation RMSE. The test scene never chooses either regularization or allocation. Three independent test-noise trials are averaged within each scene. A single noisy validation draw is used per condition; this is a limitation.

Main reconstruction controls:

1. Clean-trained affine ridge (standardized features, alpha=1e-5).
2. Noise-/quantization-matched affine ridge.
3. Noise-/quantization-matched quadratic ridge: measurements, their squares, and adjacent-channel products.
4. Piecewise-linear spectral interpolation with constant endpoint extension; no scene fitting.

All reconstructed spectra are clipped to [0,1]. The quadratic control is a modest nonlinear baseline, not a state-of-the-art neural network. The version-1 pilot used a different split (14 training scenes), larger sample sizes, and an unstandardized clean-trained affine decoder. Its numbers must not be pooled with version 2.

## Metrics and selection

Spectral RMSE is the square root of average squared error over all test pixels and 31 bands, in normalized reflectance units. Scene RMSE is averaged equally across scenes, after averaging noise trials. PSNR uses peak=1 and is computed per scene/trial before scene averaging; it is not -20log10 of the cross-scene mean RMSE.

SAM is mean spectral angle in degrees. Exactly zero-reference spectra are excluded. For nonzero reference with zero predicted vector, SAM is assigned 90 degrees instead of silently omitting failed predictions. Version-1 SAM omitted zero-denominator cases; therefore the v2 SAM convention supersedes it. Very dark nonzero spectra remain included and can disproportionately affect average SAM.

A null baseline predicts the training-set mean spectrum without inspecting any measurement. Global test-error minima and per-scene test-error minima are explicitly labelled descriptive/oracle results. They are not an unbiased estimate of a deployable design selector. Only the validation-selected test results support claims about choosing a design in advance.

Bootstrap intervals use 10,000 paired resamples of the 15 held-out scenes after averaging noise trials. Shared resamples preserve pairings. These intervals summarize variation within this selected dataset; overlapping cross-validation training sets and a nonrandom 15-scene subset limit population-level inference. Many comparisons are exploratory; intervals and permutation p-values are not multiplicity-adjusted. There is no claim of an exactly known crossover between sampled exposures.

## Complexity and compact design rule

Effective rank is exp(-sum p_i log p_i), where p_i are normalized eigenvalues of centered spectral covariance. Raw rank includes brightness variation. Shape rank first normalizes each sufficiently bright pixel's spectrum by its spectral sum (mean reflectance >0.001). Spearman associations with the test-oracle high-light K are exploratory; the test-scene rank is not input to the deployable selector. Two-sided permutation p-values use 20,000 permutations and a +1 correction. Neither association proves that dimensionality determines an optimum.

A compact empirical risk surrogate fits nonnegative C_K and A_K to validation RMSE squared at N=30, 1,000 and 30,000:

    predicted MSE(K,N) = C_K + A_K/N

It then chooses the K with smallest predicted MSE. Other exposures assess interpolation/extrapolation. Coefficients are fitted without test scenes; the formula is approximate, particularly where clipping, coarse quantization or noise-dependent decoder weights matter.

For a fixed affine decoder W, ignoring clipping and high-resolution quantizer correlations, the motivation is:

    spectral MSE approximately representation error
        + (1/31) sum_j ||W_j||^2 E[band_average_j]/(N t w_j)
        + H^2 ||W||_F^2 / [31 * 12 * (2^B-1)^2].

Here t=eta for the router and tau/K for the filter, w_j is the fraction of white photons in channel j, and W_j maps one normalized channel measurement into the 31 reconstructed bands. The quantization term is a high-resolution approximation and is not valid as a theorem for one-bit measurements. Cross terms, estimator bias and clipping can matter. This is an explanatory approximation, not a new general information-theoretic law.

## Reproducibility and integrity

The package contains all simulation-condition outputs, per-scene and aggregated tables, a sampled-spectrum cache, source code, environment versions, input checksums, fold definitions, unit checks, PNG/SVG figures, and example reconstructed image arrays. Original full-resolution CAVE source images remain in the user's archive; they are not duplicated in this smaller package. Synthetic/random quantities are generated by the stated model, not collected from physical hardware. Image examples use held-out balloons at fixed coordinates and a fixed display gain; their pseudo-color is not a calibrated RGB color transformation.

No second independent dataset, physical prototype, fabricated nanostructure, fixed-area electromagnetic design, real sensor calibration or experimental photon measurement was completed. These delimit the claim rather than invalidating a carefully reported computational result.
