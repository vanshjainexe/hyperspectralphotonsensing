# Historical pilot - superseded conclusions

These are the nine diagrams and associated numerical runs generated before the version-2 robustness study. They are retained because the user requested all outputs and because they document how the interpretation changed.

Do not mix these metrics with version 2. This pilot uses a clean-trained affine decoder, 14 training scenes per fold, different pixel counts and seed, and a different handling of zero vectors in SAM. The label `ideal_router` in CSV files is a legacy optical-mode identifier; actual efficiency must be read from the corresponding run manifest. In the main pilot it is 0.84, not 1.0.

Plots label N as incident photons per macro-pixel. The precise interpretation is white-reference photon exposure: actual scene photons depend on the spectrum. The `95% CI` shading is a normal approximation across scenes, not the paired bootstrap used in the final analysis. Error-source ablations are separate forward-model configurations, not additive RMSE terms.

The pilot's 3x8-to-4x6 transition and stronger spectral-rank correlation are not robust universal conclusions. The version-2 matched-decoder map differs, and the raw-rank association is much weaker. Use the final report to describe the current findings, and refer to these figures only as the earlier clean-decoder experiment.
