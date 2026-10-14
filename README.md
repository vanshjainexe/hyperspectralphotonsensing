# HYPERSPECTRAL PHOTON SENSING RESEARCH

Research archive for the computational study by Vansh Jain.

The experiment asks how a fixed output budget should be divided between the
number of spectral channels and the intensity precision of each channel. It
compares wavelength routing and equal-area absorptive filtering with Poisson
noise, quantization and reconstruction of 31-band spectra.

This repository contains the experiment code, data, saved results and figures.
The manuscript and paper PDFs are excluded. The results are simulations based
on measured CAVE spectra; no physical camera was fabricated or tested.

## What is included

| Path | Contents |
| --- | --- |
| `code/` | Experiment, analysis, figure and verification scripts |
| `vns_robust_outputs/` | 180 per-scene condition CSVs, scene folds, protocol and exact sampled spectra |
| `vns_quantile_outputs/` | 15 result CSVs and 15 saved calibration files for the quantile experiment |
| `VNS_research_package/data/` | Baseline aggregate tables and all 72,090 baseline trial records |
| `VNS_paper/data/` | All 9,720 extension trial records, selection tables, uncertainty estimates and numerical image cubes |
| `VNS_paper/figures/` | Final vector figures and PNG previews |
| `data/raw/CAVE_Dataset/` | The 15 source scenes from the uploaded dataset, in the full ZIP archive |
| `validation/` | Source checksums, environment, audits and archive checksums |
| `VNS_research_package/pilot_v1/` | Earlier pilot runs, retained as historical work |
| `history/` | Execution logs |

The smaller code-and-results ZIP contains the same code and results but omits the raw
scene images. It still contains the exact sampled spectra and image arrays used
for numerical reanalysis. The full ZIP contains the raw images too.

## Install and check

Use Python 3.12. Run commands from this directory, rather than from `code/`.
On Windows, replace `python` with `py` if that is your Python launcher.

```bash
python -m pip install -r requirements.txt
python code/test_vns.py
python code/test_quantizer_extension.py
python code/verify_paper_results.py
python code/verify_archive.py
```

The first two scripts run 14 checks of the forward model and quantizer. The
numerical audit recomputes representative trial results and all 270 joint
validation selections. Archive verification compares files against their SHA256
checksums. These checks establish reproducibility of the implementation, rather
than validating the optical assumptions.

## Rebuild the analysis

These commands use saved trial data and do not need the full source images.

```bash
python code/vns_robust_analysis.py
python code/vns_paper_analysis.py
python code/vns_paper_figures.py
```

Regenerating outputs can change their byte-level checksums while retaining the
same numerical values. Use `verify_archive.py` before rebuilding if you want to
check the delivered files.

## Rerun the simulations

The full ZIP includes the source images. Run the baseline sweep into a new
directory so completed results are not reused:

```bash
python code/vns_robustness.py --data-root data/raw/CAVE_Dataset --out reruns/baseline
```

The quantile experiment can be rerun directly from the delivered sample cache:

```bash
python code/vns_quantizer_extension.py --source vns_robust_outputs --out reruns/quantile
```

The analyzers expect the delivered directory names. To analyze fresh runs,
use a separate working copy, replace its `vns_robust_outputs/` and
`vns_quantile_outputs/` with the fresh outputs, then run the analysis commands.
Keep the original archive intact for comparison.

For the full-image reconstruction example, the full ZIP supplies the source
images and the protocol already points to their relative location:

```bash
python code/vns_paper_images.py
```

Large runs can take longer on a laptop than on the original machine. Limiting
BLAS threads to one can avoid oversubscription; this is optional.

## Reading the results

The primary output budget is 24 bits per spatial macro-pixel. The candidates
are 3x8, 4x6, 6x4, 8x3, 12x2 and 24x1. The baseline uses nine exposures, two
optical modes and three test noise trials. All fifteen outer folds separate
training, validation and test scenes. The final design is selected using
validation error, not the held-out test error.

At the highest exposure, joint selection of channel count and quantizer lowers
router RMSE by about 24% relative to the fixed three-channel uniform baseline,
and about 13% relative to channel-only selection. Those gains are conditional
on the stated model. The trial rows share scenes and are not independent
experimental replicates. The final study does not support a universal rule
based on spectral rank.

The folder name `VNS_paper/` is retained because the analysis scripts use it;
it contains only numerical data and figures, not the manuscript. Vector PDFs
in `figures/` are individual plots.

The historical pilots use different evaluation and training choices. Their
claims should not replace the final paper's conclusions. Historical figure and
report builders are retained for completeness; the maintained final workflow
is the one listed above. `CODE_MAP.md` identifies the scripts.

## Data provenance and GitHub

The source scenes are from the CAVE Multispectral Image Database associated
with Yasuma et al., *Generalized Assorted Pixel Camera: Postcapture Control of
Resolution, Dynamic Range, and Spectrum*, IEEE TIP 19(9), 2241-2253 (2010),
DOI: 10.1109/TIP.2010.2046811. The original collection belongs to its providers;
no new license is assigned to it here.

The full archive preserves all 465 spectral PNGs and the accompanying files
from the uploaded 15-scene subset. It is not the full 32-scene CAVE database.
The cached sample file contains 4,096 normalized 31-band spectra per scene.
The saved reference image cube is a subsampled held-out balloons scene.

`.gitignore` keeps `data/raw/` and fresh reruns out of ordinary commits. The
full ZIP is your local data backup. Before publicly mirroring the original
dataset, check its provider's redistribution terms. The official dataset
repository is https://cave.cs.columbia.edu/repository.

To archive the code and results on GitHub, extract the archive, open the folder
as a repository and commit its contents. The included checksum manifests make
it possible to identify the exact files used in this study.

## CS side of the paper 

On the computing side, I have tested three ways to turn photon counts into digital values: uniform quantization, square-root companding and empirical quantile quantization. A regression model then tries to recover the original 31-band spectrum from those readings. We choose the channel count and coding method on validation scenes, then test them on separate scenes to see how well they use the same 24-bit budget.


Any critiques, comments, opinions and ideas are appreciated!
