# Code map

Run the final workflow from the repository root. Imports between scripts are
resolved from `code/` when a script is launched by filename.

| Script | Role |
| --- | --- |
| `vns_budget_sweep.py` | CAVE loader, spectral operators, synthetic sanity check and original pilot sweep |
| `vns_robustness.py` | Main scene-separated simulation, quantization, decoders and sensitivity conditions |
| `vns_robust_analysis.py` | Baseline aggregation, bootstrap comparisons, rank diagnostics and risk approximation |
| `vns_quantizer_extension.py` | Training-calibrated quantile ADC experiment and saved codebooks |
| `vns_paper_analysis.py` | Joint validation selection, paired effects and final analysis tables |
| `vns_paper_tables.py` | Optional LaTeX table export from saved data; no manuscript included |
| `vns_paper_figures.py` | Final plots and forward-model figure |
| `vns_paper_images.py` | Held-out full-image reconstruction example |
| `test_vns.py` | Ten physics and implementation checks |
| `test_quantizer_extension.py` | Four quantizer and detected-count checks |
| `verify_paper_results.py` | Independent recomputation of representative results and final selections |
| `verify_archive.py` | File integrity check for this archive |

The remaining scripts preserve earlier work: `aggregate_loso.py`,
`vns_final_figures.py`, `vns_image_examples.py`, `vns_system_diagram.py`,
`build_research_report.py`, `package_and_validate.py` and
`assemble_paper_package.py`. Some use the original scratch-directory layout
or build an earlier deliverable. They are historical source, not additional
steps required to reproduce the final results.

All supplied Python files have consistent formatting. Their parsed syntax
trees were compared before and after formatting; experiment logic and
parameters were unchanged. Comments explain scientific choices where they
matter, particularly photon accounting, scene separation and quantizer ties.
