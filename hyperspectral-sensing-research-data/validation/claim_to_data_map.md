# Key claims and their numerical sources

- 24.0% router reduction at N=100000 versus fixed 3x8/uniform: `VNS_paper/data/paper_paired_effects.csv`, candidate `joint_selection`, reference `fixed3_uniform`, optics `router`.
- 13.1% incremental router reduction versus channel-only/uniform selection: same table, reference `selected_uniform`.
- 24.2% reduction versus a three-channel baseline that can choose its ADC: same table, reference `fixed3_calibrated`.
- The three metric columns in manuscript Table 1: `paper_strategy_summary.csv`; exact values used by `code/vns_paper_tables.py`.
- One-bit entropy 0.834 versus 0.144: `paper_code_occupancy.csv`, router, N=100000, K=24.
- One-bit RMSE 0.1168 versus 0.1209: `quantizer_cross_scene_metrics.csv`, matched decoder, router, N=100000, K=24, quantile versus main.
- Spectral-rank association rho=0.355 and p=0.198: `VNS_research_package/data/complexity_correlations.csv`, matched/raw rank.
- Original 24.6% routing advantage at N=100: `VNS_research_package/data/router_advantage_validation_selected.csv`.

The final claim is conditional on the stated optics and noise model. A numerical gain alone does not establish worldwide priority or hardware performance.
