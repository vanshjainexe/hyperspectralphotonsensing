# Prior work and originality assessment

Checked 13 October 2026. This is a targeted search, not an exhaustive systematic review or patent clearance. The earlier numeric originality score (7.5/10) was a subjective rating without a validated scale and should not appear in a paper.

## Defensible contribution

The proposed contribution is a controlled computational evaluation of joint spectral-channel/ADC-bit allocation under a stated photon exposure and macro-pixel model, comparing routing with absorptive filtering and testing dependence on reconstruction and calibration. The reconstruction methods, photon statistics, quantizers and routing principle are established techniques. Using exactly 24 bits or exactly these six pairs is not by itself a meaningful novelty claim.

This search did not identify a work implementing the complete same photon/bit/router-filter/decoder-robustness comparison. That supports investigating a potentially differentiated study; it does not prove priority. Do not claim that nobody optimized photographic spectral sensing before, that spectral routing is new, or that this is a demonstrated new optical device.

## Relevant primary sources

1. Jason N. Laska and Richard G. Baraniuk. Regime Change: Bit-Depth versus Measurement-Rate in Compressive Sensing. IEEE Transactions on Signal Processing (2012). DOI: 10.1109/TSP.2012.2194710. https://arxiv.org/abs/1110.3450

   Establishes the measurement-count/bit-depth trade-off and SNR-dependent regimes in compressive sensing. Its measurement and noise assumptions differ from splitting a fixed Poisson photon exposure among spectral bands. The direction of its trend is not automatically a prediction for this sensor.

2. Ayca Ozcelikkale and Haldun M. Ozaktas. Representation of optical fields using finite numbers of bits. Optics Letters 37, 2193-2195 (2012). DOI: 10.1364/OL.37.002193. https://opg.optica.org/abstract.cfm?uri=ol-37-12-2193

   Optimizes optical-field sample count, spacing and precision under a bit cost. This makes broad 'first to trade samples against bits in optics' language inappropriate. It is not the routed snapshot reflectance-cube experiment here.

3. Ayca Ozcelikkale and Haldun M. Ozaktas. Beyond Nyquist sampling: a cost-based approach. JOSA A 30, 645-655 (2013). https://opg.optica.org/abstract.cfm?uri=josaa-30-4-645

   Further develops joint optical sampling and amplitude accuracy. It is theoretical context for cost-aware design, not a duplicate of the proposed sensor comparison.

4. Fumihito Yasuma, Tomoo Mitsunaga, Daisuke Iso and Shree K. Nayar. Generalized Assorted Pixel Camera: Post-Capture Control of Resolution, Dynamic Range and Spectrum. Columbia technical report CUCS-061-08 (2008). https://cave.cs.columbia.edu/old/publications/pdfs/Yasuma_TR08.pdf

   Optimizes multispectral mosaic layouts and reconstruction using real multispectral scenes. This is a close imaging-system precedent and the CAVE dataset context. Its spatial/spectral/dynamic-range trade-off is broader than simple RGB photography; citing it prevents overclaiming the application area.

5. Xiujuan Zou et al. Pixel-level Bayer-type colour router based on metasurfaces. Nature Communications 13, 3288 (2022). DOI: 10.1038/s41467-022-31019-7. https://www.nature.com/articles/s41467-022-31019-7

   Reports 84% average energy utilization for a fabricated RGB-routing arrangement. This is an enabling hardware precedent, not a measured 84%-efficient family of arbitrary-K routers. Our constant efficiency is a scenario; real wavelength leakage and throughput require separate calibration.

6. Junseo Han, Seunghyun Lee, Donghyun Kim and Haejun Chung. Scaling Limits of Multichannel Spectral Routers for Snapshot Imaging. arXiv:2607.23508v2 (2026 preprint). https://arxiv.org/html/2607.23508v2

   Investigates 9, 16, 25 and 36 channels with electromagnetic simulations and optical scaling arguments. The discussion explicitly excludes detector noise and image-level reconstruction. It motivates examining those layers but is not evidence for our hypothetical efficiency curve.

7. D. M. J. van Elst et al. Application-Specific Optimization of Integrated Spectral Sensors. ACS Photonics (2025). DOI: 10.1021/acsphotonics.5c01213. https://pubs.acs.org/doi/10.1021/acsphotonics.5c01213

   Optimizes integrated spectral-band choices for an application and reports that fewer channels can improve noise-sensitive performance. This is particularly relevant to the claim that maximizing the number of bands is not always optimal. Publisher search excerpts and the institutional record were accessible; full text was not successfully retrieved in this session. Therefore we do not assert that this paper excludes every possible ADC-budget analysis.

An earlier draft cited arXiv:2609.28450 as a spatial-spectral co-design paper. That reference was not independently retrievable during this verification; it is excluded from the verified bibliography and should not be relied on without checking.

## Search record

Search phrases included: 'hyperspectral bit budget quantization spectral channels'; 'hyperspectral spectral channels ADC bit depth photon noise sensor design optimization'; 'spectral routing quantization hyperspectral'; 'hyperspectral number of bands bit depth optimization'; 'hyperspectral bit allocation sensor quantization design'; and exact titles/identifiers for the references above. Results included both acquisition design and post-capture compression. Compression bit allocation is relevant background but does not, by itself, duplicate physical photon allocation during acquisition.

## Claims permitted by this work

- We implemented and evaluated an explicit joint-budget forward model on a stated 15-scene CAVE subset.
- We measured how allocation choices change with decoder, calibration and optical assumptions.
- We tested a validation-only allocation selector and an approximate three-exposure risk surrogate on held-out scenes.

## Claims not established

- 'First-ever', globally unique, patentable, universal optimal K, or guaranteed competition significance.
- A new fabricated router, an intelligent photon sorter, or an observed commercial-camera improvement.
- A universal law that the optimal channel count equals intrinsic spectral rank.

The appropriate positioning is an empirical computational-design study using established components, with its contribution judged by the controlled comparisons, reproducibility and results rather than an originality score.
