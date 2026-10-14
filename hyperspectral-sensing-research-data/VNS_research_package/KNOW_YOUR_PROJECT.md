# Your project, in plain language

## The one-sentence question

If a camera can save only 24 bits for each tiny patch of the scene, should it measure a few colors very accurately or more colors less accurately?

## What an ordinary image leaves out

A normal RGB image records three broad color measurements. A hyperspectral image records a much more detailed description of how much light comes from different wavelengths. That detail can help distinguish materials that look similar in RGB. We are studying reconstruction of that detailed spectrum, not whether a picture looks aesthetically pleasing.

## The two knobs

Channels means the number of wavelength groups measured separately. Bits per channel means the number of different brightness values you can record for each group. Eight bits gives 256 possible codes; four bits gives 16; one bit gives just two. The fixed 24-bit rule means increasing channels forces you to decrease bits per channel.

## The problem is not just random photons

Three limitations compete. Too few channels mixes together spectral detail. Too many channels spreads photons thinly. Too few bits merges different intensities into the same code. The best compromise also depends on how well the decoder uses patterns learned from other spectra.

## Router versus filter

A filter rejects wavelengths it does not want. A router redirects different wavelengths to different outputs. Routing can preserve more light. It does not create photons, recognize purple, or choose a semantic destination for an entire mixed beam. Blue and red components remain separate wavelength contributions.

## What we actually built

A Python simulation and design-selection procedure. We did not build an optical chip. We used the 15 measured CAVE scenes in your archive as reference spectra, simulated sensor measurements, reconstructed the spectra, and compared reconstruction with the reference.

## Why the dataset is needed

We need a known correct answer. The 31 CAVE bands tell us what each sampled scene spectrum looks like. The simulation hides most of that information by converting it to a small set of noisy, quantized measurements. We then test how well the decoder recovers the original information.

## What 'optimization' means here

Try the allowed designs on validation scenes. Measure the reconstruction error. Choose the design with the smallest validation error for that light level. Test the chosen design on a different scene. It is a finite search, not magic AI. The research question is which design wins under which conditions and why.

## Before or after taking the picture?

The channel layout and code depth belong to acquisition and must be chosen before measurements are made. Reconstruction happens afterward. Our software helps choose the acquisition design and evaluates the whole process. A manufactured fixed router cannot suddenly change its channels just because a decoder asks it to; runtime reconfiguration would require additional hardware and is not demonstrated here.

## Why train on other scenes?

If the decoder sees the correct spectra of the same scene beforehand, it can appear much more capable than it would be on a new scene. Our test scene is held out from both training and design selection.

## What changed after the first experiment?

The first decoder learned from perfectly clean measurements but was evaluated on noisy measurements. We added a decoder that learns from realistic noise and quantization. That changes which channel/bit split is best. We keep the original result but no longer present its simple transition as universal.

## What the numbers mean

- RMSE: overall reconstruction error. Lower is better.
- SAM: how different the shape of the reconstructed spectrum is, measured as an angle. Lower is better. Dark pixels make this difficult.
- PSNR: another way to express reconstruction accuracy. Higher is better.
- Confidence interval: a description of uncertainty from varying the scenes. A narrow-looking curve is not proof of a universal law.
- Oracle winner: the best design found after looking at the test answers. Useful for analysis, but not a fair deployable decision procedure.
- Validation-selected winner: a choice made before looking at the test answers. This is what supports a practical design rule.

## What might be original?

The narrow joint comparison and its findings: a fixed photon-exposure/bit-budget study of routed versus absorptive spectral sensing, including decoder and calibration dependence. The general concepts already exist. Combining known methods can support a research contribution, but a literature search that finds no duplicate does not prove worldwide uniqueness.

## What you should be able to explain to a judge

Why 24 bits? What is held fixed? Why does the filter have an extra area loss? How are counts generated? How are brightness codes assigned? What does the decoder learn? Which scenes choose the design? Which scenes evaluate it? What changed when you made the model more realistic? What has not been built or tested?

The strongest presentation is one you understand and can defend, including the result that challenged the first hypothesis. Read the draft and run the checks before submitting. State the AI assistance accurately rather than claiming you personally wrote or experimentally measured things you did not.
