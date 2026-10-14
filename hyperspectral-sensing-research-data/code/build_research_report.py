"""Build a paginated research draft directly from verified v2 result tables."""

from pathlib import Path
import argparse
import json
import re
import html
import pandas as pd
from PIL import Image as PILImage
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib

fontdir = Path(matplotlib.get_data_path()) / "fonts/ttf"
pdfmetrics.registerFont(TTFont("VNS", str(fontdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("VNS-Bold", str(fontdir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily(
    "VNS", normal="VNS", bold="VNS-Bold", italic="VNS", boldItalic="VNS-Bold"
)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="VNS_research_package")
args = parser.parse_args()
OUT = Path(args.out)
D = OUT / "data"
F = OUT / "figures"
summary = json.loads((D / "results_summary.json").read_text())
win = pd.read_csv(D / "descriptive_global_winners.csv")
benefit = pd.read_csv(D / "heldout_selection_benefits.csv")
gain = pd.read_csv(D / "router_advantage_validation_selected.csv")
main = win[(win.config == "main") & (win.decoder == "matched")]
cc = pd.read_csv(D / "complexity_correlations.csv")
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="BodyV",
        fontName="VNS",
        fontSize=9.6,
        leading=13.5,
        spaceAfter=8,
        textColor=colors.HexColor("#24313d"),
    )
)
styles.add(
    ParagraphStyle(
        name="SmallV",
        fontName="VNS",
        fontSize=7.8,
        leading=10.5,
        spaceAfter=7,
        textColor=colors.HexColor("#52616b"),
    )
)
styles.add(
    ParagraphStyle(
        name="TitleV",
        fontName="VNS-Bold",
        fontSize=21,
        leading=26,
        spaceAfter=16,
        textColor=colors.HexColor("#133b4f"),
    )
)
styles.add(
    ParagraphStyle(
        name="HeadV",
        fontName="VNS-Bold",
        fontSize=14,
        leading=18,
        spaceAfter=10,
        textColor=colors.HexColor("#133b4f"),
    )
)
story = []
md = []


def p(text, small=False):
    story.append(Paragraph(text, styles["SmallV" if small else "BodyV"]))
    md.append(re.sub("<[^>]+>", "", text) + "\n")


def h(text):
    story.append(Paragraph(text, styles["HeadV"]))
    md.append("## " + text + "\n")


def page():
    story.append(PageBreak())
    md.append("\n---\n")


def image(name, caption, width=490):
    path = F / name
    w, hgt = PILImage.open(path).size
    story.append(Image(str(path), width=width, height=width * hgt / w))
    story.append(Spacer(1, 6))
    p(caption, True)
    md.append(f"![{caption}](figures/{name})\n")


def table(rows, widths=None):
    data = [[Paragraph(str(c), styles["SmallV"]) for c in row] for row in rows]
    t = Table(
        data,
        colWidths=widths or [490 / len(rows[0])] * len(rows[0]),
        repeatRows=1,
        hAlign="LEFT",
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eef2")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#287f8e")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f7fafb")],
                ),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 9))
    md.append(
        "| "
        + " | ".join(map(str, rows[0]))
        + " |\n| "
        + " | ".join(["---"] * len(rows[0]))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows[1:])
        + "\n"
    )


def row_at(mode, n):
    return main[(main.optics == mode) & (main.photons == n)].iloc[0]


def pair(r):
    return f"{int(r.channels)} x {int(r.bits)}"


story.append(
    Paragraph(
        "Photon and bit allocation for routed snapshot spectral sensing",
        styles["TitleV"],
    )
)
md.append("# Photon and bit allocation for routed snapshot spectral sensing\n")
p("<b>A decoder- and calibration-aware computational study on 15 CAVE scenes</b>")
p("Project: Vansh Jain | AI-assisted research draft | 13 October 2026", True)
h("Abstract")
g100 = gain[gain.photons == 100].iloc[0]
g100k = gain[gain.photons == 100000].iloc[0]
p(
    f'We study how a fixed digital output budget should be divided between spectral-channel count and per-channel ADC depth under photon-limited acquisition. A 31-band CAVE subset is simulated through routed and equal-area absorptive optical models, Poisson detection and finite-bit quantization. The primary budget is 24 bits per spatial macro-pixel. Fifteen outer scene folds separate training, validation and testing. A robustness sweep evaluates four decoders and twelve configurations, producing {summary["trial_condition_decoder_rows"]:,} trial-condition-decoder records. Allocation preferences depend materially on reconstruction and quantization; the initial clean-trained-decoder transition is not a universal sensor rule. After validation-only allocation selection, the modeled router changes mean RMSE relative to the filter by {g100.router_gain_percent:.1f}% at exposure N=100 and {g100k.router_gain_percent:.1f}% at N=100,000 (positive means lower router error). We release simulation outputs, code, plots and a compact validation-fitted risk rule. Conclusions are limited to the specified model and dataset subset, not a fabricated camera or a universal optimum.'
)
h("Research question")
p(
    "With a fixed area, exposure and output-bit budget, should a sensor record a few spectral channels precisely or more channels coarsely? Does preserving photons through routing change that decision, and does the answer survive changes in the reconstruction method?"
)
h("Contribution and scope")
p(
    "The contribution is a controlled joint-budget comparison and a tested design-selection procedure. The ingredients are established. No optical device was fabricated; no world-first priority or competition outcome is claimed. All noise realizations are numerical simulations applied to real measured spectra."
)
p(
    "The main body presents version 2. Earlier pilot figures are retained separately because the decoder, split and metric conventions were revised. Full equations, configuration details and assumptions are in METHODS.md.",
    True,
)

page()
h("1. What is being designed?")
image(
    "00_forward_model.png",
    "Figure 1. Forward model. The router and filter are alternative optical paths. Photon counts, not semantic colors, enter the ADC and decoder.",
    490,
)
p(
    "A spatial macro-pixel contains K detector outputs. The sensor emits B bits from each output, so K times B is fixed. At 24 bits, the candidates are 3x8, 4x6, 6x4, 8x3, 12x2 and 24x1. The 3x8 candidate is a three-band spectral sensor, not a complete commercial RGB-camera model."
)
p(
    "The optical design is chosen before acquisition. The ADC converts the resulting counts into digital codes; the decoder subsequently estimates the original 31-band spectrum. A post-capture algorithm cannot recover information that was never measured without relying on a learned prior."
)
p(
    "Routing preserves more of the available light by directing wavelengths to different outputs. Absorptive filters discard out-of-band light. Under the equal-area macro-pixel model, each filter occupies 1/K of the area, creating the extra area factor below. Spatial sampling is otherwise idealized."
)
table(
    [
        ["Quantity", "Router", "Filter"],
        [
            "Mean channel count",
            "N eta (P x)<sub>k</sub>",
            "N tau (P x)<sub>k</sub> / K",
        ],
        ["Flat-white approximation", "N eta / K", "N tau / K<super>2</super>"],
        ["Baseline throughput", "eta = 0.84", "tau = 1.0"],
    ],
    [180, 155, 155],
)
p(
    "N denotes white-reference photon exposure. Actual scene pixels vary in brightness; they do not each receive the same detected photon count. Unit quantum efficiency and equal photon weighting of the 31 input bands are assumed.",
    True,
)

page()
h("2. Validation and measurement protocol")
table(
    [
        ["Item", "Protocol"],
        ["Data", "15 supplied CAVE scenes; 31 normalized bands, 400-700 nm"],
        ["Outer folds", "Each scene is tested once"],
        ["Per fold", "11 training scenes; next 3 cyclic scenes validate; 1 test scene"],
        ["Pixel counts", "6,600 training; 3,072 validation; 4,096 test"],
        [
            "Noise realizations",
            "Two corrupted training copies; one validation draw; three test trials",
        ],
        [
            "Exposure grid",
            "10 to 100,000 white-reference photons at nine sampled levels",
        ],
        ["Primary outcome", "Mean scene spectral RMSE; SAM and PSNR secondary"],
        ["Uncertainty", "10,000 paired scene-bootstrap resamples, descriptive"],
    ],
    [145, 345],
)
p(
    "The matched affine decoder learns from the same Poisson-plus-ADC process used at evaluation. Ridge strength is chosen from four values on validation scenes, separately for each allocation and exposure. The allocation selector then chooses the smallest validation error. No test scene is used for either choice."
)
p(
    "The controls are a clean-trained affine decoder, a modest quadratic ridge model and nonlearned interpolation. All outputs are clipped to the valid reflectance range. These controls test reconstruction dependence; they do not exhaust possible algorithms."
)
p(
    "The ADC uses a white-reference full scale with headroom 1.2. Full scale is recalibrated for each exposure, optical mode and channel count, not for the test scene. The experiment therefore assumes exposure-dependent gain, rather than a single fixed full-well range."
)
p(
    "SAM excludes zero reference vectors. A nonzero reference reconstructed as zero receives 90 degrees, rather than disappearing from the average. Very dark spectra remain in the metric. PSNR is averaged after per-scene calculation and is not simply computed from the aggregate RMSE."
)
p(
    "The source archive is a subset, and outer folds reuse training scenes. Bootstrap intervals describe variation in this set, not a guaranteed population-level confidence statement. The robustness analysis followed inspection of the pilot and is explicitly exploratory.",
    True,
)

page()
h("3. Allocation curves with a matched decoder")
image(
    "01_matched_decoder_error_curves.png",
    "Figure 2. Held-out scene-mean RMSE for each 24-bit candidate. Curves are conditional on the stated decoder and uniform ADC.",
    490,
)
rows = [["Exposure N", "Router winner", "Filter winner", "Router RMSE", "Filter RMSE"]]
for n in [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]:
    r = row_at("router", n)
    f = row_at("filter", n)
    rows.append([f"{n:,}", pair(r), pair(f), f"{r.rmse:.4f}", f"{f.rmse:.4f}"])
table(rows, [87, 105, 105, 97, 96])
p(
    "These are descriptive minima over test-scene means, not the performance of a design selected without test information. Near-ties require paired comparisons: the numerical winner alone does not establish a meaningful advantage. The full dataset includes bootstrap winner support and paired 3-versus-4-channel contrasts.",
    True,
)

page()
h("4. The first result was decoder-dependent")
image(
    "02_decoder_sensitivity.png",
    "Figure 3. Preferred allocations under four reconstruction methods with the same version-2 data split. Cells report lowest mean test RMSE.",
    490,
)
p(
    "The initial pilot suggested a simple 3x8-to-4x6 transition as light increased. That statement used a decoder trained on noiseless measurements. Once training includes the actual noise and quantization, the allocation map changes. A clean-trained inverse can amplify weak-channel noise; a matched estimator can trade bias against that noise using the training prior."
)
p(
    "This is a consequential finding: a channel/bit recommendation cannot be separated from the reconstruction procedure. It also prevents treating an apparent moderate-K optimum from one decoder as a general physical law. The legacy pilot is retained for traceability, not pooled with the revised results."
)
p(
    "The quadratic control adds squared measurements and adjacent-channel products. It is a controlled nonlinear baseline, not an attention model or new neural architecture. Interpolation tests how much of the result depends on learned spectral correlations."
)
p(
    "Neither more channels nor more bits is always best. More channels reduce spectral averaging but reduce photons per output and code depth. A better decoder changes the balance; it does not repeal photon statistics or restore arbitrary missing spectral detail."
)

page()
h("5. Calibration and sensor sensitivity")
image(
    "03_sensor_quantizer_sensitivity.png",
    "Figure 4. One-factor sensitivity tests plus one combined stress condition. Entries are descriptive winners under the matched affine decoder.",
    490,
)
p(
    "Changing ADC headroom or the code distribution changes how effectively low bit depths represent the signal. Read noise penalizes measurements with small counts. Router crosstalk mixes neighboring channels while conserving total photons; the matched decoder is calibrated to the known mixing."
)
p(
    "The stress values are modeling choices, not properties measured from a fabricated device. The declining-throughput curve drops from 84% at three channels to 60% at 24 channels. It is intentionally hypothetical and must not be presented as the result of an electromagnetic design or a measured device family.",
    True,
)

page()
h("6. Does selection help on unseen scenes?")
image(
    "05_honest_design_selection.png",
    "Figure 5. Allocation selected using validation scenes, then tested on a different scene. The null predictor outputs the training mean spectrum without seeing the measurements.",
    490,
)
image(
    "06_selection_gain_uncertainty.png",
    "Figure 6. Paired RMSE differences against fixed 3x8. Intervals crossing zero do not resolve the sign of improvement within this bootstrap analysis.",
    450,
)
p(
    "A test-oracle winner is an analysis aid, not a deployable algorithm. The validation selector is the actual procedure tested here. Its errors, including occasions when it chooses worse than fixed 3x8, remain in the results. Near-zero-light conditions are retained and compared with the null predictor, rather than removed to improve the narrative.",
    True,
)

page()
h("7. What does routing add?")
image(
    "07_router_advantage.png",
    "Figure 7. Router improvement after each optical mode selects an allocation using validation scenes. Shading is a paired scene-bootstrap interval.",
    480,
)
rows = [["N", "Router RMSE change (%)", "Bootstrap interval (%)"]]
for n in [30, 100, 1000, 10000, 100000]:
    g = gain[gain.photons == n].iloc[0]
    rows.append(
        [f"{n:,}", f"{g.router_gain_percent:.1f}", f"{g.ci_low:.1f} to {g.ci_high:.1f}"]
    )
table(rows, [100, 195, 195])
p(
    "Positive change here means lower mean RMSE for the router. This is the benefit under the explicit throughput and equal-area model, not a comparison against measured commercial-camera output. At high exposure, spectral representation and quantization can dominate, reducing the benefit of recovering additional photons."
)
p(
    'The earlier phrase "a tenfold earlier transition" described two sampled pilot-grid winners. It did not estimate an exact physical transition or establish its confidence interval. Because matched reconstruction changes the map, it is not the headline conclusion of this revised study.'
)

page()
h("8. Budget sensitivity and a compact rule")
image(
    "04_total_budget_sensitivity.png",
    "Figure 8. The optimum is conditional on total output budget as well as light and optical model.",
    485,
)
p(
    "A practical designer can select a candidate by simulating or calibrating validation spectra at the intended exposure and choosing the lowest expected error. To reduce the calibration sweep, we additionally fit the approximate risk C<sub>K</sub> + A<sub>K</sub>/N at only three validation exposures: 30, 1,000 and 30,000."
)
image(
    "09_compact_design_rule.png",
    "Figure 9. Test error from the three-anchor risk surrogate versus the full validation sweep and fixed 3x8. The surrogate is an approximation, not a proven optimum.",
    475,
)
p(
    "The model is motivated by a representation/quantization floor plus inverse-exposure shot-noise error for a fixed affine decoder. Clipping, low-bit thresholds and noise-dependent fitted decoders violate its simplest assumptions. Its success or failure must be judged from held-out error, not the appeal of the formula.",
    True,
)

page()
h("9. Does spectral complexity predict channel count?")
image(
    "08_complexity_association.png",
    "Figure 10. Exploratory association between test-scene effective rank and its test-oracle high-light channel count. These are not inputs to the deployable selector.",
    485,
)
rows = [["Decoder", "Raw-rank rho", "Permutation p", "Shape-rank rho"]]
for dec in ["clean", "matched", "quadratic", "interpolation"]:
    a = cc[(cc.decoder == dec) & (cc.rank_statistic == "raw_effective_rank")].iloc[0]
    b = cc[(cc.decoder == dec) & (cc.rank_statistic == "shape_effective_rank")].iloc[0]
    rows.append(
        [
            dec,
            f"{a.spearman_rho:.3f}",
            f"{a.permutation_p_unadjusted:.3f}",
            f"{b.spearman_rho:.3f}",
        ]
    )
table(rows, [145, 115, 115, 115])
p(
    "The raw effective rank includes overall brightness variation. The shape-only statistic first normalizes spectra by their spectral sum. We report both because they answer different questions. Correlation with a test-oracle winner can be interesting but does not demonstrate a deployable intrinsic-dimension rule or causal dependence."
)
p(
    "The original pilot correlation was based on fifteen scenes and one reconstruction protocol. These follow-up tests are exploratory, have tied channel counts, and are not corrected for the multiple related comparisons. A stable dimensionality-based design rule needs independent datasets and out-of-sample prediction of allocation, not merely an association measured after reconstruction."
)
p(
    "In the revised matched-affine experiment, raw effective rank has rho=0.355 with unadjusted permutation p=0.198. The original stronger correlation therefore does not survive this change of split, pixel sample and reconstruction protocol. We do not claim that intrinsic rank predicts the optimal channel count."
)

page()
h("10. What the reconstructions look like")
image(
    "10_image_reconstructions.png",
    "Figure 11. A held-out balloons scene, spatially subsampled for display. All candidates use the same display gain and pseudo-color channels. This is a numerical reconstruction, not a hardware photograph.",
    490,
)
p(
    "A one-bit ADC is not automatically an informative photon counter. With white-reference thresholds, dark channels can produce identical codes for many distinct spectra. The decoder then falls back toward a learned prior. More light can reduce the randomness that occasionally crosses a threshold without adding new code levels, so a coarse-ADC error curve need not improve monotonically."
)
p(
    "The package also contains four fixed-grid pixel spectra and numerical arrays for these reconstructions. These examples explain appearance but are not substituted for the full scene-held-out quantitative evaluation.",
    True,
)

page()
h("11. Discussion, limitations and next validation")
p(
    "<b>Answer to the question.</b> There is no decoder-independent allocation established by this study. Under the specified constraints, allocation should be selected jointly with decoder calibration and evaluated on separate scenes. Photon-efficient routing changes the feasible error trade-off, while ADC settings and spectral priors can materially change the preferred K."
)
p(
    "<b>What remains outside this result.</b> The data cover only fifteen supplied laboratory scenes. There is no second independent dataset, real optical transfer matrix, measured detector calibration or fabricated sensor. Spatial blur, demosaicing, unknown illumination, angular response, sensor area-dependent read noise and power are not optimized. Constant throughput and exposure-dependent ADC gain simplify real hardware."
)
p(
    "<b>Originality.</b> Sampling-versus-bit-depth trade-offs are established [1-3], as are spectral mosaic optimization [4], photon routing [5] and optical channel-scaling analysis [6]. Application-specific spectral channel selection also has direct precedents [7]. The differentiating research question is the joint-budget routed/filter comparison with decoder-aware validation and quantified failure of overly simple design rules. A targeted search did not establish worldwide priority."
)
p(
    "<b>Next decisive test.</b> Freeze the protocol and repeat on an independently acquired spectral dataset using measured or published wavelength-dependent responses. That tests transferability. A full hardware project additionally requires an achievable fixed-area optical design and measured photon statistics."
)
h("References")
refs = [
    "[1] Laska & Baraniuk (2012). Regime Change: Bit-Depth versus Measurement-Rate in Compressive Sensing. IEEE TSP. doi:10.1109/TSP.2012.2194710. arxiv.org/abs/1110.3450.",
    "[2] Ozcelikkale & Ozaktas (2012). Representation of optical fields using finite numbers of bits. Optics Letters 37, 2193-2195. doi:10.1364/OL.37.002193.",
    "[3] Ozcelikkale & Ozaktas (2013). Beyond Nyquist sampling: a cost-based approach. JOSA A 30, 645-655.",
    "[4] Yasuma et al. (2008). Generalized Assorted Pixel Camera: Post-Capture Control of Resolution, Dynamic Range and Spectrum. Columbia technical report CUCS-061-08.",
    "[5] Zou et al. (2022). Pixel-level Bayer-type colour router based on metasurfaces. Nature Communications 13, 3288. doi:10.1038/s41467-022-31019-7.",
    "[6] Han et al. (2026). Scaling Limits of Multichannel Spectral Routers for Snapshot Imaging. arXiv:2607.23508v2. Preprint.",
    "[7] van Elst et al. (2025). Application-Specific Optimization of Integrated Spectral Sensors. ACS Photonics. doi:10.1021/acsphotonics.5c01213. Publisher excerpts/institutional record checked; full text not verified here.",
]
for ref in refs:
    p(ref, True)
p(
    "Research assistance disclosure: ChatGPT assisted with modeling, implementation, simulation, analysis, literature checking and drafting. Vansh Jain supplied the project direction. The student must understand and verify the work before submission. This is an AI-assisted computational research draft, not a peer-reviewed publication.",
    True,
)


def footer(c, doc):
    c.saveState()
    w, hh = A4
    c.setStrokeColor(colors.HexColor("#d7e2e7"))
    c.line(45, 39, w - 45, 39)
    c.setFont("VNS", 7.5)
    c.setFillColor(colors.HexColor("#52616b"))
    c.drawString(45, 27, "VNS | 15-scene computational study | 13 Oct 2026")
    c.drawRightString(w - 45, 27, str(doc.page))
    c.restoreState()


doc = SimpleDocTemplate(
    str(OUT / "VNS_research_draft.pdf"),
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=42,
    bottomMargin=50,
    title="Photon and bit allocation for routed snapshot spectral sensing",
    author="Vansh Jain; AI-assisted draft",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
(OUT / "VNS_research_draft.md").write_text("\n".join(md))
print("Saved PDF and editable Markdown draft")
