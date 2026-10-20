# MorphoEval report — audit `2026-10-19-butterfly-eyespot-focus-rings-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/98

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Nijhout (1990) & Brakefield et al. (1996) butterfly wing-vein focus determination and concentric French-flag eyespot ring morphogenesis (`2001-nijhout-butterfly-eyespot-focus-rings.py`) | 52 | 24 | 44 | 50 | 64 | 60 | 68 | 49.3 | 33.8 | 64.5 | 47.7 | **47.7** | partially faithful |

## Findings and remediation

### Nijhout (1990) & Brakefield et al. (1996) butterfly wing-vein focus determination and concentric French-flag eyespot ring morphogenesis  —  overall 47.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2001-nijhout-butterfly-eyespot-focus-rings.py` (consensus score 51.5/100). The v1 script plots only a 1D analytical exponential decay curve and a 2D circular disc sketch for a single parameter set, omitting Stage-1 2D Meinhardt wing-compartment vein-boundary reaction-diffusion focus determination (Nijhout 1990), Stage-2 2D focal diffusion-decay PDE relaxation, seasonal ecdysone polyphenism (wet vs dry season), *Goldeneye* Spalt-threshold shift, twin-focus hourglass fusion, pupal focal microcautery timing series, runtime SHA-256 dataset verification, and CV radial luminance/chroma ring segmentation on the vendored *Bicyclus anynana* ventral wing eyespot specimen photograph.

- **Model fidelity to source: 52** — Evaluates only the 1D analytical exponential formula c(r) = c0 * exp(-r/lambda) and a static 2D circle mask (L23-50), omitting Stage-1 2D Meinhardt activator-inhibitor reaction-diffusion from bounding wing veins to isolate the midline focus (Nijhout 1990), Stage-2 2D PDE relaxation, twin-focus superposition hourglass bridges, and pupal cautery timing series.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L23–50
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic plot (L53-86) without loading, SHA-256-verifying, or performing radial luminance and golden-chroma concentric ring segmentation on the vendored *Bicyclus anynana* specimen photograph (`datasets/butterfly_eyespot_focus_rings/bicyclus_anynana_eyespot_specimen.jpg`).  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L53–86; `docs/media/butterfly_eyespot_focus_rings_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes c0=1.0 and lambda_mm=0.48 (L23-50) without CLI parameter overrides, wet/dry seasonal ecdysone polyphenism sweep, *Goldeneye* threshold shift, twin-focus inter-focal distance sweep, or pupal cautery timing series.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L23–50
- **Morphometric validation: 50** — Reports only 3 analytical ring radii in `measure_morphoeval()` (L118-130) without validating Stage-1 midline focus convergence, wet/dry seasonal gold annulus invariance, *Goldeneye* gold-ring expansion, twin-focus bridge concentration, pupal cautery timing curve, or empirical *Bicyclus anynana* Cu1 specimen ring radii and proportions.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L118–130
- **Source and data traceability: 64** — Dataset directory `datasets/butterfly_eyespot_focus_rings/` exists on disk, but v1 does not load `butterfly_eyespot_focus_rings_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L1–22
- **Code-to-model traceability: 60** — Header docstring mentions Nijhout (1990) and Brakefield et al. (1996), but functions lack explicit mappings to published equations and to the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_098_butterfly_eyespot_focus_rings.lean`.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L23–50
- **Reproducibility package: 68** — Includes `--self-test` (L133-140), but checks only 2 basic assertions without testing SHA-256 dataset provenance, JSON parity, Stage-1 2D midline focus localization, seasonal gold annulus invariance, twin-focus hourglass fusion, specimen ring segmentation, or bitwise two-run determinism.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L133–140

Remediation:
- Implement the coupled two-stage 2D morphogenesis pipeline: Stage-1 2D Meinhardt activator-inhibitor reaction-diffusion inside a longitudinal wing-vein-bounded cell compartment [0, L_x] x [-W_y/2, W_y/2] that destabilizes the midline ridge into a discrete focal organizer at (x_f, 0) (Nijhout 1990), coupled to Stage-2 2D focal morphogen diffusion-decay and Spalt/Engrailed/Distal-less concentric ring induction across all 5 experimental regimes (Wet-Season Wild-Type, Dry-Season Polyphenism, Goldeneye Spalt-shift mutant, Twin-Focus Hourglass Fusion, and Early Pupal Focal Microcautery Ablation) plus the 6-point pupal cautery timing series.
- Load and SHA-256-verify `datasets/butterfly_eyespot_focus_rings/` at runtime and perform computer-vision Difference-of-Gaussians focus detection and radial luminance + golden-chroma concentric ring segmentation (`segment_bicyclus_eyespot_specimen`) on `bicyclus_anynana_eyespot_specimen.jpg`, measuring primary Cu1 focus coordinates (596, 231), pupil radius R_pupil = 3.663 px, black core radius R_black = 15.442 px, outer gold radius R_gold = 20.402 px, and black-to-gold ratio 0.7569.
- Expose full CLI parameter overrides (`--c0`, `--lambda-mm`, `--theta-black`, `--theta-gold`, `--d12`, `--seed`, `--skip-gif`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 2-panel Stage-1 focus + Stage-2 concentric ring morphogenesis GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Nijhout (1990) & Brakefield et al. (1996) butterfly wing-vein focus determination and concentric French-flag eyespot ring morphogenesis
- **spec_fingerprint** (semantic model + inputs): `0a2086f2436043ce9998603ef36e23feb6c621834d7d960a99310b296156772b`
- **input_dataset_sha256**: `datasets/butterfly_eyespot_focus_rings/bicyclus_anynana_eyespot_specimen.jpg` (`f1826cee44602782…`), `datasets/butterfly_eyespot_focus_rings/butterfly_eyespot_focus_rings_reference.json` (`b19590626bdaff79…`), `datasets/butterfly_eyespot_focus_rings/scout_manifest.json` (`7cfe9d2221ea5b8b…`)
- `docs/media/butterfly_eyespot_focus_rings_comparison.png` — artifact_sha256 `6ecb6d8e0b918529…`, 79,032 B, 1400×588 px

| measurement | value | reference | method |
|---|---:|---|---|
| Wet-season central pupil radius R_pupil | 0.1951 mm | 0.1951 | Analytical R_pupil = lambda * ln(c0 / theta_pupil) |
| Wet-season black melanin disc radius R_black | 0.8234 mm | 0.8234 | Analytical R_black = lambda * ln(c0 / theta_black) |
| Wet-season outer gold ring radius R_gold | 1.0883 mm | 1.0882 | Analytical R_gold = lambda * ln(c0 / theta_gold) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2001-nijhout-butterfly-eyespot-focus-rings.py` prior to Step 12 remediation in `2026-10-19-butterfly-eyespot-focus-rings-loop`.
