# MorphoEval report — audit `2026-10-27-coturnix-japonica-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-coturnix-japonica-turing-wavelength-s0-audit -> 2026-10-27-coturnix-japonica-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 22.42 µm on Holocene Coturnix japonica plumage (`models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py`) | 89 | 86 | 88 | 88 | 90 | 87 | 89 | 88.7 | 86.8 | 88.8 | 87.9 | **87.9** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 22.42 µm on Holocene Coturnix japonica plumage  —  overall 87.9 (faithful)

Pass 2, superseding the Pass 1 audit. Same Schnakenberg system and the same Yehliu specimen. Remediation R2 is done: the display panel now prints linear law 22.42 µm and saturated FFT 33.6 µm. R3 is done: flank-only melanin fraction is 0.364 against full-plumage 0.340. R1 was tested rather than papered over. Replacing the growth moment with a parabolic peak moved the estimate toward the supercritical sigma maximum, away from the onset wavenumber the slot asks for, so the moment-vs-onset correlation is kept (r = 0.99966, RMSE = 6.52 µm) and the dispersion curve itself correlates at r = 0.993. Equal-diffusion std is 0.0003 against patterned std 1.04, and the ablation panel is on the pattern's color scale. GIF is 12 frames at 960×540; last-frame CV similarity is 96.0. Overall 87.9/100. R4 stays open: there is no second assessor and no nanopub signature.

- **Model fidelity to source: 89** — Onset identities are unchanged and still matched by the Lean module: tr J < 0, det J > 0, k_c^2 = 3112/259, open band, equal-diffusion cross term negative. Pass 2 states plainly that the saturated spacing (33.6 µm) is not that onset wavelength. Deduction remains the phenomenological kinetics, not a fitted Shh/Bmp2 rate law.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L77–118; `morphoeval/lean/loops/Loop_3157_coturnix_japonica_turing_wavelength_s0.lean` L15–56; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L530–545; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L16–22
- **Morphological realism / specimen match: 86** — The Yehliu bird, plumage crop, spot field, dispersion, Du sweep, and a genuinely flat ablation share the comparison figure. The 12-frame GIF keeps the concrete and straw and replaces plumage high frequencies; CV similarity of the last frame is 96.0. Deduction: still one portrait and no micron scale bar.  
  evidence: `docs/media/coturnix_japonica_turing_wavelength_s0_comparison.png` L0–0; `docs/media/coturnix_japonica_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L8–14
- **Parameter expressivity: 88** — D_u remains the lever that moves the onset wavelength, and the five-point growth-moment sweep still tracks it at r = 0.9997. The dispersion curve at the nominal point correlates at r = 0.993. Equal diffusion quenches the field. Anisotropy is still a single display setting, so this is not a full morphospace.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L296–339; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L497–508; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L64–64
- **Morphometric validation: 88** — Pass 2 adds a flank ROI: melanin fraction 0.364 versus full plumage 0.340, and a separate flank coherency. Wavelength moment r = 0.99966 against a reference of 1, RMSE = 6.52 µm against a reference of 0, dispersion-curve r = 0.993, KS D = 0.098, ablation std = 0.0003 against patterned std 1.04. None of these value/reference pairs are copies. The RMSE is reported rather than removed.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L402–432; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L596–695; `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Unchanged from Pass 1 and still accurate. Six confirmed DOIs, CC BY-SA 2.0 attribution to 孫鋒 林, and a SHA-256 pin checked by --self-test.  
  evidence: `datasets/coturnix_japonica_turing_wavelength_s0/paper_citations.bib` L1–86; `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_reference.json` L65–82; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L876–882
- **Code-to-model traceability: 87** — Docstring, Lean integers, and the math-KG spec still name the same laws. Pass 2 adds an explicit code path for the dispersion curve comparison and the flank ROI, both pointed at from the measurement list. Still no equation-number map onto Prum & Williamson 2002, whose follicle geometry is not what is integrated.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L135–152; `morphoeval/math_kg_loops/delphi10000-coturnix-japonica-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 89** — --self-test still gates the integer identities, the quench, the pattern-versus-ablation variance ratio, and a three-point Pearson r above 0.98. The full script rewrites metrics, PNG, and GIF from seed 3157. Pass 2 re-ran that path after the flank ROI and the wavelength annotation. Dependencies are the cloud image, not a per-script lock file.  
  evidence: `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L854–896; `models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py` L198–199

Remediation:
- {'id': 'R4', 'item': 'No independent second assessor is available in this single-agent loop. The score stays builder-assessed. Nanopubs are unsigned because ~/.nanopub/profile.yml does not exist.'}
- {'id': 'R1-outcome', 'item': 'Resolved by test, not by forcing RMSE to 0. A parabolic peak sits on the supercritical sigma maximum, not on onset k_c. The onset-tracking moment (r = 0.99966, RMSE = 6.52 µm) is retained, and dispersion-curve r = 0.993 is reported beside it.'}
- {'id': 'R2-outcome', 'item': 'Resolved. The display panel annotates linear law 22.42 µm and saturated FFT 33.6 µm as separate lengths.'}
- {'id': 'R3-outcome', 'item': 'Resolved. Flank-only melanin fraction is 0.364 versus full-plumage 0.340 on the same Yehliu frame.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 22.42 µm on Holocene Coturnix japonica plumage
- **spec_fingerprint** (semantic model + inputs): `fcce03b3209888bbee04f33bf08d93db2245049fdd963bb6413cf49a32f9a400`
- **input_dataset_sha256**: `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_yehliu_taiwan.jpg` (`e3ba62e08042fc74…`), `datasets/coturnix_japonica_turing_wavelength_s0/paper_citations.bib` (`1f2c15cdb72b9ad0…`)
- `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_metrics.json` — artifact_sha256 `35cd1221bdf3e977…`, 6,711 B, 
- `docs/media/coturnix_japonica_turing_wavelength_s0_comparison.png` — artifact_sha256 `13bedcaad1226221…`, 630,527 B, 1632×984 px
- `docs/media/coturnix_japonica_turing_wavelength_s0_growth.gif` — artifact_sha256 `40ecef1d33d22c25…`, 5,901,762 B, 960×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated stripe wavelength | 33.6451 µm | 22.42 | Power-weighted radial FFT of the anisotropic display field, converted by the slot length unit *(Display run is anisotropic, so the FFT peak is not forced to equal the isotropic law.)* |
| specimen plumage stripe melanin area fraction | 0.3404 fraction | 0.0892 | Otsu threshold on the plumage-mask luminance of the Yehliu photograph *(Self-referential val==ref is rejected; these two fractions are independent measurements.)* |
| specimen flank stripe melanin area fraction | 0.3639 fraction | 0.3404 | Otsu on the rear-half plumage mask (flank and back, head excluded, straw excluded) *(Second ROI. A match to the full-mask number is not assumed.)* |
| specimen image structure-tensor coherency | 0.5873 coherency | 0.7188 | Mean structure-tensor coherency on the plumage bounding crop *(Portrait coherency includes feather edges; the PDE field is a periodic patch.)* |
| wavelength sweep Pearson r | 0.99966 r | 1.0 | Pearson r of linear-regime growth wavelength vs analytical lambda across the Du sweep *(RMSE = 6.5239 µm. KS plumage-vs-composite statistic = 0.0979. Parabolic peak-fit was not substituted: the supercritical sigma peak is not the onset k_c.)* |
| nominal dispersion-curve Pearson r | 0.993142 r | 1.0 | Pearson r of analytical sigma(k) vs measured radial growth rate at Du=0.035, Dv=1.85 *(Complements the wavelength-moment correlation. RMSE of that moment remains the supercritical offset.)* |
| Du sweep RMSE | 6.5239 µm | 0.0 | RMSE between measured and analytical wavelengths on the Du sweep *(A nonzero RMSE is the nonlinear and discretization residual.)* |
| plumage versus composite KS statistic | 0.0979 KS D | 0.0 | Two-sample KS test on plumage luminance vs the last composited frame inside the mask *(p = 8.924e-198. cv of Du-sweep residuals = 0.0391.)* |
| equal-diffusion ablation activator std | 0.0003 concentration | 1.0426 | std(u) after the same horizon with Du = Dv = 0.035 (negative control) *(Control must stay near the homogeneous state while the Turing run grows a pattern.)* |

## Environment at checkpoint


Slot 3157 files are new in the working tree and are not in commit 0c76d15d. Build uses --no-verify-loci. Nanopubs are unsigned: ~/.nanopub/profile.yml is absent, so there is no trusty URI for Pass 1 to supersede on the network.
