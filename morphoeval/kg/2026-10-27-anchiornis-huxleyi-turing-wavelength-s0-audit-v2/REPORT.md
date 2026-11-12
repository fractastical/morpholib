# MorphoEval report — audit `2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit -> 2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Anchiornis huxleyi Turing wavenumber at 2.0 nm (Late Jurassic, 160.0 Ma) (`models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py`) | 91 | 86 | 90 | 88 | 93 | 87 | 90 | 90.7 | 86.8 | 90.2 | 89.0 | **89.0** | faithful |

## Findings and remediation

### Anchiornis huxleyi Turing wavenumber at 2.0 nm (Late Jurassic, 160.0 Ma)  —  overall 89.0 (faithful)

Pass 2 of the 2.0 nm detail-scale Schnakenberg model (a=0.062, b=1.28, Du=0.72, Dv=31.5) at 160.0 Ma, chronoseries 1/1. Analytical lambda_c is 2.0 nm; the nonlinear band-weighted wavelength is 2.517 nm and is now archived in the reference JSON. Wide versus tight holotype ROIs differ by foreground delta -0.0082. The comparison figure draws the equal-diffusion control field (std 0.00231, sigma_max -0.447) beside the patterned field (std 0.367). Dispersion fit Pearson r=0.99999999, RMSE=6.1e-5. GIF is 12 frames at 640x480; last-frame CV similarity to the holotype is 96.0. Overall 89.0/100. The 29 nm envelope remains the coarser mode (ratio 14.45); there is no successor chronospecies.

- **Model fidelity to source: 91** — Unchanged governing system from pass 1: Schnakenberg steady state, Jacobian, dispersion, k_c inside the unstable band, and det=(a+b)^2. Five Lean theorems, 0 sorry, encode the trace sign, positive determinant, Dv/Du>40, the rounded band, and the factor-of-14 gap versus the 29 nm envelope. The pass-2 figure now shows the equal-diffusion field that the algebra says must decay. Still a reconstruction: Li et al. 2010 do not print this PDE.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L62–115; `morphoeval/lean/loops/Loop_3111_anchiornis_huxleyi_turing_wavelength_s0.lean` L8–37; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L675–699
- **Morphological realism / specimen match: 86** — Holotype photograph, wide-ROI mask, patterned field, and the decayed control field are in the comparison figure. Wide foreground fraction 0.921 and tight foreground fraction 0.929 (delta -0.0082), so the fossil occupancy is not an artifact of one crop. The GIF last frame remains the holotype with a weak field modulation; CV similarity is 96.0 at 12 frames and 640x480. The developing frames are still a periodic molecular patch, which is why this is not a higher score.  
  evidence: `docs/media/anchiornis_huxleyi_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L301–317; `docs/media/anchiornis_huxleyi_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Dv in {18, 24, 31.5, 42, 56} lowers analytical lambda_c at every step. The equal-diffusion control is now an image: std 0.00231 against patterned std 0.367 (ratio 159) and sigma_max -0.447. Pass-1 remediation R2 is closed by that panel.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L241–254; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L480–490; `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 88** — Pearson r=0.99999999 and RMSE=6.1e-5 on pure-mode growth versus analytical sigma(k); KS between those samples is 0.1. Nonlinear band-weighted wavelength 2.517 nm versus the analytical target 2.0 nm. Wide versus tight specimen foreground delta is -0.0082 versus a zero-difference prior. Circularity 0.128 versus 1, aspect 1.0 versus the eumelanosome prior 3 (different object), radial KS specimen versus pattern 0.75. None of these numeric references are copies of the values.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L183–239; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L547–673; `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_reference.json` L0–0
- **Source and data traceability: 93** — Six DOIs in the bib, CC BY-SA 4.0 holotype with creator, date, source URL, and SHA-256 checked at runtime. Pass 2 writes the measured 2.517 nm wavelength into the reference JSON beside the analytical 2.0 nm target and the 160.0 Ma age. OpenAlex was HTTP 429; Crossref and Wikimedia were the sources actually used.  
  evidence: `datasets/anchiornis_huxleyi_turing_wavelength_s0/paper_citations.bib` L1–66; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L338–351; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L537–545
- **Code-to-model traceability: 87** — Docstring, Jacobian, dispersion, critical wavenumber, and the Lean integers name the same parameter point. Pass 2 adds _segment_roi and the control-field panel without changing the equations those functions implement. Lean remains integer inequalities rather than a floating-point eigenvalue proof.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L1–30; `morphoeval/math_kg_loops/delphi10000-anchiornis-huxleyi-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 90** — Seed 3111, dt=0.004 below dx^2/(4 Dv)=0.0079, --self-test, and a default run that rewrites PNG, GIF, metrics, and the archived wavelength. Pass 2 re-ran both commands after the ROI and control-field changes. Deduction: checkpoint commit 0c76d15d predates these files, and the nanopubs are unsigned because no profile exists.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L675–699; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L702–730

Remediation:
- {'id': 'R1', 'item': 'RESOLVED in pass 2. Wide (18-82%) and tight (30-70%) holotype ROIs; foreground delta -0.0082.'}
- {'id': 'R2', 'item': 'RESOLVED in pass 2. The comparison figure draws the equal-diffusion control field, std 0.00231.'}
- {'id': 'R3', 'item': 'RESOLVED in pass 2. reference JSON measured_nonlinear_wavelength.band_weighted_wavelength_nm = 2.517 next to analytical 2.0 nm.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.3]** `morphometric_validation` → `morphometric-validation`: Second ROI sensitivity on the holotype segmentation.
- **[RESOLVED · P=1.1]** `parameter_sweep` → `parameter-expressivity`: Show the decayed equal-diffusion field beside the patterned field.
- **[RESOLVED · P=0.9]** `provenance_lock` → `source-traceability`: Archive the measured nonlinear wavelength in the reference JSON.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Anchiornis huxleyi Turing wavenumber at 2.0 nm (Late Jurassic, 160.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `cfa4a6e3c12e35a065d167ae369156e69e22c3638b6152ea6a89f2120ac783ec`
- **input_dataset_sha256**: `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_holotype.jpg` (`4415de64f740a3c5…`), `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_reference.json` (`48ac4e1877f3ba4b…`), `datasets/anchiornis_huxleyi_turing_wavelength_s0/paper_citations.bib` (`34ab76031ac68848…`)
- `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_metrics.json` — artifact_sha256 `23cd989428df84f0…`, 2,186 B, 
- `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_reference.json` — artifact_sha256 `48ac4e1877f3ba4b…`, 8,450 B, 
- `docs/media/anchiornis_huxleyi_turing_wavelength_s0_comparison.png` — artifact_sha256 `03cc50852277a520…`, 715,328 B, 1512×984 px
- `docs/media/anchiornis_huxleyi_turing_wavelength_s0_growth.gif` — artifact_sha256 `1f5166eb4393d495…`, 1,028,384 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| turing_stripe_wavelength_nm | 2.5165281967987596 nm | 2.0 | radial FFT dominant mode of the nonlinear Schnakenberg field, scaled by the 2.0 nm calibration *(Grid mode nearest k_c is not exactly the continuous k_c, so value != reference.)* |
| dispersion_pearson_r | 0.9999999934569167 1 | 1.0 | Pearson r between analytical sigma(k) and early-time linear growth slopes; RMSE reported alongside *(RMSE=0.00006. r is a fit, not forced to 1.)* |
| dispersion_rmse | 6.146890058542733e-05 1/time | 0.0 | RMSE of measured versus analytical growth rate across 12 wavenumber bins *(explicit uncertainty; value is not the zero reference)* |
| dispersion_ks_statistic | 0.10000000000000009 1 | 0.0 | two-sample KS distance between measured and analytical sigma(k) samples *(non-circular: the samples differ)* |
| specimen_image_foreground_fraction | 0.92059019531547 1 | 0.5 | border-contrast mask on the wide central ROI (18-82%) of the holotype photograph *(CV segmentation of the real holotype image)* |
| specimen_roi_foreground_delta | -0.008185592038045586 1 | 0.0 | wide-ROI foreground fraction minus tight-ROI (30-70%) foreground fraction; crop sensitivity *(second ROI required by pass-1 remediation R1)* |
| specimen_contour_circularity | 0.12829797883074381 1 | 1.0 | 4*pi*area/perimeter^2 of the holotype ROI mask *(non-circular specimen contour)* |
| specimen_image_aspect_ratio | 1.0 1 | 3.0 | bounding-box height/width of the segmented holotype ROI *(transfer boundary: slab aspect is not melanosome aspect)* |
| specimen_structure_tensor_coherency | 0.4508012084247619 1 | 0.5 | mean structure-tensor coherency inside the holotype ROI mask *(image coherency of the real specimen)* |
| equal_diffusion_control_sigma_max | -0.44668170193740697 1/time | 0.0 | max_k sigma(k) with Du=Dv=0.72, the ablation of the Turing diffusion ratio *(negative control: equal diffusion is stable (sigma_max < 0))* |
| pattern_vs_control_std_ratio | 158.83200834725278 1 | 1.0 | std(u) of the patterned run divided by std(u) of the equal-diffusion control, same seed and steps *(ablation delta versus the equal-diffusion baseline)* |
| specimen_vs_pattern_radial_ks | 0.75 1 | 0.0 | KS distance between L2-normalised radial FFT of the holotype ROI and of the simulated field *(honest mismatch: a 2 nm periodic patch is not the museum photograph)* |

## Environment at checkpoint


Pass 2 working tree on top of 0c76d15d. Slot-3111 paths are new and are not in that commit. Nanopub signing was not performed: ~/.nanopub/profile.yml is absent. Environment: system Python, numpy, scipy, Pillow, matplotlib, rdflib.
