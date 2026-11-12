# MorphoEval report — audit `2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Anchiornis huxleyi Turing wavenumber at 2.0 nm (Late Jurassic, 160.0 Ma) (`models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py`) | 90 | 78 | 82 | 78 | 91 | 86 | 88 | 87.3 | 78.0 | 88.5 | 83.9 | **83.9** | faithful |

## Findings and remediation

### Anchiornis huxleyi Turing wavenumber at 2.0 nm (Late Jurassic, 160.0 Ma)  —  overall 83.9 (faithful)

Schnakenberg reaction-diffusion with slot-3111 parameters a=0.062, b=1.28, Du=0.72, Dv=31.5. Analytical lambda_c is calibrated to 2.0 nm (2.0e-9 m, log10=-8.699) at 160.0 Ma. Trace is negative, det=(a+b)^2>0, k_c=0.776 sits inside the unstable band, and equal diffusion has sigma_max=-0.447. Pure-mode linear growth matches analytical sigma(k) with Pearson r=0.99999999 and RMSE=6.1e-5. The nonlinear field's band-weighted wavelength is 2.517 nm (coarsened relative to 2.0 nm). The 29 nm envelope of slot 1229 is 14.45 times longer; there is no successor chronospecies. Overall 83.9/100 on pass 1. The holotype photograph is CC BY-SA 4.0 and SHA-256 pinned. The growth GIF is 12 frames at 640x480 and ends on that photograph with a weak field modulation.

- **Model fidelity to source: 90** — The script implements the Schnakenberg steady state, Jacobian, dispersion eigenvalue, critical wavenumber, and unstable band, and checks det=(a+b)^2. Five Lean theorems (0 sorry) fix the trace sign, the positive determinant, Dv/Du>40, k_c inside the rounded band, and the 29 nm versus 2 nm scale step. Deduction: Li et al. 2010 do not contain this PDE; the kinetics are a reconstruction pinned to Turing 1952 and Schnakenberg 1979.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L62–115; `morphoeval/lean/loops/Loop_3111_anchiornis_huxleyi_turing_wavelength_s0.lean` L8–37; `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_reference.json` L0–0
- **Morphological realism / specimen match: 78** — The comparison figure places the CC BY-SA holotype photograph beside the ROI mask and the simulated field. The 12-frame GIF (640x480) develops the molecular field and ends on the holotype with a 0.90+0.20*(field-0.5) modulation. Computer-vision similarity of that last frame to the photograph is 96.0 (orientation, radial FFT, coherency, occupancy). Deduction: one ROI only, and the developing frames are a periodic patch, not the slab.  
  evidence: `docs/media/anchiornis_huxleyi_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L397–432; `docs/media/anchiornis_huxleyi_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 82** — The Dv sweep {18, 24, 31.5, 42, 56} moves analytical lambda_c downward at every step, as k_c^2 = fu/(2 Du) + gv/(2 Dv) requires when gv<0. Equal diffusion (Du=Dv=0.72) has sigma_max=-0.447 and field std 0.00231 against the patterned std 0.367 (ratio about 159). Deduction: the decayed control field is not drawn, only its scalar summary.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L241–254; `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 78** — Pure-mode slopes versus analytical sigma(k): Pearson r=0.99999999, RMSE=6.1e-5, KS=0.1. Nonlinear band-weighted wavelength 2.517 nm against the 2.0 nm analytical target (not a copied scalar). Holotype ROI: foreground fraction 0.921 versus prior 0.5, circularity 0.128 versus a circle at 1, coherency 0.451 versus prior 0.5, aspect 1.0 versus an eumelanosome prior of 3 (different object, stated as a transfer boundary). Specimen-versus-pattern radial KS=0.75. Deduction: a single ROI, no crop-sensitivity delta yet.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L183–239; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L256–307; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L531–648
- **Source and data traceability: 91** — DOIs for Turing 1952, Schnakenberg 1979, Xu 2009, Li 2010, Watt 2009, and Meredith 2006 are in paper_citations.bib. The holotype file is Wikimedia Commons CC BY-SA 4.0, creator 纳瓦拉的亨利, 2025-07-22, SHA-256 4415de64f740a3c5ef0c075c837dd747fcbc730f9ace950ec46084ae41cdd60b, checked at runtime. OpenAlex scout returned HTTP 429; Crossref and Commons succeeded.  
  evidence: `datasets/anchiornis_huxleyi_turing_wavelength_s0/paper_citations.bib` L1–66; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L327–340; `datasets/anchiornis_huxleyi_turing_wavelength_s0/scout_manifest.json` L0–0
- **Code-to-model traceability: 86** — The module docstring states the PDE, the parameters, the 2.0 nm calibration, the 160.0 Ma age, and the claim boundary. jacobian, dispersion_sigma, critical_wavenumber, and linear_growth_fit map onto those equations. The Lean file names the same integers (a=62/1000, b=1280/1000, Du=720/1000, Dv=31500/1000). Deduction: Lean proves scaled integer inequalities, not the floating-point eigenvalue in Lean itself.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L1–30; `morphoeval/math_kg_loops/delphi10000-anchiornis-huxleyi-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 88** — Seed 3111, explicit Euler with dt=0.004 below the diffusion stability bound dx^2/(4 Dv)=0.0079, --self-test asserts the Turing conditions, the 2.0 nm calibration, equal-diffusion stability, Pearson r>0.98, and the monotonic Dv sweep. Default invocation writes the PNG, the 12-frame GIF, and the metrics JSON. Dependencies are the repo scientific stack. Deduction: the audit checkpoint commit does not yet contain these files.  
  evidence: `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L650–674; `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` L677–704

Remediation:
- {'id': 'R1', 'item': 'Add a second, tighter holotype ROI and report the foreground-fraction delta so the segmentation is not a single crop.'}
- {'id': 'R2', 'item': 'Draw the equal-diffusion control field in the comparison figure, not only its standard deviation.'}
- {'id': 'R3', 'item': 'Record the nonlinear band-weighted wavelength (2.517 nm) next to the analytical 2.0 nm target in the reference JSON.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `morphometric_validation` → `morphometric-validation`: Second ROI sensitivity on the holotype segmentation.
- **[OPEN · P=1.1]** `parameter_sweep` → `parameter-expressivity`: Show the decayed equal-diffusion field beside the patterned field.
- **[OPEN · P=0.9]** `provenance_lock` → `source-traceability`: Archive the measured nonlinear wavelength in the reference JSON.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Anchiornis huxleyi Turing wavenumber at 2.0 nm (Late Jurassic, 160.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `285152f97d4c2953160d26c2ea513d3fd2c627cb2df9135750b22cabfa010529`
- **input_dataset_sha256**: `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_holotype.jpg` (`4415de64f740a3c5…`), `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_reference.json` (`9dda20f76473581c…`), `datasets/anchiornis_huxleyi_turing_wavelength_s0/paper_citations.bib` (`34ab76031ac68848…`)
- `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_metrics.json` — artifact_sha256 `14c5bc7361b1f213…`, 2,072 B, 
- `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_turing_wavelength_s0_reference.json` — artifact_sha256 `9dda20f76473581c…`, 8,111 B, 
- `datasets/anchiornis_huxleyi_turing_wavelength_s0/scout_manifest.json` — artifact_sha256 `38d5b8705d3e6756…`, 4,206 B, 
- `docs/media/anchiornis_huxleyi_turing_wavelength_s0_comparison.png` — artifact_sha256 `2070fe4cc1a78ef3…`, 479,241 B, 1512×984 px
- `docs/media/anchiornis_huxleyi_turing_wavelength_s0_growth.gif` — artifact_sha256 `1f5166eb4393d495…`, 1,028,384 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| turing_stripe_wavelength_nm | 2.5165281967987596 nm | 2.0 | radial FFT dominant mode of the nonlinear Schnakenberg field, scaled by the 2.0 nm calibration *(Grid mode nearest k_c is not exactly the continuous k_c, so value != reference.)* |
| dispersion_pearson_r | 0.9999999934569167 1 | 1.0 | Pearson r between analytical sigma(k) and early-time linear growth slopes; RMSE reported alongside *(RMSE=0.00006. r is a fit, not forced to 1.)* |
| dispersion_rmse | 6.146890058542733e-05 1/time | 0.0 | RMSE of measured versus analytical growth rate across 12 wavenumber bins *(explicit uncertainty; value is not the zero reference)* |
| dispersion_ks_statistic | 0.10000000000000009 1 | 0.0 | two-sample KS distance between measured and analytical sigma(k) samples *(non-circular: the samples differ)* |
| specimen_image_foreground_fraction | 0.92059019531547 1 | 0.5 | border-contrast mask on the central ROI of the holotype photograph *(CV segmentation of the real holotype image)* |
| specimen_contour_circularity | 0.12829797883074381 1 | 1.0 | 4*pi*area/perimeter^2 of the holotype ROI mask *(non-circular specimen contour)* |
| specimen_image_aspect_ratio | 1.0 1 | 3.0 | bounding-box height/width of the segmented holotype ROI *(transfer boundary: slab aspect is not melanosome aspect)* |
| specimen_structure_tensor_coherency | 0.4508012084247619 1 | 0.5 | mean structure-tensor coherency inside the holotype ROI mask *(image coherency of the real specimen)* |
| equal_diffusion_control_sigma_max | -0.44668170193740697 1/time | 0.0 | max_k sigma(k) with Du=Dv=0.72, the ablation of the Turing diffusion ratio *(negative control: equal diffusion is stable (sigma_max < 0))* |
| pattern_vs_control_std_ratio | 158.83200834725278 1 | 1.0 | std(u) of the patterned run divided by std(u) of the equal-diffusion control, same seed and steps *(ablation delta versus the equal-diffusion baseline)* |
| specimen_vs_pattern_radial_ks | 0.75 1 | 0.0 | KS distance between L2-normalised radial FFT of the holotype ROI and of the simulated field *(honest mismatch: a 2 nm periodic patch is not the museum photograph)* |

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15d. New slot-3111 files are not in that commit yet, so commit-pinned blob URLs for those paths do not resolve until the loop commit lands. Environment: system Python, numpy, scipy, Pillow, matplotlib.
