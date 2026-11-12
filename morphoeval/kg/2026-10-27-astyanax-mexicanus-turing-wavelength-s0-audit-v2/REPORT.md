# MorphoEval report — audit `2026-10-27-astyanax-mexicanus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-astyanax-mexicanus-turing-wavelength-s0-audit -> 2026-10-27-astyanax-mexicanus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing wavenumber at 666 nm in the Astyanax cave morph (`models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py`) | 88 | 78 | 86 | 88 | 90 | 86 | 86 | 87.3 | 81.8 | 87.3 | 85.1 | **85.1** | faithful |

## Findings and remediation

### Turing wavenumber at 666 nm in the Astyanax cave morph  —  overall 85.1 (faithful)

Pass 2. Same Schnakenberg carrier and 6.656e-7 m calibration as pass 1 (Holocene cave morph, 0 Ma). Dispersion Pearson r remains 0.99997 (RMSE 0.00455). Nonlinear wavelength 660.7 nm. The Dv/Du sweep on {10, 12, 16, 20, 24, 30} is monotonic: selected wavelength rises from 15.74 px to 19.88 px. Du = Dv = 2 suppresses the band (sigma_max = -0.1). The voucher flank and the patch differ by KS = 0.105 on magnitude-weighted Sobel orientations (p about 1e-24). The unused resample buffer in the composite is gone. Eye loss and neuromast expansion are still outside k_c.

- **Model fidelity to source: 88** — Pass 2 keeps the pseudo-spectral Schnakenberg step and the eigenvalue comparison, and now checks the two classical corollaries: the band disappears when Du = Dv, and the selected wavelength lengthens as Dv/Du increases above threshold. Kinetics remain a minimal carrier, not a cavefish-derived pair, which is why the score stays short of the top of the scale.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L84–96; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L271–275
- **Morphological realism / specimen match: 78** — The 12-frame 960x522 GIF is still the Seb Morič cave-morph photograph with a growing flank modulation. Pass 2 did not repaint the animal to chase the fitness score. The comparison figure now shows the sweep beside the fish. The photograph cannot resolve 666 nm, so specimen-match cannot be read as a pixel identity at that scale.  
  evidence: `docs/media/astyanax_mexicanus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L232–247
- **Parameter expressivity: 86** — On the unstable branch Dv/Du = 10, 12, 16, 20, 24, 30 the selected wavelength is 15.74, 16.38, 17.42, 18.26, 18.98, 19.88 px, strictly increasing, and sigma_max rises from 0.057 to 0.355. The self-test rejects a non-monotonic sweep. Beta remains a single visualization weight.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L249–268
- **Morphometric validation: 88** — Pearson r = 0.99997 and RMSE = 0.00455 on 11 modes. Nonlinear wavelength 660.7 nm versus the 665.6 nm linear calibration. Ablation sigma_max = -0.1 against a marginal reference of 0. Flank orientation KS = 0.105 (not 0). Melanin-dark fraction on the body mask is 0.002 against the Oca2-null reference of 0. The KS samples orientations with a fixed seed; it is not a spatial registration of melanophores.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L110–161; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L277–304
- **Source and data traceability: 90** — Same Crossref-checked DOIs and the same SHA-256-pinned CC BY-SA 4.0 voucher as pass 1. The README now states the sweep, the ablation and the KS test next to the 6.656e-7 m scale and the 0 Ma age. The wavelength is still a slot scale, not a literature measurement, which is said in the claim boundary rather than hidden.  
  evidence: `datasets/astyanax_mexicanus_turing_wavelength_s0/README.md` L1–50; `datasets/astyanax_mexicanus_turing_wavelength_s0/paper_citations.bib` L41–90
- **Code-to-model traceability: 86** — The sweep, the ablation and the wavelength bracket sit next to the Lean theorems for the discriminant, the equal-diffusivity failure and the 36-37x ratio against the 24 µm sibling. The composite path is now one signed resample. The integer Jacobian in Lean is the Schnakenberg state cleared by 5, which the script comments and does not re-derive in Lean.  
  evidence: `morphoeval/lean/loops/Loop_3125_astyanax_mexicanus_turing_wavelength_s0.lean` L33–59; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L104–108
- **Reproducibility package: 86** — --self-test now also requires the ablation to be negative, the sweep to be monotonic, the KS statistic to lie in (0, 1), r > 0.98, at least 10 frames and a short side of at least 480 px. Seed 3125 and the specimen hash are unchanged. Signing still needs a local nanopub profile, which this run does not have.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L550–562; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L76–82

Remediation:
- {'id': 'R5', 'item': 'The 666 nm length is imposed by the slot card. A wet-lab measurement of melanosome spacing on this species would replace the metres-per-pixel calibration.'}
- {'id': 'R6', 'item': 'No nanopub profile was present, so pass 2 is not signed. Signing waits on ~/.nanopub/profile.yml.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing wavenumber at 666 nm in the Astyanax cave morph
- **spec_fingerprint** (semantic model + inputs): `567daaa26ea32b6e0db869c7c7df6b1cf14d5ac54c7e7b02f34d577ba1dab69a`
- **input_dataset_sha256**: `datasets/astyanax_mexicanus_turing_wavelength_s0/astyanax_mexicanus_specimen.jpg` (`b7b2d7744b19592f…`), `datasets/astyanax_mexicanus_turing_wavelength_s0/paper_citations.bib` (`a4f57d067c37118a…`)
- `docs/media/astyanax_mexicanus_turing_wavelength_s0_comparison.png` — artifact_sha256 `8f2c47046806390f…`, 527,335 B, 1232×880 px
- `docs/media/astyanax_mexicanus_turing_wavelength_s0_growth.gif` — artifact_sha256 `198ad1aa6a86bd6d…`, 2,534,246 B, 960×522 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_growth_pearson_r | 0.9999681838003501 1 | 1.0 | Pearson r between closed-form Turing growth rates and forward-Euler Fourier-mode slopes |
| dispersion_growth_rmse | 0.004547985598148426 1/time | 0.0 | RMSE of Euler growth rates against analytical sigma(k), dt=0.1 |
| specimen_stripe_wavelength_nm | 660.6743783241391 nm | 24180.0 | radial FFT peak of the nonlinear patch, converted with the linear-mode calibration meters/px |
| specimen_melanin_dark_fraction | 0.002041123521088599 1 | 0.0 | fraction of body-mask pixels with max channel < 0.22 and low saturation on the CC voucher |
| homogeneous_mode_growth | -0.09999999999999998 1/time | -0.2 | sigma(k=0) of the Schnakenberg Jacobian; must be negative (stable without diffusion) |
| ablation_equal_diffusivity_sigma_max | -0.09999999999999998 1/time | 0.0 | max_k sigma(k) with Du = Dv = 2, gamma = 1 |
| specimen_flank_orientation_ks | 0.10540000000000002 1 | 0.0 | two-sample KS statistic on magnitude-weighted Sobel orientations, voucher flank vs simulated patch |
| diffusion_ratio_sweep_monotonic | 1.0 1 | 0.0 | 1 if selected wavelength is nondecreasing in Dv/Du on the unstable branch |

## Environment at checkpoint


Pass-2 files are introduced on this branch. The parent commit does not contain them, so blob URLs at that commit are not yet resolvable.
