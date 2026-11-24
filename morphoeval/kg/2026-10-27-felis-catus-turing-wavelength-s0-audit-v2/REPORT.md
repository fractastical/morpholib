# MorphoEval report — audit `2026-10-27-felis-catus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-felis-catus-turing-wavelength-s0-audit -> 2026-10-27-felis-catus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 436 µm on Holocene Felis catus (pass 2) (`models/2026-delphi10000-felis-catus-turing-wavelength-s0.py`) | 90 | 84 | 88 | 86 | 92 | 86 | 90 | 89.3 | 84.8 | 89.6 | 87.6 | **87.6** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 436 µm on Holocene Felis catus (pass 2)  —  overall 87.6 (faithful)

Pass 2 keeps the Holocene nominal point (k* = 6.135, simulated wavelength 431.3 µm versus the 436 µm slot scale). A linearised spectral integration of sigma(k) matches the analytic dispersion with Pearson r = 0.990, RMSE 0.370, KS 0.111. Raising inhibitor diffusivity Dv from 0.55 to 3.20 lengthens lambda monotonically (Pearson r = 0.984). Equal diffusion and deletion of the u^2 v term both collapse the pattern. The segmented specimen is non-circular (circularity 0.146, aspect 1.37, foreground 0.30). Isotropic kinetics select a spot lattice; Dy/Dx = 4 raises the orientation anisotropy from 0.02 to 0.59 and produces stripes. Felis chronoseries is 1/1 at 0 Ma: an 80 µm cellular domain is stable, and a 4360 µm flank holds ten repeats of the same wave.

- **Model fidelity to source: 90** — The dispersion relation is both analytic and numerically integrated (Pearson r 0.990). Ablating autocatalysis and equalising diffusion both remove the instability, which is the Turing mechanism rather than a plotted template. Kinetics remain a Schnakenberg stand-in for the Dkk4 circuit, so the score stays short of a direct molecular implementation.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L89–105; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L453–513; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L411–440
- **Morphological realism / specimen match: 84** — The growth GIF ends on the real tabby, and the comparison figure places that photograph beside the 431 µm ocellus lattice and a stripe-biased field. The phenotype panel is a registration, not a reconstruction of fetal epidermis, and the isotropic law does not by itself reproduce mackerel stripes. Those limits are why this is not a mid-90s score.  
  evidence: `docs/media/felis_catus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/felis_catus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L720–770
- **Parameter expressivity: 88** — Dv from 0.55 to 3.20 lengthens the selected wavelength monotonically (Pearson r 0.984), crossing 436 µm at the nominal point. Dy/Dx = 4 moves orientation anisotropy from 0.02 (spots) to 0.59 (stripes). Both controls do the thing their names say.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L516–525; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L556–586
- **Morphometric validation: 86** — Wavelength 431.3 µm versus 436 µm, dispersion Pearson r 0.990, RMSE 0.370, KS 0.111, and four specimen-segmentation statistics whose references are not copies of the measured values. The wavelength reference is the slot scale, not a published caliper measurement on this cat, and n = 1 photograph.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L604–718; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L528–553
- **Source and data traceability: 92** — Same Crossref-checked DOIs and SHA-256-pinned CC BY-SA specimen as pass 1. The reference JSON states 436 µm and 0 Ma and says the length was not measured by Kaelin.  
  evidence: `datasets/felis_catus_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/felis_catus_turing_wavelength_s0/felis_catus_turing_wavelength_s0_reference.json` L1–96
- **Code-to-model traceability: 86** — Dispersion, sweep, ablation, and segmentation are separate functions with the law named in the docstring. Figure titles repeat lambda, r, and the 436 µm line. Murray's section numbers are still not cited in-line.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L453–470
- **Reproducibility package: 90** — Seed 3226, --self-test now asserts Pearson r > 0.98, a collapsing ablation, a monotonic Dv sweep, and a stripe anisotropy increase. One command redraws the PNG and the 12-frame GIF.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L588–601; `datasets/felis_catus_turing_wavelength_s0/README.md` L28–40

Remediation:
- {'id': 'R-assessor', 'item': 'Still a single builder-assessor. An independent second assessor was not run.'}
- {'id': 'R-micrograph', 'item': 'No 436 µm fetal-skin micrograph. The adult CC photograph is the phenotype carrier, and 436 µm remains the slot scale rather than a Kaelin caliper value.'}
- {'id': 'R-n1', 'item': 'Segmentation and image fitness use one individual (Von.grzanka 2010).'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 436 µm on Holocene Felis catus (pass 2)
- **spec_fingerprint** (semantic model + inputs): `dc73277e26df9e4f2f17850cccebf7acc56dadcc0fc11952482db451621788f3`
- **input_dataset_sha256**: `datasets/felis_catus_turing_wavelength_s0/felis_catus_specimen.jpg` (`9360be6f1689a1df…`), `datasets/felis_catus_turing_wavelength_s0/felis_catus_turing_wavelength_s0_reference.json` (`a7694c1c70d27f08…`)
- `docs/media/felis_catus_turing_wavelength_s0_comparison.png` — artifact_sha256 `23f064546c3b9212…`, 614,649 B, 1496×902 px
- `docs/media/felis_catus_turing_wavelength_s0_growth.gif` — artifact_sha256 `1a5b6367d29faa8a…`, 2,379,367 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_stripe_wavelength | 431.251 µm | 436.0 | radial FFT peak of the Schnakenberg activator on a 5-wave periodic patch, scaled so λ(k*) = 436 µm *(reference is the slot scale, not a Kaelin measurement; value is the simulated dominant wavelength)* |
| turing_sigma_star | 2.7561 1/time | 0.0 | max real eigenvalue of J - k^2 D over k in (0.15, 18) *(reference 0 is the marginal stability boundary; a positive value is the unstable Holocene point)* |
| cellular_predecessor_fundamental_sigma | -29.405 1/time | 2.7561 | σ(k_fund) on an 80 µm cellular-tier domain; Felis chronoseries has no earlier species *(reference is σ* of the 436 µm pattern; the cellular predecessor is stable)* |
| dispersion_pearson_r | 0.9901 1 | 1.0 | Pearson r between analytic sigma(k) and a linearised spectral integration across 18 wavenumbers *(reference 1 would be a perfect match; the value is the measured correlation and is not copied from it)* |
| dispersion_rmse | 0.3695 1/time | 0.0 | RMSE of numeric versus analytic growth rates; damped modes dominate the residual *(reference 0 is the zero-error line)* |
| dispersion_ks_statistic | 0.1111 1 | 0.0 | two-sample KS statistic between the analytic and numeric sigma samples *(reference 0 would mean identical distributions)* |
| wavelength_sweep_cv | 0.1132 1 | 0.0 | CV (std/mean) of lambda across the Dv sweep from 0.55 to 3.20 *(a zero CV would mean Dv does not move the wavelength)* |
| dv_sweep_lambda_pearson_r | 0.9841 1 | 0.0 | Pearson r of selected wavelength against inhibitor diffusivity Dv *(reference 0 is no relationship; the sweep is monotonically positive)* |
| ablation_equal_diffusion_amplitude | 0.0 concentration | 9.2496 | simulated negative control with Du = Dv; pattern amplitude after 140 steps *(reference is the patterned nominal amplitude, not the control)* |
| ablation_no_autocatalysis_amplitude | 0.0 concentration | 9.2496 | ablation that drops the u^2 v term; amplitude versus the intact model *(control versus nominal; values differ)* |
| specimen_foreground_fraction | 0.3023 1 | 0.5 | body-mask area fraction on the CC BY-SA Felis catus photograph (snow is background) *(reference 0.5 is an uninformative prior, not the measured fraction)* |
| specimen_contour_circularity | 0.146 1 | 1.0 | 4*pi*area/perimeter^2 of the segmented cat contour *(reference 1 is a circle; the standing cat is far from circular)* |
| specimen_structure_tensor_coherency | 0.4812 1 | 0.0 | mean structure-tensor coherency inside the segmented specimen *(reference 0 is an isotropic null)* |
| specimen_aspect_ratio | 1.3723 1 | 1.0 | bounding-box width/height of the segmented cat *(reference 1 is a square box)* |
| stripe_anisotropy_index | 0.5863 1 | 0.0213 | structure-tensor (Jxx-Jyy)/(Jxx+Jyy) after Dy/Dx = 4; isotropic spots sit near 0 *(reference is the isotropic spot lattice; the value is the stripe-biased run)* |

## Environment at checkpoint


Pass 2 working tree. Loci were not git-verified because this checkout's origin URL does not match the public repository URL byte for byte.
