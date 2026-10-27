# MorphoEval report — audit `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit -> 2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 1.5 µm in Drosophila eugracilis (Pass 2 sweep and controls) (`models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py`) | 90 | 76 | 90 | 88 | 92 | 86 | 90 | 90.0 | 80.5 | 89.6 | 86.1 | **86.1** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 1.5 µm in Drosophila eugracilis (Pass 2 sweep and controls)  —  overall 86.1 (faithful)

Pass 2 keeps the focal 1.485 µm Schnakenberg lattice and adds a six-point gamma sweep (0.50, 0.75, 1, 1.25, 1.584, 2). Analytic versus FFT wavelength Pearson r = 0.992, RMSE = 0.701 µm (the FFT ring sits systematically long of k_c; values are not equal). Equal diffusion D_u = D_v is not Turing and decays to activator std 2.5e-4 against Turing std 1.23. Blurred-arm foreground fraction is 0.177 versus raw Otsu 0.141. Plate-versus-render KS is 0.343. CV similarity of the 12-frame 720x540 GIF to the gallery plate is 88.6. Age remains 0 Ma; erecta lambda is 2.100 µm and ficusphila is 1.180 µm.

- **Model fidelity to source: 90** — Schnakenberg kinetics, the Turing conditions, and k_c match the cited formulas. Self-test checks that the explicit dispersion and the 1/sqrt(gamma) shortcut agree to 1e-9, and that gamma=1 is exactly 1.485 µm. The length unit is assigned, not fitted to a species SEM, which is why this is not higher.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L94–120; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L699–712
- **Morphological realism / specimen match: 76** — Side-by-side, the plate is a spiral and the simulation is a spot lattice. They share the plate's brown LUT and both fill the frame. Direct CV similarity of the GIF's last frame to the plate is 88.6 (orientation, radial FFT, coherency, occupancy) on a 12-frame 720x540 animation. The plate is not a fly photograph, so specimen-match stays below the range reserved for a segmented micrograph.  
  evidence: `docs/media/drosophila_eugracilis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_eugracilis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Raising gamma from 0.50 (erecta, 2.100 µm) through 1 (eugracilis, 1.485 µm) to the ficusphila value shortens lambda_c monotonically, and the six-point nonlinear FFT series follows that order (r=0.992). Setting D_v = D_u makes the Turing discriminant fail and the activator std falls from 1.23 to 2.5e-4.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L653–683; `docs/media/drosophila_eugracilis_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Pearson r = 0.992 on six gamma points (reference 1, so the score is not self-referential). RMSE = 0.701 µm records the systematic FFT-versus-k_c offset instead of hiding it. KS luminance = 0.343. Blurred-arm fraction 0.177 versus raw Otsu 0.141. Focus spacing CV 0.294 versus a perfect lattice at 0. Absolute wavelength calibration remains biased; correlation does not remove that.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L346–365; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L291–310; `datasets/drosophila_eugracilis_turing_wavelength_s0/pass2_metrics.json` L0–0
- **Source and data traceability: 92** — Crossref-verified DOIs for Turing, Schnakenberg, Gierer-Meinhardt, Murray, Kondo and Miura, Tanaka-Barmina-Kopp 2009 PNAS 10.1073/pnas.0807875106, Yassin 2016, and Ready 1976. Specimen SHA-256 f3705e55 is pinned. The rejected mouse-PGC DOI is documented in the scout manifest.  
  evidence: `datasets/drosophila_eugracilis_turing_wavelength_s0/paper_citations.bib` L1–90; `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — jacobian, turing_wavenumber and growth_rate are commented with the steady state and sigma(k). Lean theorems prove wavelength order, wavenumber reversal, twelve waves in 17820 nm, and the gamma order. The integer Lean wavelengths round the erecta float 2100.11 nm to 2100 nm.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L79–120; `morphoeval/lean/loops/Loop_3184_drosophila_eugracilis_turing_wavelength_s0.lean` L20–43
- **Reproducibility package: 90** — --self-test passed: dispersion identity, ablation std < 0.05, patterned std > 0.4, and Pearson r > 0.98 on a four-point sweep with measured lambda not equal to analytic lambda. Seed 3184 and SHA-256 locks are checked at startup. GIF gate requires at least 10 frames and 480 px.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L699–743; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L49–62

Remediation:
- {'id': 'R5', 'item': 'An independent second assessor has not scored this audit. Left open; it is not a numeric gate failure.'}
- {'id': 'R6', 'item': 'FFT wavelengths sit about 0.70 µm long of analytic k_c (hexagonal ring versus linear mode). A spot-spacing correction could be reported beside the raw ring; the correlation is already 0.992 and was not forced through the identity line.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 1.5 µm in Drosophila eugracilis (Pass 2 sweep and controls)
- **spec_fingerprint** (semantic model + inputs): `4cd159ca0e54b6b9882667ea66ee1c36e85c25ecfd3d9cd25e35efca3e35cc7f`
- **input_dataset_sha256**: `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_reference.json` (`aeae6e2ee094bbd3…`), `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_specimen.jpg` (`f3705e558598c7f5…`), `datasets/drosophila_eugracilis_turing_wavelength_s0/paper_citations.bib` (`963fd37391fe7bc7…`)
- `datasets/drosophila_eugracilis_turing_wavelength_s0/pass2_metrics.json` — artifact_sha256 `c683acd2a4252855…`, 3,443 B, 
- `docs/media/drosophila_eugracilis_turing_wavelength_s0_comparison.png` — artifact_sha256 `6d16c18dee30d5d1…`, 476,053 B, 1792×1148 px
- `docs/media/drosophila_eugracilis_turing_wavelength_s0_growth.gif` — artifact_sha256 `b571e2432db179c4…`, 2,461,602 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytic_turing_wavelength_um | 1.485 µm | 1.808868 | Schnakenberg dispersion k_c at gamma=1, length unit fixed to the assigned 1.485 µm; reference is the nonlinear FFT wavelength of the same run |
| simulated_stripe_fft_wavelength_um | 1.808868 µm | 1.485 | Radial Fourier ring of the saturated activator field, parabolic peak, converted by the focal length unit |
| predecessor_erecta_wavelength_um | 2.100107 µm | 1.485 | lambda(gamma=0.5) = 1.485/sqrt(0.5); predecessor contrast, not a published erecta SEM |
| successor_ficusphila_wavelength_um | 1.18 µm | 1.485 | Assigned successor wavelength 1.180 µm used to set gamma_ficusphila |
| specimen_stripe_arm_foreground_fraction | 0.140679 1 | 0.3543 | Otsu dark-arm fraction of the vendored gallery plate; reference is verify_image_fitness foreground_fraction (different estimator) |
| specimen_structure_tensor_coherency | 0.205514 1 | 0.8014 | Mean structure-tensor coherency of the full plate; reference is the fitness-script crop signature |
| specimen_vs_simulation_ks_luminance | 0.342917 1 | 0.0 | Two-sample KS statistic between plate luminance and the rendered activator; 0 would be identical histograms |
| ocellus_nearest_neighbor_spacing_cv | 0.29424 1 | 0.0 | CV of periodic nearest-neighbour distances among high-activator foci; 0 is a perfect lattice |
| specimen_segment_smoothed_foreground_fraction | 0.176775 1 | 0.140679 | Gaussian-blurred Otsu arm fraction; reference is the raw grain-sensitive Otsu fraction from the same plate |
| specimen_segment_smoothed_circularity | 0.097618 1 | 0.001623 | Circularity of the blurred arm mask; reference is the raw Otsu perimeter, which grain inflates |
| goodness_of_fit_pearson_r | 0.992318 1 | 1.0 | Pearson r of analytic λ_c(γ) versus nonlinear FFT λ across the γ sweep, including erecta, eugracilis and ficusphila |
| goodness_of_fit_rmse_um | 0.70098 µm | 0.0 | RMSE between analytic and FFT wavelengths on the same sweep |
| ablation_equal_diffusion_activator_std | 0.000248 1 | 1.232439 | Final activator std with D_u=D_v (Turing h<0) versus the patterned D_v/D_u=32 control, same noise seed |

## Environment at checkpoint


Pass 2 working tree. Signing unavailable: ~/.nanopub/profile.yml is absent.
