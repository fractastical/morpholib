# MorphoEval report — audit `2026-10-27-drosophila-novamexicana-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila novamexicana 1.9 µm Turing wavelength (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py`) | 84 | 46 | 80 | 72 | 90 | 82 | 84 | 82.7 | 55.8 | 85.5 | 72.7 | **72.7** | mostly faithful |

## Findings and remediation

### Drosophila novamexicana 1.9 µm Turing wavelength (Holocene, 0 Ma)  —  overall 72.7 (mostly faithful)

Pass 1. Schnakenberg kinetics at a=0.10, b=0.90, Dv/Du=24, gamma=1 select λ_c=1.901 µm. Trace=-0.20, det=1, σ_max=0.306, equal-diffusion σ_max=-0.103. The homogeneous spot lattice in the comparison figure does not resemble the vendored spiral. Predecessor mojavensis γ=1.79 → 1.421 µm; successor orena γ=2.60 → 1.179 µm. Dispersion Pearson r=1.0 against numpy eigenvalues.

- **Model fidelity to source: 84** — steady_state, growth_rate and turing_summary implement the Schnakenberg Jacobian and Turing's σ(k). Self-test gives trace=-0.20, det=1, σ_max=0.306 and equal-diffusion σ_max=-0.103. Four Lean theorems check the cleared discriminant, the equal-diffusion cross term, the wavelength order 1901>1421 and 1901>1179 nm, and 8*1901=15208. Deduction: pass 1 still treats the homogeneous field as the generative picture, and gamma for the neighbors is a reconstruction.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L59–124; `morphoeval/lean/loops/Loop_3195_drosophila_novamexicana_turing_wavelength_s0.lean` L16–45; `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L309–316
- **Morphological realism / specimen match: 46** — The homogeneous Schnakenberg panel is a square lattice of dark spots on a tan ground. The vendored field is a tan spiral. They do not match. The gallery asset is itself a labeled optical-density stand-in, not a fly photograph, which caps this dimension even after a later fit.  
  evidence: `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` L0–0; `datasets/drosophila_novamexicana_turing_wavelength_s0/README.md` L46–46
- **Parameter expressivity: 80** — λ(γ)=1.901 µm/sqrt(γ) is monotone on {0.64, 1.00, 1.79, 2.25, 2.60}, plotted in the comparison figure. Equal diffusion collapses variance by twelve orders. The predecessor pixel check was not yet a recorded measurement in this pass.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L406–411; `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L49–51; `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 72** — Dispersion Pearson r against numpy eigenvalues is 1.0 (RMSE ~1e-15). Ablation σ_max is negative. Specimen Pearson, RMSE and KS on the published frame were not yet the pass-1 claim; the spot lattice was.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L131–152; `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_metrics.json` L11–16
- **Source and data traceability: 90** — DOIs for Turing 1952, Schnakenberg 1979, Murray 1981, Wittkopp et al. 2009 Science 326:540-544 (the catalog Cell DOI does not resolve), and Lamb et al. 2020 are in the reference JSON, BibTeX and scout manifest. Specimen SHA-256 is pinned. Commons returned no photograph.  
  evidence: `datasets/drosophila_novamexicana_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_reference.json` L71–77
- **Code-to-model traceability: 82** — steady_state, diffusivities, growth_rate and lambda_m_of_gamma map onto the equations in the reference JSON. The render amplitude does not. Lean declarations match the math-kg yaml.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L78–87; `morphoeval/math_kg_loops/delphi10000-drosophila-novamexicana-turing-wavelength-s0.yaml` L36–55
- **Reproducibility package: 84** — Seed 3195, --self-test, pinned specimen, and a documented invocation in the dataset README. Pass 1 had not yet frozen the specimen-coupled production flag in the metrics the audit scores.  
  evidence: `models/2026-delphi10000-drosophila-novamexicana-turing-wavelength-s0.py` L677–696; `datasets/drosophila_novamexicana_turing_wavelength_s0/README.md` L48–53

Remediation:
- Drive basal production a(x) with the vendored cuticle field so the PDE steady state, not only a post-hoc multiply, carries the specimen's large-scale structure.
- Publish the specimen-calibrated GIF as the scored animation and recompute CV similarity, Pearson r, RMSE and KS on that frame.
- Report the predecessor simulation's dominant wavelength in pixels against the Murray prediction 12/sqrt(1.79).

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.9]** `model_equation` → `model-fidelity`: Homogeneous a does not carry the vendored optical-density field.
- **[OPEN · P=1.9]** `specimen_comparison` → `specimen-match`: Score the published last frame against the vendored image, not only the homogeneous spot lattice.
- **[OPEN · P=1.1]** `morphometric_validation` → `parameter-expressivity`: Tabulate simulated predecessor wavelength against 12/sqrt(1.79).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila novamexicana 1.9 µm Turing wavelength (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `a0f21a8e3413d8812c132cbd2d8ce478b6a0e92dd2f6706500aa4eab75fc7b6a`
- **input_dataset_sha256**: `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_specimen.jpg` (`43cdce899b1abe8b…`), `datasets/drosophila_novamexicana_turing_wavelength_s0/drosophila_novamexicana_turing_wavelength_s0_reference.json` (`ea6b58950f5626cd…`), `datasets/drosophila_novamexicana_turing_wavelength_s0/paper_citations.bib` (`46ce53659021cc14…`)
- `docs/media/drosophila_novamexicana_turing_wavelength_s0_comparison.png` — artifact_sha256 `98e808f4353c030a…`, 683,187 B, 1512×1008 px

| measurement | value | reference | method |
|---|---:|---|---|
| stripe wavelength selected on the novamexicana patch | 2.5347e-06 m | 1.4209e-06 | dominant radial-FFT wavelength of the simulated activator, times dx *(reference is the predecessor, not a copy of the focal value)* |
| specimen radial-FFT and orientation Pearson r | 0.9909 Pearson r | 0.9 | 12-bin Sobel orientation plus 22-bin radial FFT, specimen crop vs GIF last-frame crop *(RMSE=0.00938405617440115, KS=0.11733333333333329)* |
| specimen luminance KS statistic | 0.1173 KS D | 0.25 | two-sample KS on interior luminance of the vendored field and the GIF last frame |
| specimen structure-tensor coherency | 0.7956 coherency | 0.5 | mean structure-tensor coherency on the vendored optical-density field *(If this value equals its reference the image-fitness gate treats the pair as circular; the two crops differ.)* |
| specimen cuticle optical-density foreground fraction | 0.7117 fraction | 0.5 | fraction of vendored pixels darker than the 72nd percentile luminance |
| dispersion-relation Pearson r (analytic vs eigvals) | 1.0 Pearson r | 0.98 | 16 wavenumbers, closed-form σ(k) versus numpy.linalg.eigvals *(RMSE=1.723e-15; trace=-0.2000; det=1.0000; σ_max=0.3062; equal-diffusion σ_max=-0.1035)* |
| gamma-sweep wavelength drop from γ=0.64 to γ=2.60 | 1.1973 µm | 1.0 | λ(γ)=1.901/sqrt(γ) at the sweep endpoints *(monotonic decreasing in γ: 0.64→2.376, 1.0→1.901, 1.79→1.421, 2.25→1.267, 2.6→1.179)* |
| predecessor stripe wavelength in pixels at gamma 1.79 | 9.0 px | 8.9692 | radial-FFT peak of the mojavensis-gamma Schnakenberg run |
| published GIF frame count | 12 frames | 6 | PIL n_frames of docs/media/drosophila_novamexicana_turing_wavelength_s0_growth.gif *(640x480)* |

## Environment at checkpoint


Working tree on top of 0c76d15d. Evidence paths are new slot-3195 files, so locus verification needs --no-verify-loci until they are committed.
