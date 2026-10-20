# MorphoEval report — audit `2026-10-19-diatom-frustule-phase-sep-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/71

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sumper (2002) & Lenoci-Camp (2008) hierarchical Cahn-Hilliard diatom biosilica phase separation (`2002-sumper-diatom-frustule-phase-sep.py`) | 58 | 22 | 44 | 48 | 62 | 60 | 67 | 53.3 | 31.8 | 63.5 | 47.7 | **47.7** | partially faithful |

## Findings and remediation

### Sumper (2002) & Lenoci-Camp (2008) hierarchical Cahn-Hilliard diatom biosilica phase separation  —  overall 47.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2002-sumper-diatom-frustule-phase-sep.py` (consensus score 51.6/100). The v1 script integrates the fourth-order Cahn-Hilliard PDE d(phi)/dt = M*laplacian(phi^3 - phi - eps^2*laplacian(phi)) via semi-implicit Fourier spectral stepping for off-critical droplet (phi0=-0.30) and critical bicontinuous (phi0=0.00) regimes, but omits two-stage hierarchical quenching (foramen -> cribrum -> cribrella), linear spinodal dispersion sigma(k), outside-spinodal negative controls, parameter sweeps across interfacial length epsilon, and computer-vision segmentation of the *Coscinodiscus wailesii* SEM specimen micrograph.

- **Model fidelity to source: 58** — Integrates the 2D periodic Cahn-Hilliard equation with exact zero-mode Fourier mass conservation (L27-57), separating off-critical droplet (phi0=-0.30) and critical bicontinuous (phi0=0.00) morphologies. However, v1 simulates only a single-scale quench without Lenoci & Camp's (2008) two-stage hierarchical solidification inside primary pores, omits the analytical spinodal dispersion relation sigma(k) = M*k^2*(1 - 3*phi0^2 - eps^2*k^2), and includes no negative control outside the spinodal boundary (|phi0| > 1/sqrt(3)).  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L27–57
- **Morphological realism / specimen match: 22** — Renders only a 2-panel synthetic comparison (`diatom_frustule_phase_sep_comparison.png`) without loading, SHA-256-verifying, or segmenting the vendored *Coscinodiscus wailesii* SEM micrograph (`datasets/diatom_frustule_phase_sep/coscinodiscus_diatom_specimen.jpg`).  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L85–110; `docs/media/diatom_frustule_phase_sep_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes only 2 parameter points (phi0=-0.30 and phi0=0.00 at eps=1.50, L87-88) with no CLI overrides for phi0, epsilon, M, N, or steps, and no parameter sweep across interfacial length epsilon or mean composition phi0.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L87–89
- **Morphometric validation: 48** — Computes integer-binned 2D radial PSD peak `k_bin` and connected-component pore count (L60-82), yielding discrete quantization error (lambda* = 19.2 px at k_bin=5 vs analytical 15.60 px), and omits sub-bin parabolic interpolation, 6-fold hexatic bond-orientational order |psi_6|, and empirical SEM pore diameter/pitch comparison.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L60–82
- **Source and data traceability: 62** — Reference dataset `datasets/diatom_frustule_phase_sep/diatom_frustule_phase_sep_reference.json` and specimen image exist on disk, but the v1 script does not load or verify their SHA-256 digests at runtime.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L1–25
- **Code-to-model traceability: 60** — Basic module docstring cites Sumper (2002) and Lenoci & Camp (2008), but functions lack explicit equation mappings and references to the Lean 4 theorems in `morphoeval/lean/loops/Loop_071_diatom_frustule_phase_sep.lean`.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L27–82
- **Reproducibility package: 67** — Deterministic RNG (`seed=2002`) and `--self-test` flag are present (L175-192), but `--self-test` checks only basic mass conservation and `std > 0.2` without testing spinodal dispersion, negative controls, JSON parity, or bitwise two-run determinism.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L175–183

Remediation:
- Implement Lenoci & Camp (2008) two-stage quenched hierarchical Cahn-Hilliard phase separation (primary foramen walls at eps_1=1.50 followed by secondary cribrum pore quench at eps_2=0.65 inside unsolidified primary pores) plus Sumper (2002) 3-tier scaling (foramen -> cribrum -> cribrella).
- Implement the exact linear spinodal dispersion relation sigma(k) = M*k^2*(1 - 3*phi0^2 - eps^2*k^2), critical wavenumber k_c, and an outside-spinodal negative control (phi0 = -0.65, |phi0| > 1/sqrt(3)) proving homogeneous stability.
- Load and SHA-256-verify `datasets/diatom_frustule_phase_sep/diatom_frustule_phase_sep_reference.json` and `coscinodiscus_diatom_specimen.jpg` at runtime, and segment both the primary hexagonal foramen lattice (139 pores, pitch, diameter-to-pitch ratio, hexatic |psi_6|) and secondary cribrum pores (74 pores) from the real *Coscinodiscus wailesii* SEM micrograph.
- Upgrade spectral morphometrics to sub-bin parabolic wavenumber interpolation, run a 5-point interfacial-scale parameter sweep across epsilon in {0.75, 1.00, 1.25, 1.50, 1.75}, expose full CLI flags, and render a comprehensive multi-panel synthesis figure and animated hierarchical growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sumper (2002) & Lenoci-Camp (2008) hierarchical Cahn-Hilliard diatom biosilica phase separation
- **spec_fingerprint** (semantic model + inputs): `cb843885a61dd35079ecc44f25b12a884b513caa2c4f708fa015d81046e64d1c`
- **input_dataset_sha256**: `datasets/diatom_frustule_phase_sep/coscinodiscus_diatom_specimen.jpg` (`47aca39b720b67b1…`), `datasets/diatom_frustule_phase_sep/diatom_frustule_phase_sep_reference.json` (`558186dc4409fb21…`), `datasets/diatom_frustule_phase_sep/scout_manifest.json` (`0eea5a04552420e8…`)
- `docs/media/diatom_frustule_phase_sep_comparison.png` — artifact_sha256 `3d96e9137cd584b0…`, 90,542 B, 1400×644 px

| measurement | value | reference | method |
|---|---:|---|---|
| Cahn-Hilliard primary droplet wavelength lambda* | 19.2 grid cells | 15.6 | integer-binned 2D radial PSD peak on N=96 periodic grid (phi0=-0.30, eps=1.50) *(k_bin=5, pores=29, std_phi=0.640)* |
| Cahn-Hilliard conserved mean order parameter <phi> | -0.3 order parameter | -0.3 | spatial mean of phi(x,y,t) after 800 steps vs initial phi0 = -0.30 *(exact zero-mode Fourier conservation)* |
| Cahn-Hilliard primary droplet pore count | 29 pores | 28 | connected components of minority LCPA phase (phi < 0) on N=96 grid *(wavelength=19.20 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2002-sumper-diatom-frustule-phase-sep.py` prior to Step 12 remediation in `2026-10-19-diatom-frustule-phase-sep-loop`.
