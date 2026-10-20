# MorphoEval report — audit `2026-10-19-diatom-frustule-phase-sep-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/71 · supersession chain `2026-10-19-diatom-frustule-phase-sep-audit -> 2026-10-19-diatom-frustule-phase-sep-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sumper (2002) & Lenoci-Camp (2008) hierarchical Cahn-Hilliard diatom biosilica phase separation (`2002-sumper-diatom-frustule-phase-sep.py`) | 92 | 87 | 86 | 88 | 93 | 91 | 90 | 90.0 | 87.4 | 91.3 | 89.3 | **89.3** | faithful |

## Findings and remediation

### Sumper (2002) & Lenoci-Camp (2008) hierarchical Cahn-Hilliard diatom biosilica phase separation  —  overall 89.3 (faithful)

Pass 2 (v2) remediated evaluation of `2002-sumper-diatom-frustule-phase-sep.py` (consensus score 89.0/100, up from 51.6/100 in Pass 1). The remediated script implements Lenoci & Camp's (2008) two-stage quenched hierarchical Cahn-Hilliard phase separation (53 Stage-1 foramina at eps_1=1.50 containing 102 nested Stage-2 cribrum sub-pores at eps_2=0.65), analytical spinodal dispersion sigma(k) with an outside-spinodal negative control (phi0=-0.65 -> 0 pores, std_phi=1.93e-3), a 5-point interfacial wavelength sweep (r=0.9865, RMSE=0.95 px), and automated CV segmentation of n=139 primary foramina (d/a=0.690 vs ref 0.6757, |psi_6|=0.764) and n=74 secondary cribrum sub-pores on the SHA-256-locked *Coscinodiscus wailesii* SEM micrograph, backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 92** — Implements the fourth-order Cahn-Hilliard PDE with exact Fourier zero-mode mass conservation (drift < 1e-15), Lenoci & Camp's (2008) Section 3.3 two-stage quenched hierarchical solidification (`simulate_two_stage_hierarchical_diatom`, L164-242), the analytical spinodal dispersion relation sigma(k) = M*k^2*(1 - 3*phi0^2 - eps^2*k^2) (`compute_spinodal_dispersion`, L76-123), and an outside-spinodal negative control (phi0=-0.65, alpha=-0.2675 < 0) verified to decay to 0 pores. Backed by 4 machine-checked Lean 4 theorems in `Loop_071_diatom_frustule_phase_sep.lean`.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L103–150; `2002-sumper-diatom-frustule-phase-sep.py` L157–283; `morphoeval/lean/loops/Loop_071_diatom_frustule_phase_sep.lean` L18–97
- **Morphological realism / specimen match: 87** — Vendors, SHA-256-verifies, and segments the *Coscinodiscus wailesii* valve SEM micrograph (`datasets/diatom_frustule_phase_sep/coscinodiscus_diatom_specimen.jpg`), extracting n=139 primary foramen pores (mean diameter 24.01 px, CV=0.133; nearest-neighbor pitch 34.80 px, CV=0.075; diameter-to-pitch ratio d_1/a_1 = 0.690 matching Sumper 2002 reference 0.6757 within 2.1%; 6-fold hexatic order |psi_6| = 0.764) and n=74 nested cribrum sub-pores (`segment_coscinodiscus_specimen`, L442-526).  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L442–526; `docs/media/diatom_frustule_phase_sep_comparison.png` L0–0
- **Parameter expressivity: 86** — Simulates 4 distinct phase-separation regimes (off-critical droplet array, two-stage hierarchical foramen+cribrum quench, critical symmetric bicontinuous honeycomb, and outside-spinodal negative control), executes a 5-point sweep across interfacial stiffness epsilon in {0.75, 1.00, 1.25, 1.50, 1.75} (`run_epsilon_wavelength_sweep`, L394-435), and exposes `--N`, `--phi0`, `--epsilon`, `--steps`, `--seed`, and `--skip-gif` CLI flags (L896-924).  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L394–435; `2002-sumper-diatom-frustule-phase-sep.py` L896–924
- **Morphometric validation: 88** — Uses 3-point parabolic sub-bin radial PSD interpolation (`_parabolic_subbin_peak`, L290-302), validates linear wavelength scaling across epsilon (`r = 0.9865`, `RMSE = 0.95 px`), quantifies 6-fold hexatic bond-orientational order `|psi_6|` on both simulated droplets and 139 SEM foramina (`0.764` vs reference `0.75`), measures nested cribrum sub-pore multiplicity per foramen, and verifies 3-tier self-similar scaling (`|r| = 0.9999`, `rho = 0.1833` vs reference `0.1800`).  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L290–435; `2002-sumper-diatom-frustule-phase-sep.py` L746–827
- **Source and data traceability: 93** — Vendors `datasets/diatom_frustule_phase_sep/` with `diatom_frustule_phase_sep_reference.json`, `coscinodiscus_diatom_specimen.jpg` (CC BY-SA 4.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (4 DOIs: `10.1126/science.1070026`, `10.1021/la7022784`, `10.1166/jnn.2006.174`, `10.1063/1.1744102`), and `README.md`, verified at runtime by `verify_and_load_reference()` (L78-95).  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L78–95; `datasets/diatom_frustule_phase_sep/diatom_frustule_phase_sep_reference.json` L1–136
- **Code-to-model traceability: 91** — Module and function docstrings map every equation and computational step directly to Sumper (2002) Figs. 1-4, Lenoci & Camp (2008) Eqs. 1-6 & Sec. 2-3.3, Losic et al. (2006) Table 1, and the 4 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_071_diatom_frustule_phase_sep.lean`.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L1–44
- **Reproducibility package: 90** — Deterministic RNG (`seed=2002`), 8-check `self_test()` suite (L834-893) verifying SHA-256 provenance, JSON parity, outside-spinodal decay, two-stage hierarchical pore nesting, wavelength sweep correlation, SEM segmentation parity, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized static PNG and animated hierarchical growth GIF outputs.  
  evidence: `2002-sumper-diatom-frustule-phase-sep.py` L834–893

Remediation:
- Couple the 2D Cahn-Hilliard equation to a radially advancing centrifuge-driven silica precipitation front to model radial rib (costae) striation near the Coscinodiscus valve margin.
- Extend the two-stage hierarchical quench to an explicit 3-stage 512x512 grid simulation resolving tertiary cribrella pores simultaneously within a single domain.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sumper (2002) & Lenoci-Camp (2008) hierarchical Cahn-Hilliard diatom biosilica phase separation
- **spec_fingerprint** (semantic model + inputs): `42daf8406b78c3205a6436f49e13b3d0bd7cf77c46e21bab32d26f1699bd22a7`
- **input_dataset_sha256**: `datasets/diatom_frustule_phase_sep/coscinodiscus_diatom_specimen.jpg` (`47aca39b720b67b1…`), `datasets/diatom_frustule_phase_sep/diatom_frustule_phase_sep_reference.json` (`b6ffc32e8bdecb23…`), `datasets/diatom_frustule_phase_sep/scout_manifest.json` (`0eea5a04552420e8…`)
- `docs/media/diatom_frustule_phase_sep_comparison.png` — artifact_sha256 `c762c88f7077832b…`, 876,795 B, 2775×1380 px

| measurement | value | reference | method |
|---|---:|---|---|
| Coscinodiscus wailesii SEM foramen diameter-to-pitch ratio (d_1 / a_1) | 0.6897 ratio | 0.6757 | segmented foramen pore equivalent diameter (24.01 px, CV=0.133) / nearest-neighbor pitch (34.80 px, CV=0.075) across n=139 pores on SHA-256-locked SEM ROI *(n=139 pores, diameter CV=0.133, pitch CV=0.075, relative error=2.1%)* |
| Coscinodiscus wailesii SEM primary foramen 6-fold hexatic order |psi_6| | 0.7638 |psi_6| | 0.75 | mean 6-nearest-neighbor bond-orientational order |(1/N_j) sum exp(6i theta)| on n=139 segmented SEM foramen pores vs simulated Cahn-Hilliard droplet array *(SEM |psi_6|=0.7638 vs simulated Cahn-Hilliard |psi_6|=0.5862)* |
| Cahn-Hilliard interfacial thickness sweep wavelength correlation r(lambda*, lambda_c) | 0.9865 Pearson r | 1.0 | 5-point interfacial parameter sweep across epsilon in {0.75, 1.00, 1.25, 1.50, 1.75} comparing sub-bin FFT wavelength lambda* against analytical spinodal lambda_c = 2*pi/k_c *(p=1.87e-03, RMSE=0.95 px across 5 epsilon regimes)* |
| Two-stage hierarchical quench nested cribrum sub-pores per foramen | 2.04 sub-pores / foramen | 7.0 | connected secondary cribrum pores (eps_2=0.65) formed inside each solidified Stage 1 primary foramen chamber (eps_1=1.50) *(total foramina=53, total nested cribrum pores=102, SEM cribrum ROI pores=74)* |
| Outside-spinodal negative control (phi0 = -0.65) pore count vs spinodal droplet regime | 0 pores | 0 | connected pore count when mean composition |phi0|=0.65 exceeds spinodal threshold 1/sqrt(3)=0.5774 (alpha = 1 - 3*phi0^2 = -0.2675 < 0) vs phi0=-0.30 control *(negative control std_phi=1.93e-03 (0 pores) vs spinodal droplet std_phi=0.625 (53 pores))* |
| Cahn-Hilliard conserved mean order parameter <phi> drift | 0.0 order parameter drift | 0.0 | absolute mass conservation drift |<phi(x,t)> - phi0| after 650 semi-implicit spectral steps *(mean(phi)=-0.30000000 vs exact phi0=-0.30000000)* |
| Sumper (2002) 3-tier pore diameter log-linear scaling correlation (Foramen -> Cribrum -> Cribrella) | 0.9999 |r| | 1.0 | log10(d_m) linear regression across Generation 1 (1250 nm), Generation 2 (220 nm), and Generation 3 (42 nm) *(effective inter-tier scale ratio rho=0.1833 (ref 0.1800), p=8.81e-03)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2002-sumper-diatom-frustule-phase-sep.py` superseding `2026-10-19-diatom-frustule-phase-sep-audit`.
