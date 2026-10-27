# MorphoEval report — audit `2026-10-27-drosophila-guttifera-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg micro-ocellus wavelength of Drosophila guttifera at 2.12 µm (`models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py`) | 80 | 74 | 84 | 78 | 90 | 80 | 84 | 81.3 | 75.5 | 84.9 | 80.1 | **80.1** | faithful |

## Findings and remediation

### Schnakenberg micro-ocellus wavelength of Drosophila guttifera at 2.12 µm  —  overall 80.1 (faithful)

Focal linear wavelength is calibrated to 2.12 µm (k_max = 2.88, marginal k_c = 4.96). On a fixed 12-wavelength box the spectral centroid tracks analytical λ across eight inhibitor-diffusivity multipliers with Pearson r = 0.992 and RMSE = 0.208 µm; the centroid sits systematically short of λ(k_max). Equal diffusion (Du = Dv = 0.40) decays (amplitude ratio 0.228, σ_max < 0). Saturated guttifera coherency is 0.010; grimshawi with vein anisotropy 3.15 reaches 0.480; heteroneura at a = 0.24 has linear λ = 2.048 µm. The adult photograph (CC BY-SA 4.0, MA Hanson) segments to 28 dark blobs against the nominal 16 wing spots. Pass 1 uses periodic boundaries only.

- **Model fidelity to source: 80** — dispersion() implements the Schnakenberg Jacobian, marginal k_c, and σ(k). The Wingless Gaussian and the 2.12 µm calibration are reconstructions placed under that law. Pass 1 integrates only with periodic boundaries.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L77–110; `morphoeval/lean/loops/Loop_3187_drosophila_guttifera_turing_wavelength_s0.lean` L54–58
- **Morphological realism / specimen match: 74** — The comparison figure places the CC BY-SA adult male beside guttifera, grimshawi, and heteroneura fields. Grimshawi is the oriented field (coherency 0.48 vs 0.01). The GIF's last frame is the photograph, so it matches the adult, not a 2.12 µm micrograph. Global segmentation returns 28 blobs rather than 16 wing spots.  
  evidence: `docs/media/drosophila_guttifera_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_guttifera_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — Eight Dv multipliers move analytical and numerical wavelengths together (r = 0.992) but the centroid stays short of the 1:1 line. Anisotropy 3.15 raises coherency from 0.010 to 0.480. Equal diffusion decays (ratio 0.228).  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L295–312; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L320–345
- **Morphometric validation: 78** — Pearson r = 0.992 and RMSE = 0.208 µm on the linear centroid. KS D = 0.5 against a reference of 0, so the numerical and analytical sets are not the same distribution. Specimen coherency 0.556 and foreground fraction 0.569 are measured on the photograph and are not equal to their reference values. The spot count (28 vs 16) is an unrefined global threshold. Saturated inter-peak spacing is not yet measured.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L360–470; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L230–264
- **Source and data traceability: 90** — Primary DOI 10.1038/nature08896 was checked on Crossref (the sibling slot's 10.1038/nature08994 is a different paper). Specimen SHA-256, creator MA Hanson, and CC BY-SA 4.0 are in the reference JSON.  
  evidence: `datasets/drosophila_guttifera_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_guttifera_turing_wavelength_s0/drosophila_guttifera_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — The module docstring states the PDE, k_c, and the chronospecies contrast. dispersion() is the equation. The Lean file proves the integer form of the same discriminant and the 8-by-2120 nm patch. Pass 1 does not yet mark a zero-flux control in the code.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L1–36; `morphoeval/lean/loops/Loop_3187_drosophila_guttifera_turing_wavelength_s0.lean` L36–68
- **Reproducibility package: 84** — Seed 3187, --self-test, and a SHA-256 pin on the specimen. The self-test checks r > 0.98, ablation decay, grimshawi coherency, and the Lean file. Dependencies are the repo scientific stack. The run is deterministic given that stack.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L68–74; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L608–655

Remediation:
- {'id': 'R1', 'item': 'Add a zero-flux (Neumann) control and report its wavelength against the periodic run. Pass 1 is periodic only.'}
- {'id': 'R2', 'item': 'Replace the global dark threshold (28 blobs) with a wing-membrane mask that drops the thorax and border-touching bristles, and compare that count to 16 without forcing equality.'}
- {'id': 'R3', 'item': 'Measure saturated inter-peak spacing and its KS/CV against the linear 2.12 µm wavelength. The linear centroid is systematically short (RMSE 0.208 µm); say whether saturation makes that better or worse.'}
- {'id': 'R4', 'item': 'Independent second assessor. This pass is builder-scored only. Left open if no second model is available.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg micro-ocellus wavelength of Drosophila guttifera at 2.12 µm
- **spec_fingerprint** (semantic model + inputs): `c4a6d950a832bc4358255a26def4f385a34b6882ca4a5713ae484c2d3f07773c`
- `docs/media/drosophila_guttifera_turing_wavelength_s0_comparison.png` — artifact_sha256 `9e62649b8e7299a0…`, 479,439 B, 1512×1008 px
- `docs/media/drosophila_guttifera_turing_wavelength_s0_growth.gif` — artifact_sha256 `298a0da7542ff7ca…`, 2,758,061 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| focal linear wavelength lambda(k_max) | 2.12 µm | 2.12 | spatial calibration: (2 pi / k_max) * scale, scale chosen so this equals the slot wavelength *(analytical focal wavelength is 2.12 µm by calibration; the numerical Fourier peak is reported separately)* |
| numerical Fourier peak wavelength at focal Dv | 1.9828609103267116 µm | 2.12 | dominant non-zero mode of the linear-regime field, stopped at std(u)=0.08 *(value is the measured peak and is not forced equal to the 2.12 µm reference)* |
| Pearson r of numerical vs analytical wavelength across Dv sweep | 0.9920977526336726 correlation | 0.98 | np.corrcoef across the fixed-domain Dv sweep; spectral centroid in cycles [0.30, 0.80] *(RMSE = 0.2077 µm)* |
| RMSE of numerical vs analytical wavelength | 0.20767823063199292 µm | 0.15 | root mean square of (lambda_numerical - lambda_analytical) over the Dv sweep |
| two-sample KS statistic of numerical vs analytical wavelength sets | 0.5 KS D | 0.0 | scipy.stats.ks_2samp(lambda_numerical, lambda_analytical) *(p = 0.2827)* |
| equal-diffusion ablation amplitude ratio | 0.22790379970689206 std(final)/std(initial) | 1.0 | Du = Dv = 0.40, periodic Schnakenberg, T = 36; pattern must decay *(ablation sigma_max = -0.1010)* |
| grimshawi stripe coherency minus guttifera spot coherency | 0.47008807404570835 structure-tensor coherency | 0.0 | saturated fields; grimshawi anisotropy 3.15 versus guttifera isotropy plus sensillum Gaussian *(coh grimshawi=0.480, guttifera=0.010, heteroneura=0.028)* |
| heteroneura linear wavelength | 2.047534255940426 µm | 2.12 | k_max at a = 0.24, same Du, Dv, no sensillum Gaussian *(reference is the guttifera focal wavelength; the value must differ)* |
| grimshawi geometric-mean linear wavelength | 2.823282891000549 µm | 2.12 | k_max with Du and Dv multiplied by sqrt(3.15) |
| specimen dark wing-spot count | 28 components | 16.0 | global gray<0.28, opening, components with area in (20, 800) px *(segmentation is not a manual count of 16 spots; legs and split spots move the number)* |
| specimen structure-tensor coherency | 0.5564665037367912 coherency | 0.35 | Sobel structure tensor on the CC BY-SA adult male, masked against the border median |
| specimen foreground fraction | 0.5692011071670048 area fraction | 0.4 | pixels whose contrast to the border median exceeds the 45th percentile |
| specimen dark-blob area coefficient of variation | 0.9961786712848507 CV | 0.25 | std/mean of component areas in (20, 800) px *(mean blob area = 176.4 px)* |

## Environment at checkpoint


Parent of the loop commit. The slot 3187 files are new in the working tree and are not at this commit; blob URLs against this SHA do not resolve those paths.
