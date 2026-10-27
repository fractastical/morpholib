# MorphoEval report — audit `2026-10-27-drosophila-guttifera-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-27-drosophila-guttifera-turing-wavelength-s0-audit -> 2026-10-27-drosophila-guttifera-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg micro-ocellus wavelength of Drosophila guttifera at 2.12 µm (`models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py`) | 84 | 82 | 88 | 86 | 90 | 84 | 86 | 85.3 | 83.5 | 86.8 | 85.0 | **85.0** | faithful |

## Findings and remediation

### Schnakenberg micro-ocellus wavelength of Drosophila guttifera at 2.12 µm  —  overall 85.0 (faithful)

Pass 2 keeps the periodic Dv sweep (Pearson r = 0.992, RMSE = 0.208 µm, focal centroid 1.983 µm against the 2.12 µm linear calibration). The new zero-flux control selects 1.767 µm. Saturated guttifera peaks are spaced 1.332 µm apart (97 peaks, CV = 0.255, KS = 0.801 against a normal centered on 2.12 µm), so saturation shortens the spacing further; that gap is reported rather than fitted away. The wing-membrane mask counts 18 dark spots against Werner's 16; the pass-1 global threshold had counted 28. Grimshawi coherency remains 0.480 against guttifera 0.010. Heteroneura linear wavelength remains 2.048 µm.

- **Model fidelity to source: 84** — The periodic dispersion relation is unchanged and still matches the Lean discriminant. Pass 2 adds an explicit zero-flux integrator instead of leaving the boundary untested. The Wingless Gaussian and the 2.12 µm unit choice remain reconstructions under Turing's law, not equations from Werner 2010.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L77–110; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L222–266
- **Morphological realism / specimen match: 82** — Wing-membrane segmentation now returns 18 spots against the nominal 16, down from the pass-1 global count of 28. The comparison figure still shows the adult male beside the three chronospecies fields. The GIF ends on the photograph, which is the right adult phenotype and the wrong magnification for 2.12 µm.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L303–354; `docs/media/drosophila_guttifera_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_guttifera_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Dv still moves the centroid with the analytical curve (r = 0.992) while staying short of the 1:1 line. Anisotropy moves coherency from 0.010 to 0.480. Equal diffusion decays (ratio 0.228). Switching to zero-flux moves the centroid from 1.983 µm to 1.767 µm, so the boundary condition is not inert.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L386–420; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L535–552
- **Morphometric validation: 86** — Linear centroid: r = 0.992, RMSE = 0.208 µm, focal numerical value 1.983 µm against reference 2.12 µm. Saturated nearest-neighbour spacing is 1.332 µm (CV = 0.255, KS = 0.801 versus a normal at 2.12 µm), which widens the linear shortfall instead of cancelling it. Wing-membrane count is 18 versus 16. None of these specimen or spacing values were set equal to their references.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L270–293; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L520–560
- **Source and data traceability: 90** — Unchanged from pass 1. Crossref-checked DOI 10.1038/nature08896, specimen SHA-256, MA Hanson, CC BY-SA 4.0. The sibling slot's DOI 10.1038/nature08994 is a different paper and is not used.  
  evidence: `datasets/drosophila_guttifera_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_guttifera_turing_wavelength_s0/drosophila_guttifera_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 84** — Docstring, dispersion(), integrate_neumann(), and the Lean discriminant are the same law. Pass 2's new measurements name the control they come from. Spieth 1981 is not cited in the code as a wavelength source.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3187_drosophila_guttifera_turing_wavelength_s0.lean` L43–68
- **Reproducibility package: 86** — Same seed, hash pin, and --self-test. The self-test now also requires the wing mask to beat the global count, a Neumann wavelength inside 1.4-2.8 µm, and a positive saturated spacing. Pass 2 re-rendered the PNG and the 12-frame GIF.  
  evidence: `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L722–770; `models/2026-delphi10000-drosophila-guttifera-turing-wavelength-s0.py` L68–74

Remediation:
- {'id': 'R4', 'item': 'Independent second assessor. Pass 2 is still scored by the builder alone.'}
- {'id': 'R5', 'item': 'The saturated spacing (1.33 µm) and the zero-flux centroid (1.77 µm) both sit short of 2.12 µm. A weakly nonlinear amplitude expansion, or a fit of Du and Dv to a real micrograph, would be the next measurement. No such micrograph is in this slot.'}

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
| specimen wing-membrane spot count | 18 components | 16.0 | dark components on foreground outside the dilated reddish body, area in (18, 900) px *(pass-1 global dark count was 28; wing mask is not forced to 16)* |
| zero-flux control wavelength | 1.766819077357544 µm | 2.12 | explicit reflecting-boundary Schnakenberg, N=48, spectral centroid in the same fixed band *(steps=921, std=0.200, finite=True)* |
| saturated activator nearest-neighbour spacing | 1.3315482846620719 µm | 2.12 | local maxima of the saturated guttifera field, minimum-separation window 0.55 λ *(n_peaks=97, CV=0.2551344310914646, KS vs N(2.12, s)=0.8012877714914374)* |
| specimen structure-tensor coherency | 0.5564665037367912 coherency | 0.35 | Sobel structure tensor on the CC BY-SA adult male, masked against the border median |
| specimen foreground fraction | 0.5692011071670048 area fraction | 0.4 | pixels whose contrast to the border median exceeds the 45th percentile |
| specimen dark-blob area coefficient of variation | 0.9080574179871628 CV | 0.25 | std/mean of component areas in (20, 800) px *(mean blob area = 193.7 px)* |

## Environment at checkpoint


Parent of the loop commit. Slot 3187 paths are new in the working tree.
