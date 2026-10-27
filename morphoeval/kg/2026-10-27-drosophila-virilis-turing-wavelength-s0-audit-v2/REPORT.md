# MorphoEval report — audit `2026-10-27-drosophila-virilis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-virilis-turing-wavelength-s0-audit -> 2026-10-27-drosophila-virilis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila virilis Schnakenberg onset wavelength at 3.1 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py`) | 90 | 84 | 90 | 88 | 90 | 87 | 89 | 90.0 | 85.5 | 88.8 | 87.8 | **87.8** | faithful |

## Findings and remediation

### Drosophila virilis Schnakenberg onset wavelength at 3.1 µm (Holocene, 0 Ma)  —  overall 87.8 (faithful)

Pass 2, superseding the Pass 1 audit. The fixed-patch model is unchanged: onset wavelength 3.101 µm for D. virilis, 2.924 µm for D. teissieri, and no selected wavenumber for D. willistoni, all at 0 Ma. Dispersion Pearson r remains 0.998 (RMSE 0.032). The new domain-length series, at fixed dx = 0.5 µm, counts 4, 6, and 13 midline activator peaks on patches of 16, 32, and 48 µm. The count rises with length; the spacing is not constant (4.33, 5.00, 3.625 µm), so this is mode insertion in the weak sense, not a clean L/λ doubling. The habitus Nyquist length remains 7.76 µm and is now printed on the comparison figure as a negative result.

- **Model fidelity to source: 90** — Pass 1 relations stand: Schnakenberg steady state, onset k_c, and the Turing inequalities, with Du solved so virilis onset λ is 3.101 µm. Pass 2 adds domain_length_series, three periodic patches at fixed dx whose midline peak counts are 4, 6, and 13. That is the Murray scale check this slot was missing. Deductions that remain: explicit Euler, species activator levels are reconstructed, and the habitus is a modulated photograph rather than the PDE domain.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L80–125; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L171–210; `morphoeval/lean/loops/Loop_3210_drosophila_virilis_turing_wavelength_s0.lean` L14–48; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L294–321
- **Morphological realism / specimen match: 84** — The comparison figure places the Hanson habitus, its foreground mask, the saturated activator field, the three-species dispersion curves, and the adult endpoint side by side. The endpoint resembles the fly because it is the photograph with a bounded modulation, which is the right macro appearance and a weak display of the 3.1 µm texture. The mask follows the body, wings, and legs. Willistoni's dispersion curve stays negative while virilis and teissieri cross above zero.  
  evidence: `docs/media/drosophila_virilis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_virilis_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L334–394
- **Parameter expressivity: 90** — Activator sweep on [0.06, 0.12] remains monotone, equal diffusion still quenches, and patch length is now a second control: peak count goes 4 → 6 → 13 as length goes 16 → 32 → 48 µm. Spacing is not locked, so length does not act as a pure multiplier of a fixed wavelength. The diffusion ratio is still a single near-onset value.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L127–162; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L323–331; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L478–479; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L294–321
- **Morphometric validation: 88** — Dispersion Pearson r = 0.998 against continuum σ(k), RMSE 0.032, neither equal to its reference anchor. Emergent spacing 3.625 µm versus onset 3.101 µm. Domain-length counts 4, 6, 13 versus each other. Habitus Nyquist 7.76 µm is reported as a failed resolution of the 3.101 µm mode, not as a confirmation. Orientation KS and pixel Pearson are measured on the Hanson photograph. Still one specimen, and a is not an empirical series.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L224–292; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L514–631; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L560–585
- **Source and data traceability: 90** — Turing, Schnakenberg, Gierer–Meinhardt, Kondo–Miura, Gompel, and Murray are cited with DOIs. The habitus file is SHA-256 pinned to the Hanson 2019 Wikimedia original, CC BY-SA 4.0. The slot scale and 0 Ma age are stated in the dataset README.  
  evidence: `datasets/drosophila_virilis_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/drosophila_virilis_turing_wavelength_s0/drosophila_virilis_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_virilis_turing_wavelength_s0/drosophila_virilis_habitus.jpg` L0–0
- **Code-to-model traceability: 87** — Docstring, onset helpers, and Lean nanometre theorems still match the script's rounding checks. domain_length_series is labeled as the Pass 2 mode-count check. The figure title prints both the peak counts and the habitus Nyquist length.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L1–24; `morphoeval/math_kg_loops/delphi10000-drosophila-virilis-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 89** — --self-test now also requires the 48 µm patch to hold more peaks than the 16 µm patch. Seed 3210, pinned SHA-256, and the 12-frame 640×480 GIF are unchanged. Still one assessor and no lockfile beyond the repository requirements.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L658–696; `datasets/drosophila_virilis_turing_wavelength_s0/README.md` L1–40

Remediation:
- R1 closed. domain_length_series counts 4, 6, and 13 midline peaks at 16, 32, and 48 µm. Spacing is 4.33, 5.00, and 3.625 µm, so the count rises without a locked wavelength.
- R2 closed as a negative result. Nyquist length 7.76 µm is measured, asserted in --self-test, and printed on the habitus panel. The photograph does not resolve 3.101 µm.
- R3 still open. Basal activator levels remain a reconstruction from pigmentation class, not a fit to a measured concentration series.
- R4 still open. One assessor.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=5.0]** `model_equation` → `model-fidelity`: Closed in Pass 2: midline peaks are 4, 6, and 13 on 16, 32, and 48 µm patches. Spacing is not constant, so the closure is the inequality N(48) > N(16), not an exact doubling law.
- **[OPEN · P=3.0]** `specimen_comparison` → `morphometric-validation`: One lateral habitus. Pixel scale cannot confirm the 3.101 µm mode.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila virilis Schnakenberg onset wavelength at 3.1 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `7b617b1b25587059791fa537573c287d828ce4628df471d7ad0d187921f12af8`
- `datasets/drosophila_virilis_turing_wavelength_s0/drosophila_virilis_habitus.jpg` — artifact_sha256 `b82bab27947e6e61…`, 187,515 B, 1103×1188 px
- `docs/media/drosophila_virilis_turing_wavelength_s0_comparison.png` — artifact_sha256 `3547b4f69ff07d6a…`, 571,020 B, 1512×984 px
- `docs/media/drosophila_virilis_turing_wavelength_s0_growth.gif` — artifact_sha256 `f6348172a0be6c22…`, 2,544,362 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| virilis onset wavelength lambda_c | 3.101 µm | 3.101 | 2π/k_c with k_c^2 = (Dv f_u + Du g_v)/(2 Du Dv) after solving Du at a=0.10, Dv/Du=10 *(Assigned detail scale. Equality here is the Du calibration, reported separately from the emergent spacing.)* |
| virilis emergent midline peak spacing | 3.625 µm | 3.101 | Mean distance between positive midline maxima of the saturated Schnakenberg activator *(Near-onset k_max is close to but not identical with onset k_c, so this need not equal 3.101.)* |
| Pearson r of numerical versus analytic dispersion sigma(k) | 0.99808 correlation | 0.98 | Linearized 1D growth on 21 real-eigenvalue Fourier modes versus continuum eig(J - k^2 D) *(Reference is the pre-registered acceptance threshold, not the measured correlation.)* |
| RMSE of dispersion sigma(k) | 0.03183 1/time | 0.05 | Root mean square of numeric growth rate minus analytic σ(k) |
| teissieri onset wavelength | 2.9243 µm | 3.101 | Same Du and Dv as virilis, a=0.058 (predecessor, 0 Ma) |
| willistoni selected wavelength | 0.0 µm | 4.9948 | a=0.360 crosses the Turing discriminant; selected wavelength is 0 even though the formal onset expression is still defined *(Successor at 0 Ma. Reference is the formal continuation, which is not a selected mode.)* |
| ablation to wild-type activator std ratio | 0.0 ratio | 1.0 | Equal-diffusion control Dv=Du versus wild-type Dv=10 Du, same seed and duration *(Reference 1 would mean the control failed to quench. The measured ratio is the control result.)* |
| activator peaks on a 48 µm patch versus a 16 µm patch | 13 peaks | 4 | Midline positive maxima at fixed dx = 0.5 µm and Ly = 32 µm; reference is the 16 µm count *(Pass 2 domain-length series. Counts: 16µm=4, 32µm=6, 48µm=13)* |
| specimen habitus Nyquist wavelength | 7.762 µm | 3.101 | 2 × (3400 µm atlas length / 4√λ_max of the foreground second-moment). The habitus cannot resolve 3.101 µm when this Nyquist length is larger. |
| specimen foreground fraction | 0.2009 fraction | 0.45 | Border-median contrast mask on the Hanson 2019 habitus *(0.45 is the segmentation percentile used as a threshold label, not the measured area.)* |
| KS statistic of gradient orientation, specimen versus adult endpoint | 0.0315 KS distance | 0.2 | Two-sample KS on Sobel orientations of the habitus mask versus the last composed frame |
| pixel Pearson r of adult endpoint versus specimen habitus | 0.99572 correlation | 0.95 | Pearson correlation of luminance after the bounded PDE pigment modulation *(Tests that the modulation preserves the specimen; it does not claim the PDE drew the wing veins.)* |
| virilis tergite mode count | 156.632 modes | 122.128 | (adult length / 7) / lambda_onset using atlas lengths 3.4 mm and, as reference, the teissieri count |
| specimen structure-tensor coherency | 0.5743 coherency | 0.5 | Mean structure-tensor coherency on the habitus foreground |

## Environment at checkpoint


Pass 2 working tree. Adds the domain-length peak series on top of Pass 1. Cited paths are new in this loop and are not in 0c76d15d.
