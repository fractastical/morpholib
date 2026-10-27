# MorphoEval report — audit `2026-10-27-drosophila-virilis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila virilis Schnakenberg onset wavelength at 3.1 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py`) | 86 | 84 | 88 | 86 | 90 | 86 | 88 | 86.7 | 84.8 | 88.1 | 86.3 | **86.3** | faithful |

## Findings and remediation

### Drosophila virilis Schnakenberg onset wavelength at 3.1 µm (Holocene, 0 Ma)  —  overall 86.3 (faithful)

Pass 1. Shared Du and Dv/Du = 10, species-specific basal activator a. Onset wavelength is 3.101 µm for D. virilis (a = 0.10), 2.924 µm for predecessor D. teissieri (a = 0.058, still unstable), and no selected wavenumber for successor D. willistoni (a = 0.360, discriminant negative). All three are Holocene (0 Ma). Linearized dispersion Pearson r = 0.998 against continuum σ(k) (RMSE 0.032). Equal-diffusion ablation quenches activator std from 0.043 to 0. Saturated midline spacing is 3.625 µm, not the onset value. The Hanson 2019 habitus has a Nyquist length of 7.76 µm and does not resolve 3.1 µm.

- **Model fidelity to source: 86** — Schnakenberg steady state, Jacobian, onset k_c, and the four Turing inequalities are the published relations, and Du is solved from that onset law so virilis sits at 3.101 µm. The 2D field is explicit Euler on a fixed periodic patch. The chronospecies split is a change in a at fixed Du, which is a reconstruction, not a measured diffusivity. Deduction for the fixed domain and for using the habitus only as a modulated photograph rather than as the PDE domain.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L80–125; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L171–210; `morphoeval/lean/loops/Loop_3210_drosophila_virilis_turing_wavelength_s0.lean` L14–48
- **Morphological realism / specimen match: 84** — The comparison figure places the Hanson habitus, its foreground mask, the saturated activator field, the three-species dispersion curves, and the adult endpoint side by side. The endpoint resembles the fly because it is the photograph with a bounded modulation, which is the right macro appearance and a weak display of the 3.1 µm texture. The mask follows the body, wings, and legs. Willistoni's dispersion curve stays negative while virilis and teissieri cross above zero.  
  evidence: `docs/media/drosophila_virilis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_virilis_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L334–394
- **Parameter expressivity: 88** — Onset wavelength increases monotonically with basal activator a on [0.06, 0.12]. Equal diffusion (Dv = Du) drops activator standard deviation from 0.043 to 0. Teissieri, virilis, and willistoni are the same Du and Dv at three values of a, and only the first two are unstable. The diffusion ratio itself is not swept.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L127–162; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L323–331; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L478–479
- **Morphometric validation: 86** — Linearized growth rates on the real-eigenvalue band correlate with continuum σ(k) at Pearson r = 0.998 (RMSE 0.032), above the 0.98 threshold and not equal to it. Emergent midline spacing is 3.625 µm against the 3.101 µm onset reference. Habitus Nyquist is 7.76 µm. Orientation KS and pixel Pearson are computed against the photograph and are not copied from the reference anchors. Single specimen; a is not an independent empirical series.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L224–292; `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L514–631
- **Source and data traceability: 90** — Turing, Schnakenberg, Gierer–Meinhardt, Kondo–Miura, Gompel, and Murray are cited with DOIs. The habitus file is SHA-256 pinned to the Hanson 2019 Wikimedia original, CC BY-SA 4.0. The slot scale and 0 Ma age are stated in the dataset README.  
  evidence: `datasets/drosophila_virilis_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/drosophila_virilis_turing_wavelength_s0/drosophila_virilis_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_virilis_turing_wavelength_s0/drosophila_virilis_habitus.jpg` L0–0
- **Code-to-model traceability: 86** — Module docstring states the scale, the age, and which species changes a. steady_state, onset_k2, and sigma_of_k are the textbook objects. Lean theorems restate the ordering the script checks by rounding nanometres. The render composite is labeled as a modulation, not as a derivation of the veins.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L1–24; `morphoeval/math_kg_loops/delphi10000-drosophila-virilis-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 88** — Deterministic seed 3210, --self-test covers the wavelength ordering, the quench, Pearson r > 0.98, ablation, monotonicity, Nyquist, and the pinned SHA-256. One command renders the PNG and the 12-frame GIF. No dependency lock beyond the repository requirements file. Single-assessor scores.  
  evidence: `models/2026-delphi10000-drosophila-virilis-turing-wavelength-s0.py` L658–696; `datasets/drosophila_virilis_turing_wavelength_s0/README.md` L1–40

Remediation:
- R1. The PDE lives on a fixed 48 by 32 µm patch. Add a domain-length series and show that the number of activator peaks increases with patch length (mode count), which this pass does not measure.
- R2. The Hanson habitus Nyquist length is 7.76 µm. Keep that as an explicit negative result: the photograph is not evidence for the 3.101 µm wavelength.
- R3. Basal activator levels are reconstructed from pigmentation class. They are not a fit to a measured virilis concentration series.
- R4. Scores are from one assessor.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=5.0]** `model_equation` → `model-fidelity`: Fixed 48 by 32 µm patch. No growing ocellar disc, so mode insertion with domain length is not demonstrated.
- **[OPEN · P=3.0]** `specimen_comparison` → `morphometric-validation`: One lateral habitus. Pixel scale cannot confirm the 3.101 µm mode.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila virilis Schnakenberg onset wavelength at 3.1 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `7b617b1b25587059791fa537573c287d828ce4628df471d7ad0d187921f12af8`
- `datasets/drosophila_virilis_turing_wavelength_s0/drosophila_virilis_habitus.jpg` — artifact_sha256 `b82bab27947e6e61…`, 187,515 B, 1103×1188 px
- `docs/media/drosophila_virilis_turing_wavelength_s0_comparison.png` — artifact_sha256 `f27ebb8d2c1a036d…`, 564,225 B, 1512×984 px
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
| specimen habitus Nyquist wavelength | 7.762 µm | 3.101 | 2 × (3400 µm atlas length / 4√λ_max of the foreground second-moment). The habitus cannot resolve 3.101 µm when this Nyquist length is larger. |
| specimen foreground fraction | 0.2009 fraction | 0.45 | Border-median contrast mask on the Hanson 2019 habitus *(0.45 is the segmentation percentile used as a threshold label, not the measured area.)* |
| KS statistic of gradient orientation, specimen versus adult endpoint | 0.0315 KS distance | 0.2 | Two-sample KS on Sobel orientations of the habitus mask versus the last composed frame |
| pixel Pearson r of adult endpoint versus specimen habitus | 0.99572 correlation | 0.95 | Pearson correlation of luminance after the bounded PDE pigment modulation *(Tests that the modulation preserves the specimen; it does not claim the PDE drew the wing veins.)* |
| virilis tergite mode count | 156.632 modes | 122.128 | (adult length / 7) / lambda_onset using atlas lengths 3.4 mm and, as reference, the teissieri count |
| specimen structure-tensor coherency | 0.5743 coherency | 0.5 | Mean structure-tensor coherency on the habitus foreground |

## Environment at checkpoint


Pass 1 working tree for slot 3210. Cited paths are new files and are not in 0c76d15d.
