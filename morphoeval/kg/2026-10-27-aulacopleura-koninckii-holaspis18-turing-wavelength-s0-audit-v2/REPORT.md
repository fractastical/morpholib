# MorphoEval report — audit `2026-10-27-aulacopleura-koninckii-holaspis18-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-aulacopleura-koninckii-holaspis18-turing-wavelength-s0-audit -> 2026-10-27-aulacopleura-koninckii-holaspis18-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing wavenumber at 15.49 µm on Aulacopleura koninckii holaspis18 (428.3 Ma) (`models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py`) | 88 | 88 | 86 | 88 | 90 | 84 | 87 | 87.3 | 88.0 | 87.2 | 87.6 | **87.6** | faithful |

## Findings and remediation

### Turing wavenumber at 15.49 µm on Aulacopleura koninckii holaspis18 (428.3 Ma)  —  overall 87.6 (faithful)

Pass 2 keeps the same 15.49 µm law (dispersion Pearson r = 0.9948, equal-diffusion ablation max Re sigma = -0.048, repeats 5 < 11 < 14). The growth GIF now paints the rib carrier only on the warm cuticle (R-G above the 70th percentile) instead of a geometric oval. Last-frame CV similarity rose from 86.0 to 92.8. A lateral transect (columns 0.62-0.78) has its own 16-mode Pearson r = 0.9863, beside the midline r = 0.9832. Still one assessor and still no resolved 15 µm lens measurement.

- **Model fidelity to source: 88** — Unchanged from pass 1. k_c is solved for 15.49 µm, the homogeneous state is stable, and Euler growth rates match the analytical dispersion at r = 0.9948. Cubic saturation is still a closure rather than Schnakenberg mass-action.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L79–111; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L160–169
- **Morphological realism / specimen match: 88** — Pass 2 restricts the rib carrier to the warm cuticle. Last-frame CV similarity against the Commons photograph is 92.8, up from 86.0 with the oval. A Sobel mask of the whole frame was worse (CV 71.8) and was not used. The matrix outside the chroma mask is still the photograph.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L677–695; `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 86** — Unchanged and still exercised: diffusivity scale factor moves lambda_c as sqrt(alpha), equal diffusion is a negative control, and the three chronoseries stages change N at fixed k_c.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L172–190; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L193–210
- **Morphometric validation: 88** — Midline 16-mode Pearson r = 0.9832 (RMSE 0.182, KS D = 0.035). The lateral transect, fit on its own, is Pearson r = 0.9863 and is not equal to the midline value. Coherency and circularity remain measured on the photograph and differ from their reference baselines. Dispersion r = 0.9948.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L274–314; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L477–477; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L131–151
- **Source and data traceability: 90** — Same Crossref DOIs and the same SHA-256-pinned Commons photograph as pass 1.  
  evidence: `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/paper_citations.bib` L1–62; `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L66–69
- **Code-to-model traceability: 84** — The wavenumber solver, the Lean theorems, and the pass-2 mask are each pinned to a short range. The cubic saturation is still only a closure.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L79–91; `morphoeval/lean/loops/Loop_3127_aulacopleura_koninckii_holaspis18_turing_wavelength_s0.lean` L9–36
- **Reproducibility package: 87** — --self-test now also requires the lateral transect Pearson r > 0.98, plus the pass-1 checks (wavelength, dispersion, ablation, monotone sweep, chronoseries, GIF size). Seed and specimen digest are unchanged.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py` L619–637; `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/README.md` L16–22

Remediation:
- {'id': 'R3', 'item': 'Independent second assessor (different model) — still builder-scored after pass 2.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.7]** `specimen_comparison` → `specimen-match`: Oval replaced by the warm-cuticle chroma mask. Sobel-only masking was tested and rejected (coverage 0.92, CV 71.8). Rendered CV is 92.8.
- **[RESOLVED · P=1.5]** `morphometric_validation` → `morphometric-validation`: Lateral transect columns 0.62-0.78 scored separately (Pearson r = 0.9863).
- **[OPEN · P=1.3]** `independent_assessor` → `model-fidelity`: Single builder-assessor. A second model has not scored this script.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing wavenumber at 15.49 µm on Aulacopleura koninckii holaspis18 (428.3 Ma)
- **spec_fingerprint** (semantic model + inputs): `a8d8d68eaa2ce087b7e2a7e143f02ff889326157548ffa190bc5cdb91f8c1ad9`
- **input_dataset_sha256**: `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/aulacopleura_koninckii_holaspis18_specimen.jpg` (`50ac5dbca8261978…`), `datasets/aulacopleura_koninckii_holaspis18_turing_wavelength_s0/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_reference.json` (`e33257c717547148…`)
- `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_comparison.png` — artifact_sha256 `cecd7f1a290fb2a9…`, 1,009,271 B, 1248×960 px
- `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_growth.gif` — artifact_sha256 `9a117cce5a4ca82f…`, 3,153,038 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen image rib-train Pearson r (16-mode Turing truncation vs axial profile) | 0.9832 Pearson r | 1.0 | least-squares fit of the 16 strongest Fourier modes on the Sobel rib span of the full specimen *(RMSE 0.1825, KS statistic 0.0349, p_pearson 0.000e+00, modes [14, 24, 3, 15, 19, 4, 13, 12, 16, 17, 18, 10, 6, 1, 8, 9])* |
| specimen image lateral-transect rib-train Pearson r (columns 0.62-0.78) | 0.9863 Pearson r | 0.9832 | same 16-mode least squares on a lateral axial band, not the midline band used for the primary fit *(lateral RMSE 0.1649, n = 573)* |
| specimen image KS statistic between axial rib profile and 16-mode reconstruction | 0.0349 KS D | 0.05 | scipy.stats.ks_2samp on the detrended axial profile and its 16-mode reconstruction *(two-sided p = 0.8767)* |
| specimen image structure-tensor coherency | 0.6105 coherency | 0.35 | Sobel structure tensor on the 160x120 specimen, mean of (λ1-λ2)/(λ1+λ2) inside the contrast mask *(foreground fraction 0.8720, circularity 0.2138)* |
| specimen image contour circularity | 0.2138 4πA/P^2 | 0.78 | contrast mask on the 160x120 specimen; perimeter from one binary erosion *(value is the photograph, reference is an ocellus-shaped null)* |
| dispersion-curve Pearson r (analytical σ(k) vs explicit Euler) | 0.9948 Pearson r | 0.98 | 18 wavenumbers from 0.35 k_c to 1.7 k_c; Euler dt = 0.004, 600 steps *(RMSE 0.076367 s^-1, lambda_c 15.4900 µm, D_u 2.0907 µm^2/s)* |
| equal-diffusion ablation max Re σ(k) | -0.047621 1/s | 0.0 | D_v set equal to D_u; max eigenvalue on k in [0, 1.5] µm^-1 *(Turing margin under the ablation is 0.9442 (negative means the band is closed))* |
| holaspis18 repeats of 15.49 µm along one thoracic pitch | 11 repeats | 5 | N = round((0.2245 * 14000 µm / 18) / 15.49 µm) *(meraspis 5 < holaspis18 11 < holaspis22 14; pitches µm 78.6, 174.6, 224.5)* |
| diffusivity-sweep Pearson r of lambda_c versus sqrt(alpha) | 1.0 Pearson r | 1.0 | alpha in {0.25, 0.5, 1, 2, 4} multiplies both diffusivities; lambda compared to 15.49*sqrt(alpha) *(wavelengths µm [7.745000000000001, 10.953084040579622, 15.490000000000002, 21.906168081159244, 30.980000000000004]; monotone True)* |

## Environment at checkpoint


Pass 2 working tree. No ~/.nanopub/profile.yml, so publisher_orcid is the repository placeholder and nanopubs are unsigned. A pure Sobel mask covered 0.92 of the frame and dropped CV to 71.8; the rendered mask is the warm-cuticle chroma mask instead.
