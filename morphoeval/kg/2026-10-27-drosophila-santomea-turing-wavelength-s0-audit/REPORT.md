# MorphoEval report — audit `2026-10-27-drosophila-santomea-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila santomea Schnakenberg wavelength at 2.326 µm (Holocene, 0 Ma), pass 1 (`models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py`) | 86 | 68 | 64 | 70 | 86 | 82 | 84 | 78.7 | 68.8 | 84.1 | 76.3 | **76.3** | mostly faithful |

## Findings and remediation

### Drosophila santomea Schnakenberg wavelength at 2.326 µm (Holocene, 0 Ma), pass 1  —  overall 76.3 (mostly faithful)

Pass 1 locks Schnakenberg parameters (a=0.130, b=0.870, Du=1, Dv=40) so the continuous critical wavelength is 2.326 µm at γ=1, age 0 Ma. Numeric growth rates match the analytic dispersion (Pearson r rounds to 1.0 at six decimals, RMSE 4.9e-5). The 12-frame 800×600 GIF and the comparison figure were written. The γ sweep, equal-diffusion ablation, scale-bar segmentation, and KS distances are listed as remediation_pending in pass1_metrics.json.

- **Model fidelity to source: 86** — The Jacobian, the Turing inequalities, and k_c are the Schnakenberg linearization, and the numeric σ(k) matches it. Pass 1 does not yet exercise the equal-diffusion control or the γ family.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L75–111; `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L136–233; `morphoeval/lean/loops/Loop_3201_drosophila_santomea_turing_wavelength_s0.lean` L16–48
- **Morphological realism / specimen match: 68** — The comparison figure puts the sepia gallery spiral beside a spot field in the same palette, and the GIF's last frame is blended toward the gallery image. Pass 1 does not measure the drawn 100 µm bar, so the scale mismatch is visible but not quantified.  
  evidence: `datasets/drosophila_santomea_turing_wavelength_s0/pass1_comparison.png` L0–0; `datasets/drosophila_santomea_turing_wavelength_s0/pass1_growth.gif` L0–0
- **Parameter expressivity: 64** — Only γ=1 is integrated on this pass. The chronoseries table states the other two wavelengths, but their growth-rate curves are not computed.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L115–134; `datasets/drosophila_santomea_turing_wavelength_s0/pass1_metrics.json` L0–0
- **Morphometric validation: 70** — Dispersion Pearson r and RMSE are real. There is no KS distance, no ablation amplitude, and no specimen length yet, so the morphometric claim is only the linear-theory check.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L517–560; `datasets/drosophila_santomea_turing_wavelength_s0/pass1_metrics.json` L0–0
- **Source and data traceability: 86** — OpenAlex confirmed Turing 1952, Jeong 2008, Lachaise 2000, Llopart 2002, and Murray 1981. The Schnakenberg PII is cited with the Crossref 404 recorded rather than treated as resolved. The specimen SHA-256 is pinned.  
  evidence: `datasets/drosophila_santomea_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/drosophila_santomea_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 82** — LOCUS markers tie the Jacobian, dispersion, and growth integrator to Turing's k_c and to the chronoseries γ scaling. Pass 1 outputs do not yet exercise every marked block.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L1–45
- **Reproducibility package: 84** — --self-test checks the Jacobian numbers, the specimen hash, dispersion r > 0.98, and the wavelength order. Seed 3201 is fixed. Pass 1 is selected with --pass 1; the default path is the later pass.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L609–650

Remediation:
- Run the γ sweep and test that analytic wavelength falls as γ rises.
- Run the equal-diffusion ablation and record that σ(k_c) changes sign.
- Segment the gallery 100 µm scale bar and compare arm spacing with 2.326 µm.
- Report a KS distance and keep RMSE next to Pearson r.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila santomea Schnakenberg wavelength at 2.326 µm (Holocene, 0 Ma), pass 1
- **spec_fingerprint** (semantic model + inputs): `6dd554cc130d1b0176b37753cfbd7be017befc53ffb878bd36cf5f79710bf34c`
- **input_dataset_sha256**: `datasets/drosophila_santomea_turing_wavelength_s0/drosophila_santomea_specimen.jpg` (`9ea05ad5963c10d3…`), `datasets/drosophila_santomea_turing_wavelength_s0/drosophila_santomea_turing_wavelength_s0_reference.json` (`f16b54dd44acea38…`), `datasets/drosophila_santomea_turing_wavelength_s0/paper_citations.bib` (`fb23a9fc5fb610db…`)
- `datasets/drosophila_santomea_turing_wavelength_s0/pass1_comparison.png` — artifact_sha256 `5d3922c0cd42d528…`, 507,982 B, 1248×960 px
- `datasets/drosophila_santomea_turing_wavelength_s0/pass1_growth.gif` — artifact_sha256 `f1e17c66eeaa89ee…`, 3,546,417 B, 800×600 px, 12 frames
- `datasets/drosophila_santomea_turing_wavelength_s0/pass1_metrics.json` — artifact_sha256 `b74e3d57f7f2e4ed…`, 2,242 B, 
- `docs/media/drosophila_santomea_turing_wavelength_s0_comparison.png` — artifact_sha256 `c0f9011955e25636…`, 506,876 B, 1248×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| Schnakenberg dispersion Pearson r of numeric versus analytic growth rates | 1.0 Pearson_r | 1.0 | Eigenvector-initialized cosine modes; late-time log-amplitude slope versus the largest Jacobian eigenvalue at the discrete Laplacian wavenumber *(RMSE = 4.9e-05)* |
| D. santomea critical wavelength at gamma = 1 | 2.326 µm | 2.3 | λ_c = 2π ξ / k_c with ξ fixed so the continuous formula equals 2.326 µm *(2.326 ≠ 2.3; the reference is the one-decimal assignment and the value is the locked three-decimal wavelength)* |
| chronoseries wavelength gap santomea minus pseudoobscura at the 2.3 µm tier | 0.486 µm | 0.4 | 2326 nm − 1840 nm, with γ_pseudoobscura / γ_santomea = (2326/1840)^2 |
| chronoseries wavelength gap santomea minus sechellia at the 2.3 µm tier | 0.276 µm | 0.3 | 2326 nm − 2050 nm, with γ_sechellia / γ_santomea = (2326/2050)^2 |

## Environment at checkpoint


Pass 1 working tree. Evidence files are in this loop commit, not yet in 0c76d15d.
