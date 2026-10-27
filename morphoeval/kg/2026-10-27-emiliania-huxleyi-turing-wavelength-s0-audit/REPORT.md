# MorphoEval report — audit `2026-10-27-emiliania-huxleyi-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Emiliania huxleyi Holocene Turing wavelength at 5.808 nm (pass 1) (`models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py`) | 86 | 82 | 80 | 74 | 90 | 82 | 86 | 84.0 | 79.0 | 86.3 | 82.7 | **82.7** | faithful |

## Findings and remediation

### Emiliania huxleyi Holocene Turing wavelength at 5.808 nm (pass 1)  —  overall 82.7 (faithful)

Pass 1 implements Schnakenberg kinetics with the Turing peak fixed at 5.808 nm (0 Ma, extant). Selection-band Pearson r = 0.994, RMSE = 0.018. The first coccolith annulus used raw intensity and was a poor spoke estimator (spoke score 0.22, harmonic 40, centre on the panel edge). The damped wing was not yet reported separately. Those two items are the remediation list. The SEM is micrometre-scale; the field is 5.808 nm.

- **Model fidelity to source: 86** — The Jacobian, critical-wavenumber formula, and semi-implicit spectral integrator match the cited dispersion law. Kinetic parameters are chosen to hit 5.808 nm; they are not measured CAP rates.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L90–128; `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L137–180
- **Morphological realism / specimen match: 82** — The 12-frame 720x540 GIF reads as a coccosphere plus a radial coccolith and scores CV similarity 92.3 against the real SEM. Spoke count on this pass was not yet taken from a reliable annulus, so the shield geometry is only partly specimen-locked.  
  evidence: `docs/media/emiliania_huxleyi_turing_wavelength_s0_growth.gif` L0–0; `docs/media/emiliania_huxleyi_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 80** — Analytic k_c falls monotonically across six D_u values at ratio 18, and equal diffusion is stable. The pass did not yet publish the damped-wing error, so the unstable band is better exercised than the stable wing.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L260–281
- **Morphometric validation: 74** — Selection-band Pearson r = 0.994 against analytic growth rates (reference 1, so the comparison is not circular). The intensity annulus used for element count was low-confidence (spoke score 0.22).  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L231–242; `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Maini, Young, and Henriksen. Specimen is the Wikimedia CC BY-SA 4.0 SEM, SHA-256 pinned.  
  evidence: `datasets/emiliania_huxleyi_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — Module docstring and chronoseries helper state the wavelength law, the 0 Ma node, and the six-versus-eight mode reconstruction. The Lean file records the same integer identities.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L1–45; `morphoeval/lean/loops/Loop_3217_emiliania_huxleyi_turing_wavelength_s0.lean` L1–59
- **Reproducibility package: 86** — Seed 3217, --self-test, and fixed grid, dt, and step count. Self-test requires Pearson r > 0.98, a negative equal-diffusion control, and the chronoseries identities.  
  evidence: `models/2026-delphi10000-emiliania-huxleyi-turing-wavelength-s0.py` L666–701

Remediation:
- {'id': 'R1', 'item': 'Replace the raw-intensity coccolith annulus (spoke score 0.22, harmonic 40) with a tangential-Sobel annulus so the spoke count is actually measured on the SEM.'}
- {'id': 'R2', 'item': 'Report the damped-wing growth-rate RMSE separately and integrate that wing at a smaller timestep than the selection band.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.9]** `morphometric_validation` → `morphometric-validation`: Tangential-Sobel spoke count on the real coccolith.
- **[OPEN · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Separate damped-wing RMSE at a smaller dt.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Emiliania huxleyi Holocene Turing wavelength at 5.808 nm (pass 1)
- **spec_fingerprint** (semantic model + inputs): `ccb36351ddf24d5f1022c17c85f5291551afc2f963515079e7b0ebff6253d163`
- **input_dataset_sha256**: `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_coccosphere_coccolith.jpg` (`94d4b431c4912727…`), `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_reference.json` (`7eca0ef524dbb598…`)
- `datasets/emiliania_huxleyi_turing_wavelength_s0/emiliania_huxleyi_turing_wavelength_s0_metrics.json` — artifact_sha256 `495fe7d962e84e14…`, 4,141 B, 
- `docs/media/emiliania_huxleyi_turing_wavelength_s0_comparison.png` — artifact_sha256 `daeb9a91e4300345…`, 659,755 B, 1512×888 px
- `docs/media/emiliania_huxleyi_turing_wavelength_s0_growth.gif` — artifact_sha256 `c90f6524d06e8f07…`, 3,694,038 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (analytic vs integrator) | 0.99443 correlation | 1.0 | Pearson r of σ(k) on 8 wavenumbers, unstable eigenvector projection *(RMSE 0.0182; KS 0.125; p=4.31e-07)* |
| dispersion RMSE | 0.01822 1/time | 0.0 | Root mean square of analytic minus numeric growth rates |
| specimen foreground fraction | 0.6748 fraction | 0.4347 | Border-contrast mask on the CC BY-SA SEM *(reference is the simulation mask, not a copy of the specimen value)* |
| specimen contour circularity | 0.0431 dimensionless | 0.3292 | 4 pi A / P^2 on the specimen foreground mask |
| specimen structure-tensor coherency | 0.5871 dimensionless | 0.7435 | Mean (λ1-λ2)/(λ1+λ2) on the specimen foreground |
| specimen dominant angular harmonic | 30 cycles/revolution | 6 | Peak angular frequency (band 15-70) on the right-hand coccolith annulus *(phenotype map used 30 spokes; reference is not a copy of this harmonic)* |
| Holocene minus predecessor wavelength | -1.936 nm | 0.0 | 5.808 nm (8 modes) minus 7.744 nm (6 modes on the same 46.464 nm patch) *(predecessor spacing is reconstructed, not a fossil measurement)* |
| sweep k_c drop from smallest to largest D_u | 0.7088 1/nm | 0.0 | Monotone decline of analytic k_c across six D_u values at ratio 18 |

## Environment at checkpoint


Pass 1 of slot 3217. New files are not in this parent commit, so the knowledge-graph build uses --no-verify-loci. Lean 4 is not installed in this VM; the proofs are sorry-free integer identities and were not machine-checked here. Nanopub signing profile is absent.
