# MorphoEval report — audit `2026-10-27-drosophila-kikkawai-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila kikkawai Turing wavelength at 2.3 µm (detail, 0 Ma) (`models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py`) | 86 | 78 | 84 | 80 | 88 | 84 | 86 | 85.3 | 78.8 | 86.1 | 82.9 | **82.9** | faithful |

## Findings and remediation

### Drosophila kikkawai Turing wavelength at 2.3 µm (detail, 0 Ma)  —  overall 82.9 (faithful)

Pass 1. Schnakenberg activator-substrate kinetics on a 46.86 µm periodic patch, with inhibitor diffusivity set so the linear growth-rate peak is 2.344 µm (campaign detail scale 2.343e-06 m, Holocene 0 Ma). A single-mode PDE growth assay tracks that peak across five diffusivities at Pearson r = 0.9993. The saturated hero field that is drawn, however, has a radial FFT wavelength of 2.182 µm (6.9% short of 2.343 µm). Chronoseries at the same a, b, Du: D. mauritiana 1.820 µm, D. kikkawai 2.344 µm, D. hydei 3.056 µm. GIF 12 frames at 640×480, cv_sim_similarity 94.4 against the pinned gallery plate. Wikimedia Commons had no species photograph.

- **Model fidelity to source: 86** — The script integrates the Schnakenberg PDE and the Turing growth-rate peak, with parameters derived so the linear peak sits on 2.344 µm. The field that is actually rendered saturates at an FFT wavelength of 2.182 µm, so the drawn pattern is not yet the critical mode.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L86–107; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L138–167; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L212–233
- **Morphological realism / specimen match: 78** — The 12-frame 640×480 GIF scores cv_sim_similarity 94.4 against the pinned plate because the render fades the activator onto that plate. The plate is a repository gallery image with a 100 µm bar, not a 2.3 µm micrograph, so the visual match is to the envelope rather than to a measured carrier.  
  evidence: `docs/media/drosophila_kikkawai_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L366–394
- **Parameter expressivity: 84** — Inhibitor diffusivity, substrate load b, and the equal-diffusion ablation each move a different observable: wavelength along the chronoseries, dark-versus-light wavelength, and pattern amplitude. The comparison figure shows all three.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L109–120; `docs/media/drosophila_kikkawai_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 80** — Pearson r = 0.9993 between the analytic peak and the integer mode the PDE amplifies, with RMSE 0.193 µm. Orientation KS against the plate is 0.110 and the two coherencies are not equal. The hero field's own FFT (2.182 µm versus 2.343 µm) is the measurement that still fails a 3% band.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L169–210; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L302–330; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L288–292
- **Source and data traceability: 88** — Five DOIs were checked on Crossref or OpenAlex on 2026-10-27, including Gibert 1999 for this species. The sibling slot's Mol. Biol. Evol. DOI was rejected because it resolves to a plant paper. The specimen SHA-256 is pinned. Commons had no species photograph, and that miss is written in the scout manifest.  
  evidence: `datasets/drosophila_kikkawai_turing_wavelength_s0/paper_citations.bib` L1–53; `datasets/drosophila_kikkawai_turing_wavelength_s0/drosophila_kikkawai_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_kikkawai_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — Jacobian, integrator, mode assay, ablation, and render are separate functions. Lean theorems state the wavelength order, the mode count, the diffusion gap, and the dark-morph activator inequality. The render weights are constants inside render_gif rather than entries read back from the reference JSON.  
  evidence: `morphoeval/lean/loops/Loop_3190_drosophila_kikkawai_turing_wavelength_s0.lean` L14–48; `morphoeval/math_kg_loops/delphi10000-drosophila-kikkawai-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — Self-test checks the wavelength band, chronoseries order, ablation, Pearson gate, GIF geometry, CV gate, and the Lean text. Seed 3190 and the specimen digest are fixed. Pass 1 had not yet been re-measured by measure_outputs.py.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L72–84; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L549–610; `datasets/drosophila_kikkawai_turing_wavelength_s0/README.md` L1–29

Remediation:
- {'id': 'R1', 'item': "Seed the hero activator on Fourier mode m=20 so the rendered field's radial FFT wavelength lies within 3% of 2.343 µm. Pass 1 measures 2.182 µm on that field."}
- {'id': 'R2', 'item': 'Put the corrected hero wavelength in the comparison-figure title and keep the pure field visible beside the plate.'}
- {'id': 'R3', 'item': 'Re-run image fitness after the retune. cv_sim_similarity must stay at or above 58 and the GIF must stay at least 12 frames and 480 px.'}
- {'id': 'R4', 'item': 'Record signing as unavailable if ~/.nanopub/profile.yml is still absent. Do not invent a trusty URI.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila kikkawai Turing wavelength at 2.3 µm (detail, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `dab1edfd3ce2230430dac30458215ff2efe58254b8b6a2e95b6d00930fe73df6`
- **input_dataset_sha256**: `datasets/drosophila_kikkawai_turing_wavelength_s0/Drosophila_kikkawai_specimen.jpg` (`1025334f37e7770a…`), `datasets/drosophila_kikkawai_turing_wavelength_s0/drosophila_kikkawai_turing_wavelength_s0_reference.json` (`45a01a384a7d9fe1…`), `datasets/drosophila_kikkawai_turing_wavelength_s0/paper_citations.bib` (`37400061d5790447…`), `datasets/drosophila_kikkawai_turing_wavelength_s0/scout_manifest.json` (`cfe6e4b2724a3595…`)
- `docs/media/drosophila_kikkawai_turing_wavelength_s0_comparison.png` — artifact_sha256 `5a94896a106df197…`, 536,437 B, 1512×984 px
- `docs/media/drosophila_kikkawai_turing_wavelength_s0_growth.gif` — artifact_sha256 `cf30dcbbcfa9c3cf…`, 2,318,215 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated tergite stripe wavelength | 2.1824 µm | 2.343 | radial FFT peak of the focal Schnakenberg activator on the 46.86 µm periodic patch *(analytical growth-rate peak 2.3437 µm)* |
| dispersion-sweep Pearson r of analytical versus PDE wavelength | 0.9993 r | 0.98 | pearsonr across Dv in {0.413, 0.800, 1.364, 2.400, 4.470}; analytical λ_σ versus the integer mode with the largest PDE growth rate *(RMSE 0.1931 µm; KS orientation 0.1099)* |
| specimen image structure-tensor coherency | 0.6481 coherency | 0.4391 | mean structure-tensor coherency of the pinned Drosophila kikkawai plate *(value is the plate; reference_value is the simulation, so the pair is non-circular)* |
| specimen-versus-simulation orientation KS distance | 0.1099 KS | 0.25 | Kolmogorov-Smirnov distance between 12-bin Sobel orientation histograms of plate and activator *(seed-to-seed CV of FFT λ = 0.0973)* |
| predecessor hydei detail wavelength | 3.0556 µm | 1.8196 | growth-rate peak at Dv=4.470, same a, b, Du as the focal species *(ordering mauritiana < kikkawai < hydei at 0 Ma)* |
| equal-diffusion ablation amplitude ratio | 102.0346 std(u)_Turing / std(u)_equal | 1.0 | matched-horizon PDE, Du=Dv=0.020 versus Dv=1.364 *(linear σ at λ=2.343 µm under equal diffusion = -0.2232)* |

## Environment at checkpoint


Pass 1 baseline. New slot files are not in this parent commit; the knowledge-graph build uses --no-verify-loci and records that limit.
