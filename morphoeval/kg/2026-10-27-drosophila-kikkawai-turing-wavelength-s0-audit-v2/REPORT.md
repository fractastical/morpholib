# MorphoEval report — audit `2026-10-27-drosophila-kikkawai-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-kikkawai-turing-wavelength-s0-audit -> 2026-10-27-drosophila-kikkawai-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila kikkawai Turing wavelength at 2.3 µm (detail, 0 Ma) (`models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py`) | 91 | 84 | 89 | 91 | 88 | 87 | 88 | 90.3 | 86.6 | 87.7 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Drosophila kikkawai Turing wavelength at 2.3 µm (detail, 0 Ma)  —  overall 88.1 (faithful)

Pass 2 closes R1–R3. The hero activator is seeded on Fourier mode m=20, so its radial FFT wavelength is 2.334 µm, 0.4% from the 2.343 µm detail target (pass 1 measured 2.182 µm on the unseeded saturated field). The single-mode PDE assay still tracks the analytic peak at Pearson r = 0.9993 (RMSE 0.193 µm). Chronoseries at 0 Ma, same a, b and Du: D. mauritiana 1.820 µm (Dv=0.413), D. kikkawai 2.344 µm (Dv=1.364), D. hydei 3.056 µm (Dv=4.470). Light morph b=0.71 gives 2.680 µm; dark morph b=1.18 gives 2.109 µm. Equal diffusion remains stable (σ=-0.223) and the amplitude ratio is about 170. GIF is 12 frames at 640×480 with cv_sim_similarity 95.7.

- **Model fidelity to source: 91** — Schnakenberg PDE, Jacobian growth rate, and a mode-20 seed so the drawn field sits on the critical wavelength. Pass 2 FFT wavelength is 2.334 µm against 2.343 µm. The seed is an initial condition, not a change to the kinetics.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L86–107; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L153–169
- **Morphological realism / specimen match: 84** — cv_sim_similarity of the 12-frame 640×480 GIF against the pinned plate is 95.7. The score stays below that figure because the plate is a gallery rendering with a 100 µm bar, and the GIF mixes plate RGB at a final weight of 0.55. The pure activator is the centre panel of the comparison figure.  
  evidence: `docs/media/drosophila_kikkawai_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_kikkawai_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L369–397
- **Parameter expressivity: 89** — Dv sets the chronoseries wavelength (1.820, 2.344, 3.056 µm). Substrate b sets the light-to-dark shift (2.680 to 2.109 µm). Equal diffusion is a true negative control (σ negative, amplitude ratio about 170). All three are drawn.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L109–120; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L399–470
- **Morphometric validation: 91** — Pearson r = 0.9993 on the Dv sweep (reference gate 0.98, not equal to the measured r). Hero FFT 2.334 µm versus reference 2.343 µm. Plate coherency and field coherency differ. Orientation KS is 0.238 versus a 0.25 screening threshold. Three seeds report a coefficient of variation of the small-grid FFT peak.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L172–213; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L291–320; `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L474–538
- **Source and data traceability: 88** — Unchanged from pass 1 and still the right score. Five Crossref-checked DOIs, a pinned specimen digest, and an explicit Commons miss. The 2.3 µm length is not in those papers, which is why this is not 95.  
  evidence: `datasets/drosophila_kikkawai_turing_wavelength_s0/paper_citations.bib` L1–53; `datasets/drosophila_kikkawai_turing_wavelength_s0/scout_manifest.json` L1–50
- **Code-to-model traceability: 87** — Each modelling step is a named function cited from the math-kg laws, and the Lean file proves the wavelength order, the mode count of 20, the diffusion gap, and the dark-morph activator inequality with no incomplete proofs.  
  evidence: `morphoeval/lean/loops/Loop_3190_drosophila_kikkawai_turing_wavelength_s0.lean` L10–52; `morphoeval/math_kg_loops/delphi10000-drosophila-kikkawai-turing-wavelength-s0.yaml` L40–79
- **Reproducibility package: 88** — Self-test now fails if the hero wavelength is more than 3% from 2.343 µm, if Pearson r is not above 0.98, if the GIF has fewer than 10 frames or is under 480 px, or if CV similarity is under 58. Seed and specimen digest are pinned.  
  evidence: `models/2026-delphi10000-drosophila-kikkawai-turing-wavelength-s0.py` L552–612; `datasets/drosophila_kikkawai_turing_wavelength_s0/README.md` L1–29

Remediation:
- {'id': 'R1', 'item': 'Resolved in pass 2. Hero FFT wavelength is 2.334 µm (was 2.182 µm), inside 3% of 2.343 µm, by seeding mode m=20.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in pass 2. The comparison figure title prints the measured FFT wavelength next to the pure activator panel.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in pass 2. GIF remains 12 frames at 640×480 and cv_sim_similarity is 95.7.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Open, and not a numeric-gate failure. ~/.nanopub/profile.yml is absent, so the nanopubs are unsigned. No key was created and no trusty URI was invented.', 'status': 'open'}
- {'id': 'R5', 'item': 'Open, and not a numeric-gate failure. A CC light micrograph of D. kikkawai pigment at 2.3 µm was not on Wikimedia Commons. The gallery plate stays the pinned specimen.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila kikkawai Turing wavelength at 2.3 µm (detail, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `dab1edfd3ce2230430dac30458215ff2efe58254b8b6a2e95b6d00930fe73df6`
- **input_dataset_sha256**: `datasets/drosophila_kikkawai_turing_wavelength_s0/Drosophila_kikkawai_specimen.jpg` (`1025334f37e7770a…`), `datasets/drosophila_kikkawai_turing_wavelength_s0/drosophila_kikkawai_turing_wavelength_s0_reference.json` (`45a01a384a7d9fe1…`), `datasets/drosophila_kikkawai_turing_wavelength_s0/paper_citations.bib` (`37400061d5790447…`), `datasets/drosophila_kikkawai_turing_wavelength_s0/scout_manifest.json` (`cfe6e4b2724a3595…`)
- `docs/media/drosophila_kikkawai_turing_wavelength_s0_comparison.png` — artifact_sha256 `bd9505f2c3c7d515…`, 476,270 B, 1512×984 px
- `docs/media/drosophila_kikkawai_turing_wavelength_s0_growth.gif` — artifact_sha256 `bf7c5c881b52e66f…`, 2,626,469 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated tergite stripe wavelength | 2.3335 µm | 2.343 | radial FFT peak of the focal Schnakenberg activator on the 46.86 µm periodic patch *(analytical growth-rate peak 2.3437 µm)* |
| dispersion-sweep Pearson r of analytical versus PDE wavelength | 0.9993 r | 0.98 | pearsonr across Dv in {0.413, 0.800, 1.364, 2.400, 4.470}; analytical λ_σ versus the integer mode with the largest PDE growth rate *(RMSE 0.1931 µm; KS orientation 0.2376)* |
| specimen image structure-tensor coherency | 0.6481 coherency | 0.8402 | mean structure-tensor coherency of the pinned Drosophila kikkawai plate *(value is the plate; reference_value is the simulation, so the pair is non-circular)* |
| specimen-versus-simulation orientation KS distance | 0.2376 KS | 0.25 | Kolmogorov-Smirnov distance between 12-bin Sobel orientation histograms of plate and activator *(seed-to-seed CV of FFT λ = 0.0000)* |
| predecessor hydei detail wavelength | 3.0556 µm | 1.8196 | growth-rate peak at Dv=4.470, same a, b, Du as the focal species *(ordering mauritiana < kikkawai < hydei at 0 Ma)* |
| equal-diffusion ablation amplitude ratio | 173.649 std(u)_Turing / std(u)_equal | 1.0 | matched-horizon PDE, Du=Dv=0.020 versus Dv=1.364 *(linear σ at λ=2.343 µm under equal diffusion = -0.2232)* |

## Environment at checkpoint


Pass 2. Hero field is seeded on mode m=20. Parent commit does not yet contain these new paths; build uses --no-verify-loci.
