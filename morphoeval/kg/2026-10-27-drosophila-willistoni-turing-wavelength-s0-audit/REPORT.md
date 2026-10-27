# MorphoEval report — audit `2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila willistoni subcellular Turing wavelength (1.5 µm, 0 Ma) (`models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py`) | 80 | 74 | 70 | 68 | 88 | 82 | 84 | 76.7 | 71.8 | 84.8 | 77.1 | **77.1** | mostly faithful |

## Findings and remediation

### Drosophila willistoni subcellular Turing wavelength (1.5 µm, 0 Ma)  —  overall 77.1 (mostly faithful)

Pass 1 of slot 3211. Schnakenberg kinetics on a periodic 8.796 µm patch, calibrated so the fastest mode is the assigned 1.5 µm (1.466e-6 m) detail scale at geological age 0 Ma. Predecessor Drosophila virilis and successor Drosophila yakuba keep the same reaction terms and diffusivity ratio; absolute diffusivities follow body-length^0.35. The Hanson 2019 photograph is the specimen. Full credit for the sweep, ablation, segmentation, and goodness-of-fit waits on the measurements sidecar and the image-fitness test.

- **Model fidelity to source: 80** — The script implements Schnakenberg kinetics, the steady-state Jacobian, the classical onset wavenumber, and the growth-rate eigenvalue. Parameters are a reconstruction constrained to 1.466 µm, which is declared, so this is not a transcription of a willistoni rate table. Pass 1 withholds the top of the scale until the ablation control is in the measurements sidecar.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L82–128; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L165–250; `datasets/drosophila_willistoni_turing_wavelength_s0/drosophila_willistoni_turing_wavelength_s0_reference.json` L32–52
- **Morphological realism / specimen match: 74** — The comparison figure places the Hanson photograph beside the abdomen composite, and the growth GIF is a 12-frame 720×540 animation on that photograph. The raw activator field is a spot/labyrinth pattern and the ablation panel is flat. Pass 1 does not yet treat segmentation metrics as fingerprinted, and the photograph cannot resolve 1.5 µm granules.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L306–357; `docs/media/drosophila_willistoni_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_willistoni_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 70** — Diffusivity scale, body-size allometry, and the equal-diffusivity ablation are in the script, and the figure has a sweep panel and a three-species bar chart. Pass 1 does not award full credit until measure_outputs records the monotone sweep.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L130–163; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L252–304
- **Morphometric validation: 68** — The self-test enforces Pearson r > 0.98, RMSE < 0.05 µm, wavelength order, and non-equal specimen metrics. Those numbers are not yet in a measurements sidecar, so Pass 1 holds this dimension down. The commensurate-mode sweep is expected to sit on the analytical line; the noise-seeded field is the independent wavelength residual.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L172–195; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L566–684; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L691–753
- **Source and data traceability: 88** — Turing, Schnakenberg, Clark, Jeong, Kondo, and the Wikimedia file have resolvable identifiers. The TIFF and the RGB PNG are SHA-256 pinned, as are the README, bibliography, and scout manifest. The scout notes that slot 3211 is outside campaign_100_slots.yaml and was queried directly. No collision with the 86 µm sibling slot.  
  evidence: `datasets/drosophila_willistoni_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_willistoni_turing_wavelength_s0/drosophila_willistoni_turing_wavelength_s0_reference.json` L78–105; `datasets/drosophila_willistoni_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 82** — The module docstring states the PDEs, the onset formula, the allometry, and the refusal to copy slot 2931. Functions match those equations. The numerical values of a, b, Du, and Dv are calibration outputs rather than equation numbers in Clark 2007, which is the right claim boundary and the reason this is not a higher score.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L1–30; `morphoeval/lean/loops/Loop_3211_drosophila_willistoni_turing_wavelength_s0.lean` L8–60; `morphoeval/math_kg_loops/delphi10000-drosophila-willistoni-turing-wavelength-s0.yaml` L28–70
- **Reproducibility package: 84** — --self-test checks hashes, the fastest mode, chronospecies order, the Lean integer pins, Pearson r, RMSE, ablation, and non-circular specimen metrics. The default invocation writes the PNG and the GIF. The run is seeded. Pass 1 has not yet re-executed measure_outputs.py.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L69–80; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L691–790; `datasets/drosophila_willistoni_turing_wavelength_s0/README.md` L1–40

Remediation:
- Fingerprint the diffusivity-scale sweep (Pearson r, RMSE, monotonicity) in the measurements sidecar.
- Fingerprint the equal-diffusivity ablation (negative control) beside the Turing field.
- Record non-circular specimen CV metrics: abdomen mask, structure-tensor coherency, foreground fraction, and KS against the Hanson photograph.
- Record goodness-of-fit (Pearson r > 0.98, RMSE, KS) and run the image-fitness test on the growth GIF.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.7]** `parameter_sweep` → `parameter-expressivity`: Fingerprint the diffusivity-scale sweep and show that wavelength moves monotonically.
- **[OPEN · P=1.5]** `parameter_sweep` → `model-fidelity`: Fingerprint the equal-diffusivity ablation as a negative control.
- **[OPEN · P=1.9]** `specimen_comparison` → `specimen-match`: Record non-circular specimen segmentation metrics (coherency, foreground, KS) against the Hanson photograph.
- **[OPEN · P=1.7]** `morphometric_validation` → `morphometric-validation`: Record Pearson r, RMSE, and KS for the wavelength law and the abdomen intensities.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila willistoni subcellular Turing wavelength (1.5 µm, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `db43d14901fb6b4601bb2d4fcd69a0bd3c5388cab0d6eb64de70aadd7ba01054`
- `docs/media/drosophila_willistoni_turing_wavelength_s0_comparison.png` — artifact_sha256 `f0b879c8a574ac06…`, 447,136 B, 1752×1128 px
- `docs/media/drosophila_willistoni_turing_wavelength_s0_growth.gif` — artifact_sha256 `481ccd337d573ff3…`, 1,833,875 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated ocellus dominant wavelength | 1.479976 µm | 1.466 | sub-bin radial FFT of the noise-seeded Schnakenberg field *(sub-bin estimate is not forced equal to the reference bin centre)* |
| onset wavelength k_c | 1.395551 µm | 1.466 | k_c^2 = gamma (Dv fu + Du gv) / (2 Du Dv) *(supercritical fastest mode is longer than the onset wavenumber)* |
| wavelength sweep pearson r | 1.0 1 | 0.98 | pearsonr(analytical λ, FFT λ) across six diffusivity scales *(gate is r > 0.98; the reference stored here is that gate, not the measured r)* |
| wavelength sweep rmse | 0.0 µm | 0.05 | RMSE of FFT wavelength against the analytical sqrt scaling *(reference is the acceptance tolerance, not a copy of the RMSE)* |
| specimen abdomen pigment KS statistic | 0.068359 1 | 0.0 | ks_2samp of abdomen grayscale before and after granule modulation *(KS D = 0 would mean the modulation left intensities unchanged)* |
| specimen structure-tensor coherency | 0.560967 1 | 0.569752 | mean structure-tensor coherency on the body mask *(simulation value is measured on the rendered frame; reference is the untouched photograph)* |
| specimen foreground fraction | 0.106667 1 | 0.32287 | segmented abdomen area fraction versus border-median foreground of the whole photograph *(the abdomen is a subset of the fly, so the two fractions are different measurements)* |
| ablation control std ratio versus turing | 0.0 1 | 1.0 | std(u | Du=Dv) / std(u | δ=13.5) after the same stripe seed *(a ratio near 1 would mean the ablation failed to suppress the pattern)* |
| virilis predecessor wavelength | 1.675833 µm | 1.466 | λ = 1.466 * (L_body / 2.32 mm)^0.35 *(longer than the focal 1.466 µm wavelength)* |
| yakuba successor wavelength | 1.509049 µm | 1.466 | λ = 1.466 * (L_body / 2.32 mm)^0.35 *(intermediate between willistoni and virilis)* |
| willistoni granules across epidermal cell | 5.770805 1 | 7.398275 | cell_diameter / λ with cell_diameter linear in body length *(reference is the virilis predecessor count, which is larger)* |
| specimen abdomen intensity pearson r | 0.924906 1 | 1.0 | pearsonr of abdomen pixels, photograph versus modulated frame *(r = 1 would mean the granule field was not painted on)* |

## Environment at checkpoint


Pass 1 scored before measure_outputs fingerprinting and before the image-fitness gate. New loop files are not in this commit.
