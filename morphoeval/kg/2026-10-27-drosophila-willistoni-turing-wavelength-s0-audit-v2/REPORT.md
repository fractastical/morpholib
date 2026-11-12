# MorphoEval report — audit `2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit -> 2026-10-27-drosophila-willistoni-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila willistoni subcellular Turing wavelength (1.5 µm, 0 Ma) (`models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py`) | 86 | 84 | 90 | 88 | 91 | 86 | 88 | 87.3 | 85.5 | 88.5 | 86.9 | **86.9** | faithful |

## Findings and remediation

### Drosophila willistoni subcellular Turing wavelength (1.5 µm, 0 Ma)  —  overall 86.9 (faithful)

Pass 2. The same Schnakenberg reconstruction is now fingerprinted. A six-point diffusivity sweep matches the analytical wavelength with Pearson r = 1 and RMSE at roundoff on commensurate modes. The noise-seeded field selects 1.480 µm against the 1.466 µm reference. Equal diffusivities drive the seeded stripe to a flat field. The Hanson abdomen is segmented; coherency, foreground fraction, and KS are recorded with values that are not copies of the references. Scale 1.5 µm, age 0 Ma. Virilis (predecessor) 1.676 µm and 7.40 granules per cell; willistoni 1.466 µm and 5.77; yakuba (successor) 1.509 µm and 6.09.

- **Model fidelity to source: 86** — Pass 2 keeps the Pass 1 reading of the Jacobian, onset wavenumber, and forward-Euler integrator, and now treats the ablation as executed. With Du = Dv the discriminant is negative, σ_max is negative, and the seeded stripe decays. The focal ratio 13.50 sits above the Schnakenberg onset ratio 11.37, and the fastest mode is 1.466 µm while the onset wavelength is 1.396 µm. That split is the supercritical shift, not a second ad-hoc target. The score stays below 90 because a, b, Du, and Dv are calibrated rather than measured in willistoni epidermis.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L82–128; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L197–250; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L289–304; `morphoeval/lean/loops/Loop_3211_drosophila_willistoni_turing_wavelength_s0.lean` L48–56
- **Morphological realism / specimen match: 84** — The GIF's last frame is the Hanson female with the granule field painted only on the segmented abdomen (eye and wing veins held fixed), at 720×540 and 12 frames. The comparison PNG shows the photograph, the composite, the activator labyrinth, and the flat ablation on one page. Computer-vision similarity against the photograph is high because the body, wing veins, and background are the specimen, which is what the 1.5 µm claim can honestly show at photographic resolution. The score is not higher because no electron micrograph of willistoni pigment granules is in the dataset.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L306–357; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L447–463; `docs/media/drosophila_willistoni_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_willistoni_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Raising the diffusivity scale from 0.75 to 1.55 lengthens the analytical and numerical wavelengths together, and the self-test rejects a non-monotone sweep. Body length moves both λ and granules per cell in the order virilis > yakuba > willistoni. The ablation is the zero of that family: equal diffusivities produce no pattern. The dispersion panel is limited to the unstable band so the shift is visible rather than crushed by the stable tails.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L134–163; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L252–287; `docs/media/drosophila_willistoni_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Commensurate-mode FFT wavelengths match the analytical sqrt scaling with Pearson r = 1, above the 0.98 gate, and RMSE at roundoff (reference tolerance 0.05 µm, not a copy of the RMSE). The noise-seeded hero field measures 1.480 µm against the 1.466 µm assignment. Abdomen KS D is 0.068 against a reference of 0. Specimen coherency and foreground fraction are measured on the rendered frame and referenced to the untouched photograph, and the two numbers are not equal. Granules per cell are referenced to the virilis count, not copied from the willistoni count.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L172–195; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L366–445; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L566–684
- **Source and data traceability: 91** — DOIs for Turing, Schnakenberg, Clark, Jeong, Kondo, and Wittkopp resolve at Crossref. The specimen is MA Hanson's CC BY-SA 4.0 TIFF, with the RGB PNG used for segmentation, both SHA-256 pinned in the reference JSON along with the README, bibliography, and scout manifest. The scout records that campaign slot 3211 does not exist and that the 86 µm sibling is a different sim id.  
  evidence: `datasets/drosophila_willistoni_turing_wavelength_s0/paper_citations.bib` L1–90; `datasets/drosophila_willistoni_turing_wavelength_s0/drosophila_willistoni_turing_wavelength_s0_reference.json` L78–120; `datasets/drosophila_willistoni_turing_wavelength_s0/README.md` L1–25
- **Code-to-model traceability: 86** — Each modelling step is a named function whose docstring states the equation it implements. The Lean module proves the integer form of the wavelength order, the granule-count order, the onset margin, and the ablation margin, with no sorry. The math-kg file points at those declarations and at the Python lines. Calibration constants remain reconstructions, which the reference JSON says in writing.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3211_drosophila_willistoni_turing_wavelength_s0.lean` L25–60; `morphoeval/math_kg_loops/delphi10000-drosophila-willistoni-turing-wavelength-s0.yaml` L28–72
- **Reproducibility package: 88** — python3 --self-test passed (hashes, λ_max within 0.01 µm, chronoseries order, Lean pins, r > 0.98, RMSE < 0.05, ablation ratio < 0.25, non-circular specimen metrics). The default run writes a 12-frame 720×540 GIF and the comparison PNG. RNG seed 3211. Dependencies are the repository scientific stack (numpy, scipy, Pillow, matplotlib). Nanopub signing is not available in this environment (no ~/.nanopub/profile.yml); that is recorded rather than simulated.  
  evidence: `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L691–753; `models/2026-delphi10000-drosophila-willistoni-turing-wavelength-s0.py` L755–790; `docs/media/drosophila_willistoni_turing_wavelength_s0_growth.gif` L0–0

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.7]** `parameter_sweep` → `parameter-expressivity`: Diffusivity-scale sweep fingerprinted; wavelength is monotone and Pearson r = 1 on commensurate modes.
- **[RESOLVED · P=1.5]** `parameter_sweep` → `model-fidelity`: Equal-diffusivity ablation decays the seeded stripe; std ratio versus the Turing field is 0.
- **[RESOLVED · P=1.9]** `specimen_comparison` → `specimen-match`: Specimen coherency, foreground fraction, and abdomen KS are non-circular measurements against the Hanson photograph.
- **[RESOLVED · P=1.7]** `morphometric_validation` → `morphometric-validation`: Pearson r = 1 (> 0.98), RMSE at roundoff, noise-seeded wavelength 1.480 µm versus 1.466 µm, abdomen KS D = 0.068.

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


Pass 2 re-scores after the sweep, ablation, specimen segmentation, and goodness-of-fit were executed. New loop files are introduced by this loop's commit; locus verification uses --no-verify-loci until that commit exists.
