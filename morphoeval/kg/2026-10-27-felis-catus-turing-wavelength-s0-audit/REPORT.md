# MorphoEval report — audit `2026-10-27-felis-catus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 436 µm on Holocene Felis catus (pass 1) (`models/2026-delphi10000-felis-catus-turing-wavelength-s0.py`) | 76 | 74 | 58 | 64 | 90 | 80 | 84 | 70.0 | 70.2 | 84.9 | 74.6 | **74.6** | mostly faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 436 µm on Holocene Felis catus (pass 1)  —  overall 74.6 (mostly faithful)

Nominal Holocene point (a=0.073, b=1.186, Du=0.031, Dv=1.742, gamma=6) selects k*=6.135 with simulated dominant wavelength 431.3 µm against the 436 µm slot scale (about 1.1 percent). Equal diffusion is linearly stable. An 80 µm cellular predecessor cannot fit one wave (fundamental growth rate about -29). A 4360 µm successor flank holds ten wavelengths of the same k*; there is no second Felis chronospecies (series 1/1, age 0 Ma). The rendered patch is an isotropic spot lattice, not the mackerel stripes of the photographed tabby. Pass 1 does not yet sweep Dv, simulate an ablation, report specimen-segmentation statistics, or give Pearson/RMSE/KS for the dispersion relation.

- **Model fidelity to source: 76** — growth_rate implements the leading eigenvalue of J - k^2 D and fastest_mode selects k*. Kinetics are Schnakenberg, which is a standard realisation of Turing's law, not the Dkk4 circuit Kaelin identified. Nominal point only.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L89–105; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L137–179
- **Morphological realism / specimen match: 74** — The GIF's last frame is the Holocene tabby with a light pigment modulation, so it resembles the CC BY-SA photograph. The quantitative panel is a spot lattice on a black field; this cat's coat is broken mackerel striping. Scale transfer from 436 µm to the whole-animal photo is stated, not hidden.  
  evidence: `docs/media/felis_catus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/felis_catus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L234–247
- **Parameter expressivity: 58** — Du, Dv, a, b, and gamma are named constants and the equal-diffusion control is shown analytically to be stable, but no sweep yet demonstrates that Dv moves lambda monotonically.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L52–66; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L349–366
- **Morphometric validation: 64** — Dominant wavelength 431.3 µm is measured on the field and compared with the 436 µm slot scale. Cellular-predecessor growth rate is negative. Pearson r, RMSE, KS, and specimen segmentation statistics are not yet reported, so this score stays below the pass-2 bar.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L182–201; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L369–404
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing 1952, Schnakenberg 1979, Murray 1981, Kaelin 2012, and Kaelin 2021. Specimen is Von.grzanka 2010, CC BY-SA 3.0, SHA-256 pinned. The 436 µm scale is identified as the slot assignment, not as a number from those papers.  
  evidence: `datasets/felis_catus_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/felis_catus_turing_wavelength_s0/felis_catus_turing_wavelength_s0_reference.json` L1–80
- **Code-to-model traceability: 80** — The dispersion relation, steady state, and calibration sentence are in the module docstring and in growth_rate. Pass-1 figure panels are titled with the same lambda and age. Murray section numbers are not pinned line by line.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L89–97
- **Reproducibility package: 84** — Deterministic seed 3226, --self-test, and a direct python invocation. Dependencies are the repo scientific stack (numpy, scipy, matplotlib, pillow) and are not re-pinned in a private requirements file. Pass-1 self-test does not yet assert Pearson r.  
  evidence: `models/2026-delphi10000-felis-catus-turing-wavelength-s0.py` L349–366; `datasets/felis_catus_turing_wavelength_s0/README.md` L28–40

Remediation:
- {'id': 'R-sweep', 'item': 'Sweep inhibitor diffusion Dv and report k* and lambda at each value. Pass 1 only evaluates the nominal point.'}
- {'id': 'R-ablation', 'item': 'Simulate the equal-diffusion negative control and an ablation that removes the u^2 v term. Pass 1 only checks the analytic sign of sigma.'}
- {'id': 'R-segment', 'item': 'Report specimen CV segmentation metrics (foreground fraction, circularity, structure-tensor coherency, aspect) measured on the real photograph, with references that are not copies of the values.'}
- {'id': 'R-gof', 'item': 'Goodness of fit of the dispersion relation: Pearson r, RMSE, and a KS statistic between analytic sigma(k) and a numerical linear integration. Require Pearson r > 0.98.'}
- {'id': 'R-stripes', 'item': 'The isotropic patch is a spot lattice. The photographed cat is a broken mackerel tabby. Either bias the pattern toward stripes or keep the spot reading explicit.'}
- {'id': 'R-assessor', 'item': 'Single builder-assessor. No independent second assessor.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 436 µm on Holocene Felis catus (pass 1)
- **spec_fingerprint** (semantic model + inputs): `dc73277e26df9e4f2f17850cccebf7acc56dadcc0fc11952482db451621788f3`
- **input_dataset_sha256**: `datasets/felis_catus_turing_wavelength_s0/felis_catus_specimen.jpg` (`9360be6f1689a1df…`), `datasets/felis_catus_turing_wavelength_s0/felis_catus_turing_wavelength_s0_reference.json` (`a7694c1c70d27f08…`)
- `docs/media/felis_catus_turing_wavelength_s0_comparison.png` — artifact_sha256 `128af19622dfeb40…`, 715,098 B, 1344×1008 px
- `docs/media/felis_catus_turing_wavelength_s0_growth.gif` — artifact_sha256 `1a5b6367d29faa8a…`, 2,379,367 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_stripe_wavelength | 431.251 µm | 436.0 | radial FFT peak of the Schnakenberg activator on a 5-wave periodic patch, scaled so λ(k*) = 436 µm *(reference is the slot scale, not a Kaelin measurement; value is the simulated dominant wavelength)* |
| turing_sigma_star | 2.7561 1/time | 0.0 | max real eigenvalue of J - k^2 D over k in (0.15, 18) *(reference 0 is the marginal stability boundary; a positive value is the unstable Holocene point)* |
| cellular_predecessor_fundamental_sigma | -29.405 1/time | 2.7561 | σ(k_fund) on an 80 µm cellular-tier domain; Felis chronoseries has no earlier species *(reference is σ* of the 436 µm pattern; the cellular predecessor is stable)* |

## Environment at checkpoint


Pass 1 working tree. Checkpoint is the branch base; loop files are not in that commit yet.
