# MorphoEval report — audit `2026-10-27-danio-aesculapii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `cfb8887378258c806d4412f6b1faac9a96299054` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio aesculapii cellular Turing wavenumber at 21 µm (Holocene, 0 Ma) (`models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py`) | 86 | 82 | 84 | 86 | 90 | 84 | 88 | 85.3 | 83.5 | 87.6 | 85.3 | **85.3** | faithful |

## Findings and remediation

### Danio aesculapii cellular Turing wavenumber at 21 µm (Holocene, 0 Ma)  —  overall 85.3 (faithful)

Pass 1. Anisotropic Schnakenberg system at λ_c = 2.08e-05 m (nominal 21 µm, log10 = -4.682), Holocene 0 Ma, chronoseries 1/5, successor D. albolineatus. Trace -0.5044, det 1.2544, R=28 above R_c=1.6725, equal-diffusion and successor (R=1.20) controls have negative diffusion-weighted trace and σ(k_c)<0. Dorsal field selects vertical bars at 16 px = 20.8 µm; ventral field breaks them into spots. Linear-regime Pearson r of log|û(k_c)| vs t is >0.999 with log-RMSE 0.025. Specimen body circularity 0.262 (reference circle 1), aspect 4.18, KS luminance 0.147. GIF 12 frames, 1437×480, cv_sim_similarity 93.2 against the CC BY-SA photograph. Weighted overall 85.3.

- **Model fidelity to source: 86** — The script implements the Turing dispersion eigenvalue, the classical k_c formula, and Schnakenberg kinetics, and the Lean module checks the scaled trace, determinant, aesculapii threshold, successor sign change, and discriminant with zero sorry. Deduction: the rate constants are a reconstruction that pins λ_c to the assigned 20.8 µm scale, not a parameter table measured on D. aesculapii, and argmax σ sits at a longer wave than formula k_c (both are drawn).  
  evidence: `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L96–168; `morphoeval/lean/loops/Loop_3160_danio_aesculapii_turing_wavelength_s0.lean` L10–51; `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L63–78
- **Morphological realism / specimen match: 82** — The last GIF frame keeps the photographed outline, eye, and dorsal iridophore row and lays vertical bars across the flank. Computer-vision similarity against the Andrew Bogott photograph is 93.2. Deduction: the bars are the magnified cellular field, so their count is not the adult bar count, and the reflection cut clips the tail tip.  
  evidence: `docs/media/danio_aesculapii_turing_wavelength_s0_growth.gif` L0–0; `docs/media/danio_aesculapii_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L384–410
- **Parameter expressivity: 84** — With λ_c held fixed, σ(k_c) rises across the diffusion-ratio sweep R = 8, 12, 18, 28, 44 (negative at R=8, positive and increasing after the threshold). Switching dorsal anisotropy for ventral isotropy changes bars into spots. The sweep is tabulated in the run, not drawn as its own panel.  
  evidence: `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L286–324; `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L733–739
- **Morphometric validation: 86** — Body circularity 0.262 versus a circle at 1, aspect 4.18 versus 1, luminance CV versus a flat field at 0, KS statistic 0.147 between render and specimen, stripe spacing 16 px versus the 20.8 µm anchor (different units, so not a self-comparison), and Pearson r > 0.999 versus a null of 0. A second CC BY plate is segmented separately. n=2 images.  
  evidence: `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L333–381; `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L622–701; `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L242–284
- **Source and data traceability: 90** — DOIs for Turing 1952, Nakamasu 2009, Yamanaka & Kondo 2014, Kullander & Fang 2009, Spiewak et al. 2018, Podobnik et al. 2020, Parichy 2015, and Irion & Nüsslein-Volhard 2019 are in the BibTeX. Both images are CC-licensed Wikimedia files with authors and SHA-256 pins checked at runtime. The synthetic species-gallery spiral was rejected.  
  evidence: `datasets/danio_aesculapii_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/danio_aesculapii_turing_wavelength_s0/danio_aesculapii_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Each modelling step is a named function cited from the module docstring, and the Lean declarations name the same scaled integers the self-test checks. The offset between formula k_c and argmax σ is drawn and not yet a measurement row.  
  evidence: `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L1–48; `morphoeval/math_kg_loops/delphi10000-danio-aesculapii-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 88** — Seed 3160, spectral diffusion, --self-test passes (Pearson r>0.98, both controls negative, sweep monotonic, SHA-256 pins). The default invocation writes the PNG and the 12-frame GIF. Deduction: --self-test does not itself open the GIF, so the frame-count and 480 px gates are enforced only on the render path.  
  evidence: `models/2026-delphi10000-danio-aesculapii-turing-wavelength-s0.py` L715–742; `datasets/danio_aesculapii_turing_wavelength_s0/README.md` L18–27

Remediation:
- {'id': 'R1', 'item': 'Emit formula k_c minus argmax σ as a measurement and assert its sign in --self-test. The figure already draws both lines.'}
- {'id': 'R2', 'item': 'Make --self-test open the GIF and assert n_frames >= 10 and min(width, height) >= 480, the same gates the render path already checks.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio aesculapii cellular Turing wavenumber at 21 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `243520b4201453b66768f25b30961995e65983baa128d9b7d5af4520fe5a690c`
- **input_dataset_sha256**: `datasets/danio_aesculapii_turing_wavelength_s0/danio_aesculapii_parichy2015_plate.jpg` (`26511cee521d38a6…`), `datasets/danio_aesculapii_turing_wavelength_s0/danio_aesculapii_specimen.png` (`6c6f62863f3ec000…`)
- `docs/media/danio_aesculapii_turing_wavelength_s0_comparison.png` — artifact_sha256 `7830914d2b5b6fb1…`, 473,586 B, 1632×912 px
- `docs/media/danio_aesculapii_turing_wavelength_s0_growth.gif` — artifact_sha256 `0ba99bf05f1b5478…`, 3,486,240 B, 1437×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen body circularity | 0.2618 4πA/P^2 | 1.0 | largest bright component on the CC BY-SA specimen photograph; morphological perimeter *(aspect ratio 4.182 (reference 1); foreground fraction 0.374)* |
| specimen flank aspect ratio | 4.1824 width/height | 1.0 | bounding box of the segmented D. aesculapii body |
| specimen image luminance CV | 0.3562 std/mean | 0.0 | coefficient of variation of luminance inside the body mask *(KS statistic render vs specimen luminance = 0.1474)* |
| cellular stripe spacing | 16.0 px | 20.8 | peak spacing of the dorsal activator transect; 1 px = 1.30 µm so 16 px = 20.8 µm *(spacing CV = 0.0000; RMSE of log-amplitude vs σ(k_c)t = 0.0248)* |
| linear-regime Pearson r of critical mode | 1.0 Pearson r | 0.0 | corr(log |û(k_c)|, t) on the window before nonlinear saturation *(fitted slope 0.2176 vs analytical σ(k_c)=0.2037; n=155)* |
| successor control sigma at k_c | -0.6401 1/time | 0.2037 | D. albolineatus inhibitor collapse R=1.20 at the same Du, evaluated at aesculapii k_c *(equal-diffusion control σ=-0.6048; sweep σ(k_c)=[-0.18417975347897686, -0.0009615607169632945, 0.11692151885858504, 0.20366366489600285, 0.26302269445588955])* |
| Parichy 2015 plate circularity | 0.6622 4πA/P^2 | 0.2618 | second CC BY plate (eLife 2015.05635) segmented independently of the primary photograph *(Parichy plate aspect 2.362)* |
| formula k_c minus argmax sigma | 0.0785 rad/px | 0.0 | argmax_k σ(k) minus the classical k_c = sqrt((D_v f_u + D_u g_v)/(2 D_u D_v)) *(argmax σ at k=0.3142; formula k_c=0.3927; σ_max=0.2435)* |

## Environment at checkpoint


Loop 3160 evidence files are in cfb88873. This audit's commit field was written immediately after that commit.
