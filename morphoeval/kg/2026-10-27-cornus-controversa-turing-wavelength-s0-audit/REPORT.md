# MorphoEval report — audit `2026-10-27-cornus-controversa-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cornus controversa Turing wavenumber at 1.939 mm (Holocene, 0 Ma) (`models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py`) | 90 | 72 | 86 | 82 | 88 | 84 | 86 | 88.7 | 75.8 | 86.1 | 82.7 | **82.7** | faithful |

## Findings and remediation

### Cornus controversa Turing wavenumber at 1.939 mm (Holocene, 0 Ma)  —  overall 82.7 (faithful)

Pass 1. Schnakenberg kinetics with a=0.06, b=0.94 (trace -0.120, det 1) and diffusivities solved so the continuum critical wavelength is 1.939 mm (0.001939 m, log10 -2.712). Eleven real-eigenvalue Fourier modes give Pearson r = 0.9999998 and RMSE = 2.13e-4 against analytic sigma(k). The equal-diffusion control has sigma_max = -0.0615 and std ratio 0.018. A Dv sweep from 0.65 to 2.4 times nominal moves lambda monotonically from 1.998 mm to 1.881 mm. The saturated 8-wavelength patch peaks at 2.216 mm, the neighboring Fourier bin. The crown photograph has no scale bar. PASS2_REMEDIATION is False, so the 16-wavelength remeasure, the horizontal bias, and the resolution block are not in this run. Weighted score 82.7/100.

- **Model fidelity to source: 90** — Equations (1)–(6) in the module docstring are the Schnakenberg PDE, the steady state, the Jacobian, and Turing's k_c. solve_diffusivities returns lambda = 1.939000 mm and a self-test checks trace < 0, det > 0, Dv > Du, and a closed band when Du = Dv. Modes whose eigenvalues are a complex pair are excluded, because |u-hat| is then not a pure exponential; the eleven real modes that remain track sigma(k). Deduction: a and b were chosen for this slot so the wavelength would land on 1.939 mm. They are not parameters reported for Cornus.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L90–155; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L157–172; `morphoeval/lean/loops/Loop_3155_cornus_controversa_turing_wavelength_s0.lean` L10–44
- **Morphological realism / specimen match: 72** — The comparison figure shows the Meise crown beside the dispersion fit and the lamina field, and the GIF has 12 frames at 480x640 whose last frame is that photograph (CV similarity 96.0 against the specimen file). The generative field itself is a spot lattice on a square, not a cream leaf margin and not the pagoda tiers. Foreground fraction of the photo is 0.866; bright foliage above the lawn is 0.212. Those are crown statistics. Score stays in the 70s because the organism is on the page while the mechanism's picture is not the organism's picture.  
  evidence: `docs/media/cornus_controversa_turing_wavelength_s0_comparison.png` L0–0; `docs/media/cornus_controversa_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L581–629
- **Parameter expressivity: 86** — lambda_of_dv moves monotonically as Dv/Dv* runs through 0.65, 0.80, 1.00, 1.35, 1.80, 2.40: wavelengths 1.998, 1.966, 1.939, 1.913, 1.894, 1.881 mm. The nominal factor returns the assignment. The equal-diffusion ablation is a separate switch, not a point on that curve. The horizontal bias exists in integrate() but PASS2_REMEDIATION is False, so this pass does not exercise it.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L175–183; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L318–325; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L82–83
- **Morphometric validation: 82** — Linear probe: Pearson r = 0.9999998 (n = 11 real modes), RMSE = 2.13e-4 per time unit, against reference bar 0.98. KS statistic of the residuals versus Normal(0, s) is 0.514 (p = 0.003): the residuals are structured truncation error, not a normal sample, and that failure is reported rather than dropped. Nonlinear dominant wavelength on the 8-lambda box is 2.216 mm versus 1.939 mm (ratio 1.14), i.e. Fourier bin 7 instead of bin 8. Ablation std ratio 0.018 with sigma_max -0.0615. Specimen foreground 0.866 versus null 0.50, foliage fraction 0.212 versus null 0.25, horizontal-edge fraction 0.433 versus the field's 0.421. The crown frame is 960x1280 and has no scale bar, so none of those photo numbers is a millimetre. The resolution block is not written on this pass.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L236–283; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L327–347; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L381–415; `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 88** — Eight DOIs were checked against Crossref on 2026-10-27, including Turing 1952, Schnakenberg 1979, and Xiang et al. 2006. The specimen SHA-256 be3da2bb8b817bec105efa4dec6064d234f1c47ab458bdf8059d885529dccff4 matches the species-gallery file and is pinned in the reference JSON. License recorded as public domain (Meise / Wikimedia). Deduction: slot 3155 is absent from campaign_100_slots.yaml, so the scout CLI could not be pointed at --slot 3155; OpenAlex lexical search did not return the seed papers, which were resolved on Crossref instead.  
  evidence: `datasets/cornus_controversa_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/cornus_controversa_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — jacobian, solve_diffusivities, growth_rate, integrate, and ablation_control are named for equations (1)–(6). The Lean module uses the same scaled integers (trace -120 per thousand, wavelength product 1939*3240, equal-diffusion contrast). The proofs are kernel `decide` witnesses on those integers, not a formal PDE.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L197–224; `morphoeval/math_kg_loops/delphi10000-cornus-controversa-turing-wavelength-s0.yaml` L1–60
- **Reproducibility package: 86** — --self-test checks the hash, the band, the monotonic sweep, Pearson r > 0.98, RMSE < 0.05, and the decaying ablation. Seed is 3155. Full run writes the PNG, the 12-frame GIF, and the metrics JSON. Deduction: dependency versions are the environment's scientific stack, not a per-script lockfile, and the checkpoint commit does not yet contain these new paths.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L803–842; `datasets/cornus_controversa_turing_wavelength_s0/README.md` L1–31

Remediation:
- {'id': 'R1', 'item': 'Nonlinear peak is 2.216 mm on the 8-wavelength box (Fourier bin 7). Re-measure on a 16-wavelength patch before treating 1.939 mm as the saturated wavelength.'}
- {'id': 'R2', 'item': 'Horizontal midrib bias is implemented but PASS2_REMEDIATION is False, so orientation is not exercised as a parameter.'}
- {'id': 'R3', 'item': 'resolution_limit is null. Record that the crown photograph has no scale bar and that 1.939 mm is not read off those pixels.'}
- {'id': 'R4', 'item': "No second assessor. This score is the builder's own."}
- {'id': 'R5', 'item': 'No ~/.nanopub/profile.yml. Do not invent a signature.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cornus controversa Turing wavenumber at 1.939 mm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `889436d47b7f6a20b524144198f645dcccbc348d99fd9ce666f715dfafc2a48a`
- **input_dataset_sha256**: `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_specimen.jpg` (`be3da2bb8b817bec…`), `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_reference.json` (`61581ec4249a6ec3…`)
- `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_metrics.json` — artifact_sha256 `a59643aad0b7ce82…`, 3,955 B, 
- `docs/media/cornus_controversa_turing_wavelength_s0_comparison.png` — artifact_sha256 `18349142d83a1893…`, 336,856 B, 1260×840 px
- `docs/media/cornus_controversa_turing_wavelength_s0_growth.gif` — artifact_sha256 `797a672c01f13d30…`, 1,841,619 B, 480×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (PDE vs analytic sigma) | 1.0 1 | 0.98 | early-time Fourier growth of 12 discrete Laplacian modes vs eigenvalue σ(k) *(RMSE=0.00021, KS=0.5144, p_pearson=4.21e-30)* |
| dispersion RMSE | 0.0002 1/time | 0.0 | root mean square of σ_measured − σ_analytic over the probed band |
| dispersion KS statistic | 0.5144 1 | 0.0 | Kolmogorov–Smirnov statistic of σ residuals against Normal(0, s) *(ks_p=0.0031, n_modes=11)* |
| nonlinear dominant wavelength | 2.216 mm | 1.939 | radial FFT peak of the saturated Schnakenberg activator on an 8 λ patch |
| ablation equal-diffusion std ratio | 0.0183 1 | 1.0 | std(u_final)/std(u_initial) with D_u = D_v = D_u*, noise seed 3162 *(sigma_max=-0.0615 (must be negative))* |
| specimen foreground fraction | 0.8659 1 | 0.5 | border-contrast mask on cornus_controversa_specimen.jpg (crown photograph, not the 1.939 mm patch) |
| specimen structure-tensor coherency | 0.3634 1 | 0.4215 | mean structure-tensor coherency inside the foreground mask *(reference is the simulated orientation fraction, not a copy of the coherency)* |
| specimen horizontal-edge fraction | 0.4328 1 | 0.4215 | share of Sobel energy within 30 degrees of a horizontal edge on the crown photograph |
| specimen foliage bright fraction | 0.2118 1 | 0.25 | luminance > 0.45 above the lower 20 percent lawn band |

## Environment at checkpoint


Pass 1 working tree on top of 0c76d15d. New slot files are not on origin/main yet, by the zero-conflict rule. PASS2_REMEDIATION is False. Python 3.12, numpy, scipy, Pillow, matplotlib.
