# MorphoEval report — audit `2026-10-27-danio-albolineatus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio albolineatus Schnakenberg wavelength at 37 µm (Holocene, 0 Ma) (`models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py`) | 84 | 76 | 84 | 74 | 88 | 78 | 82 | 84.0 | 75.2 | 82.9 | 80.2 | **80.2** | faithful |

## Findings and remediation

### Danio albolineatus Schnakenberg wavelength at 37 µm (Holocene, 0 Ma)  —  overall 80.2 (faithful)

Pass 1. Schnakenberg reaction-diffusion on a periodic 96×60 patch. Activator diffusivity is solved so the pearl-danio fastest mode is 3.686e-05 m (36.9 µm). At equal time the predecessor (anisotropic inhibitor, Danio aesculapii) forms vertical bars at 41.8 µm (std 0.327), the focal species stays near the homogeneous state (std 0.007), and the successor (Danio margaritatus) forms spots at 34.2 µm (std 0.707). Equal diffusion is stable (σ_max = −0.137). Closed-form dispersion matches matrix eigenvalues at Pearson r = 1.000, RMSE ~ 1e-15. The radial-spectrum correlation against linear theory is 0.9799. One Commons photograph; flank depth/length 0.221 versus the FishBase landmark ratio 0.250. Weighted overall 80.2/100.

- **Model fidelity to source: 84** — The steady state, Jacobian, dispersion root, critical inhibitor ratio, and equal-diffusion control are implemented and checked against the matrix eigenvalue (r = 1, RMSE ~ 1e-15). Five Lean theorems with zero sorry record the steady state, the sign of tr J and det J, the inhibitor ratio, the chronoseries order, and the k·λ product. The kinetics are generic Schnakenberg, not a transcription of Nakamasu's cell-interaction network or Volkening's three-cell iridophore model.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L63–114; `morphoeval/lean/loops/Loop_3161_danio_albolineatus_turing_wavelength_s0.lean` L14–41; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L1–16
- **Morphological realism / specimen match: 76** — The comparison figure places the Commons photograph beside the three simulated regimes on a shared colour scale, and the regimes are visually distinct (bars, near-blank, spots, decayed control). The growth GIF keeps the photograph and shows the cellular field in an inset because a 37 µm mode is sub-pixel on the animal. Computer-vision similarity of that GIF to the photograph is 96.0 (12 frames, 1024×680). A full-frame ablation of the cellular field alone scores 85.2. Deduction: the high GIF score is carried by the unchanged background, and the pearl-danio panel on the shared scale is almost featureless, which is the biological claim but a weak visual match to any stripe the eye expects.  
  evidence: `docs/media/danio_albolineatus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/danio_albolineatus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L395–431
- **Parameter expressivity: 84** — The inhibitor-ratio sweep at fixed Du moves σ_max from −0.040 (δ/δ_c = 0.90) through onset to 0.319 (δ/δ_c = 3.20), and λ lengthens monotonically from 34.4 µm to 45.3 µm. The chronoseries changes one geometric factor at a time: AP inhibitor anisotropy selects bars, a shorter activator range plus a higher ratio selects spots, equal diffusion decays. No command-line override for a, b, or α beyond editing the constants.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L141–204; `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 74** — The segmented flank has depth/length 0.221 against the FishBase landmark ratio 0.250, circularity 0.321 against 1, and structure-tensor coherency 0.360. A tighter ROI gives 0.226, so the ratio is not an artefact of one bounding box. Dispersion Pearson r = 1.000 and RMSE ~ 1e-15. The radial spectrum versus linear theory is r = 0.9799, just under 0.98, and has not yet been split into the unstable band. Kolmogorov-Smirnov distances of the final activator against a Gaussian are 0.015 (pearl danio), 0.072 (bars), 0.153 (spots). Wavelength has been measured at one noise seed.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L176–184; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L268–322; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L324–379
- **Source and data traceability: 88** — DOIs were resolved on OpenAlex on 2026-10-27. The specimen is Wikimedia Commons File:Danio albolineatus.jpg, public domain, Steven Coburn, SHA-256 f42000111fe647d8… pinned in the reference JSON and checked at runtime. The FishBase depth ratio is cited to the repo's fish-shapes table. Deduction: sister-species wavelengths are not tied to a measured histological table.  
  evidence: `datasets/danio_albolineatus_turing_wavelength_s0/paper_citations.bib` L1–84; `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L530–540
- **Code-to-model traceability: 78** — Each function corresponds to a named piece of the dispersion relation, and the Lean file cites the same integer witnesses the self-test recomputes (nanometre order within 80 nm). There is no map from code lines onto numbered equations of Turing 1952 or Nakamasu 2009.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L1–16; `morphoeval/math_kg_loops/delphi10000-danio-albolineatus-turing-wavelength-s0.yaml` L1–69
- **Reproducibility package: 82** — Seed 3161, dt = 0.007 under the explicit-Euler stability bound, --self-test passes, and the full render writes the PNG, the GIF, and the metrics JSON. The specimen hash is checked before the integration. Dependencies are the repo scientific stack, not a per-script lockfile, and the checkpoint commit does not yet contain these files.  
  evidence: `models/2026-delphi10000-danio-albolineatus-turing-wavelength-s0.py` L806–845; `datasets/danio_albolineatus_turing_wavelength_s0/README.md` L1–50

Remediation:
- {'id': 'R1', 'item': 'All-band spectral Pearson r is 0.9799. Recompute the correlation on the unstable band σ(k) > 0 and report that number even if it is worse.'}
- {'id': 'R2', 'item': 'FFT wavelength is reported for one noise seed. Repeat seeds 7 and 99 and record the coefficient of variation.'}
- {'id': 'R3', 'item': 'One aquarium photograph. The depth/length ratio is a bounding box, not the FishBase landmark measurement.'}
- {'id': 'R4', 'item': 'Single assessor. This pass is not a cross-provider audit.'}
- {'id': 'R5', 'item': 'Predecessor and successor wavelengths are reconstructions from the diffusion change, not measurements at 37 µm.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio albolineatus Schnakenberg wavelength at 37 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `4c792f240f2eddeb1b293e2ff8dcb16085a188074fadafd2384021469481a83a`
- **input_dataset_sha256**: `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_reference.json` (`fb3bd6f8d0bd368c…`), `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_specimen.jpg` (`f42000111fe647d8…`), `datasets/danio_albolineatus_turing_wavelength_s0/paper_citations.bib` (`80659eeead83362e…`), `datasets/danio_albolineatus_turing_wavelength_s0/scout_manifest.json` (`2c0d2a9d801e9099…`)
- `datasets/danio_albolineatus_turing_wavelength_s0/danio_albolineatus_turing_wavelength_s0_metrics.json` — artifact_sha256 `7f10cbaaab98a96d…`, 3,534 B, 
- `docs/media/danio_albolineatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `056fbfdf9ee2dc34…`, 433,658 B, 1584×1224 px
- `docs/media/danio_albolineatus_turing_wavelength_s0_growth.gif` — artifact_sha256 `0d3134fed5cf6d06…`, 4,349,071 B, 1024×680 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen flank depth over length | 0.22127659574468084 bbox height / bbox width | 0.25003 | largest blue-silver connected component on the Commons photograph *(Photograph bbox includes the caudal fin and is not the landmark body-depth ratio, so the two numbers differ.)* |
| specimen contour circularity | 0.32120558575183245 4πA/P^2 | 1.0 | morphological perimeter of the segmented flank *(Non-circular specimen metric. Value is the fish outline, reference is the circle.)* |
| specimen flank structure-tensor coherency | 0.35986021060553475 (λ1-λ2)/(λ1+λ2) | 0.0 | Sobel structure tensor inside the fish mask, Gaussian σ=2 |
| simulated stripe critical wavelength | 3.6858408006837576e-05 m | 3.686e-05 | argmax_k σ(k) of the Schnakenberg dispersion at the solved Du, Dv *(Live fastest-mode wavelength. It is solved to sit on the assigned scale; the stored value is the solver output and is compared, not copied, in the self-test.)* |
| predecessor bar wavelength | 4.1798633640911064e-05 m | 3.6858408006837576e-05 | fastest mode of the AP inhibitor-anisotropic tensor (alpha=1.8) *(Longer than the uniform-interstripe mode.)* |
| successor spot wavelength | 3.421679435816217e-05 m | 3.6858408006837576e-05 | fastest mode at beta=1.55 activator shrink and delta=2.40 delta_c *(Shorter than the uniform-interstripe mode.)* |
| dispersion goodness-of-fit pearson r | 0.9999999999999999 Pearson r | 1.0 | closed-form σ(k) versus numpy eigenvalue σ(k) at 48 wavenumbers *(RMSE is reported alongside. r is computed, not assigned.)* |
| dispersion RMSE | 6.945944863625148e-16 growth rate | 0.0 | root mean square of analytic σ minus eigenvalue σ |
| linear-spectrum pearson r | 0.9798906225916986 Pearson r | 1.0 | log radial FFT of the integrated pearl-danio field versus exp(2 σ(k) T), all bins |
| unstable-band stripe spectrum pearson r | 0.9359556315507189 Pearson r | 1.0 | same radial FFT correlation restricted to bins with σ(k) > 0 *(Restricting to σ(k)>0 lowers r to about 0.936: those bins are few and the weakly grown mode is still close to the noise floor. The 0.98 goodness-of-fit is the closed-form versus eigenvalue dispersion correlation, not this band.)* |
| seed-ensemble wavelength CV | 0.02684822672882177 std/mean of FFT wavelength | 0.0 | dominant FFT wavelength of the pearl-danio field at seeds 3161, 7 and 99 *(wavelengths nm=[35759, 34513, 36860])* |
| albolineatus activator KS versus Gaussian | 0.015217039709082836 KS distance | 0.09934528757039407 | scipy.stats.kstest of standardised final u against N(0,1) *(Coefficient of variation of the pattern is std(u)/mean(u), stored as std_u.)* |
| pattern amplitude CV albolineatus | 0.006620850155858999 std(u)/u0 | 0.32688748377159915 | coefficient of variation of the activator at T=32 |
| equal-diffusion control max growth rate | -0.1370382928379291 1/time | 0.0 | max_k σ(k) at Du=Dv *(Negative control. σ_max < 0 means the homogeneous state is stable.)* |
| tight-ROI specimen depth over length | 0.22560975609756098 bbox height / bbox width | 0.22127659574468084 | fish mask eroded by 15% of the bbox on each side *(Sensitivity check. The two ROIs are not the same number.)* |

## Environment at checkpoint


Pass 1 on the loop branch. New files are not in 0c76d15d; that commit is the base the branch started from. Python 3.12, numpy, scipy, Pillow, matplotlib.
