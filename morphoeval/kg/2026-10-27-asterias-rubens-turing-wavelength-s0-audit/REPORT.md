# MorphoEval report — audit `2026-10-27-asterias-rubens-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Asterias rubens Turing wavelength at 115.2 µm (Holocene, Pass 1 square patch) (`models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`) | 90 | 74 | 88 | 84 | 91 | 86 | 88 | 89.3 | 77.8 | 88.5 | 84.4 | **84.4** | faithful |

## Findings and remediation

### Asterias rubens Turing wavelength at 115.2 µm (Holocene, Pass 1 square patch)  —  overall 84.4 (faithful)

Schnakenberg kinetics on a 576.0 µm Neumann patch (five catalog wavelengths of 115.2 µm). Diffusivity ratio 7.62 is just above onset, so the p=5 mode is unstable (σ≈0.0073) and the p=4 mode is stable (σ≈−0.051). Analytical versus FTCS growth: Pearson r = 0.999817, RMSE = 0.003533, KS = 0.0833. The 2D radial-FFT wavelength is 116.4904 µm. Equal diffusion is a negative control (max numerical σ = −0.103). The Hillewaert CC BY-SA studio photograph is segmented (circularity 0.164 versus a disk at 1; foreground 0.212 versus Otsu 0.198). The GIF ends on that photograph. The photograph has no scale bar and is not a 115.2 µm micrograph. Garm & Nilsson 2014 experimented on Linckia laevigata.

- **Model fidelity to source: 90** — The script integrates Schnakenberg kinetics and the Turing/Murray critical-wavenumber formula, and it places the ratio just above onset so the catalog mode is the unstable one. Rate constants are fitted to the catalog scale, not taken from an Asterias dose-response.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L98–145; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L147–163
- **Morphological realism / specimen match: 74** — The comparison figure places the Hillewaert animal beside the segmented outline and the activator field, and the GIF ends on the photograph, so the organism is recognisable. The generative field is still a square patch, and the photograph is not a 115.2 µm micrograph, so the tubercle texture is not a pixel fit to the wavelength.  
  evidence: `docs/media/asterias_rubens_turing_wavelength_s0_comparison.png` L0–0; `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Scaling both diffusivities by s moves the numerical wavelength as λ_c √s (sweep Pearson r = 1 on the sampled scales; λ(s=4)/λ(s=1) = 2). Setting D_u = D_v makes the maximum growth rate negative.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L245–274; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L215–243
- **Morphometric validation: 84** — Dispersion Pearson r = 0.999817, RMSE = 0.003533, KS = 0.0833 against the analytical curve. The 2D FFT wavelength is 116.4904 µm against 115.2 µm. Specimen circularity 0.164 is compared with a disk (1), and the fixed-threshold foreground 0.212 is compared with Otsu 0.198. The 0.02 arm-radius spacing reference is a transferred length, not a measurement on this plate.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L351–373; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L608–709; `datasets/asterias_rubens_turing_wavelength_s0/asterias_rubens_turing_wavelength_s0_reference.json` L72–87
- **Source and data traceability: 91** — Crossref-checked DOIs for Turing, Schnakenberg, Murray, and Garm & Nilsson. The specimen is the Hillewaert CC BY-SA file, SHA-256 pinned. The Linckia species limit is written in the reference and the bib note.  
  evidence: `datasets/asterias_rubens_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/asterias_rubens_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/asterias_rubens_turing_wavelength_s0/asterias_rubens_turing_wavelength_s0_reference.json` L55–69
- **Code-to-model traceability: 86** — The module docstring and the coefficient routine name the Schnakenberg terms and the Murray form of k_c. The Lean file proves the trace sign, the drive sign at r=7.62, the five-wavelength identity, and the tier inequality, with zero sorry.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3123_asterias_rubens_turing_wavelength_s0.lean` L18–38
- **Reproducibility package: 88** — Seed 3123, --self-test (Pearson r > 0.98, ablation σ < 0, p=5 unstable and p=4 stable, deterministic hash), and a documented plain invocation. Dependencies are the repo scientific stack. The full horizon is several tens of seconds because the near-onset growth rate is small.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L712–748; `datasets/asterias_rubens_turing_wavelength_s0/README.md` L1–30

Remediation:
- {'id': 'R1', 'item': 'Test the pentaradial body silhouette as the reaction domain on the 576 µm patch. If the arms are narrower than λ_c the mode should quench; record that control instead of forcing the mask into the production run.'}
- {'id': 'R2', 'item': 'Replace the transferred 0.02 arm-radius spacing reference with a within-plate null so the tubercle metric is not anchored to an assumed 5.8 mm arm.'}
- {'id': 'R3', 'item': 'No independent second assessor.'}
- {'id': 'R4', 'item': 'The studio photograph has no scale bar, so absolute tubercle spacing in micrometres cannot be read from it. Keep that limit explicit.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Asterias rubens Turing wavelength at 115.2 µm (Holocene, Pass 1 square patch)
- **spec_fingerprint** (semantic model + inputs): `f1205721de3cf1a837201a21f488df08c3a58b144285920a89c856f2c63ab2f0`
- **input_dataset_sha256**: `datasets/asterias_rubens_turing_wavelength_s0/asterias_rubens_specimen.jpg` (`7d7d25e638cab5b2…`), `datasets/asterias_rubens_turing_wavelength_s0/asterias_rubens_turing_wavelength_s0_reference.json` (`a5253b2ab52b4a62…`)
- `docs/media/asterias_rubens_turing_wavelength_s0_comparison.png` — artifact_sha256 `1e0766613517dd66…`, 367,344 B, 1632×984 px
- `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` — artifact_sha256 `b0b3ac2fa60389c9…`, 1,111,935 B, 840×666 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion pearson r | 0.999817 1 | 1.0 | Pearson r of analytical Turing σ(k) vs 1D FTCS growth at 12 wavenumbers around k_c *(reference is exact agreement; the value is the numerical correlation and is not identical to 1)* |
| dispersion RMSE | 0.003533 1/τ | 0.0 | RMSE between analytical and FTCS σ(k); KS statistic reported alongside *(KS statistic=0.0833 (two-sample KS of the σ curves))* |
| dispersion KS statistic | 0.083333 1 | 0.05 | scipy.stats.ks_2samp on analytical vs numerical σ(k) *(reference is a 0.05 screening tolerance, not the measured statistic)* |
| simulated stripe wavelength | 116.4904 µm | 115.2 | sub-bin radial FFT peak of the 2D Schnakenberg activator on the 576 µm patch *(catalog λ_c = 115.2 µm; sub-bin interpolation keeps the measured peak off the exact bin centre)* |
| wavelength ratio s=4 over s=1 | 2.0 1 | 2.0 | numerical σ-peak wavelength at diffusivity scale 4 divided by scale 1; law predicts 2 *(sweep Pearson r of λ vs λ_c√s = 1.0000)* |
| equal-diffusion control max sigma | -0.103318 1/τ | 0.0 | negative control: D_u = D_v, max numerical growth rate over the same k grid *(Turing drive is negative when r=1 because trace(J)<0; pattern must not grow)* |
| specimen contour circularity | 0.1637 1 | 1.0 | 4πA/P² on the segmented Hillewaert Asterias rubens studio photograph *(reference is a circular disk; the five-armed outline is the measured value)* |
| specimen image foreground fraction | 0.2122 1 | 0.1981 | fixed-threshold segmentation (gray>22) of the CC BY-SA specimen photograph *(reference is the Otsu foreground on the same plate, a different segmenter)* |
| specimen structure-tensor coherency | 0.3851 1 | 0.0 | mean (λ1−λ2)/(λ1+λ2) of the structure tensor inside the body mask *(reference is an isotropic null field; the photographed body is coherent along the arms)* |
| specimen tubercle spacing over arm radius | 0.0168 arm-radius | 0.01 | median nearest-neighbour spacing of pale-tubercle peaks, divided by the 98th-percentile body radius *(CV of nearest-neighbour distances = 0.7146; n_peaks=327. Reference is the within-plate 10th percentile of the same distances, not a transferred millimetre arm length. The studio plate has no scale bar.)* |
| silhouette-domain control std ratio | 0.020947 1 | 1.0 | negative control: std(u) with the pentaradial photograph mask scaled onto the 576 µm patch (t=48), divided by std(u) on the square production domain *(silhouette std=0.00143514, square std=0.0685123, quenched FFT wavelength=129.70 µm. A ratio near 0 means the organism outline is narrower than λ_c at this scale and must not be the reaction domain.)* |

## Environment at checkpoint


Pass 1 working tree. New slot-3123 files are not in this commit; build with --no-verify-loci.
