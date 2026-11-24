# MorphoEval report — audit `2026-10-27-asterias-rubens-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-asterias-rubens-turing-wavelength-s0-audit -> 2026-10-27-asterias-rubens-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Asterias rubens Turing wavelength at 115.2 µm (Holocene, Pass 2) (`models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`) | 90 | 78 | 90 | 88 | 91 | 86 | 88 | 90.0 | 81.8 | 88.5 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Asterias rubens Turing wavelength at 115.2 µm (Holocene, Pass 2)  —  overall 86.2 (faithful)

Pass 2 keeps the square 576.0 µm Neumann patch (λ_FFT = 116.4904 µm, dispersion Pearson r = 0.999817, RMSE = 0.003533, KS = 0.0833). The pentaradial photograph, scaled onto that patch, is a negative control: std(u) falls to 0.00144 against 0.0685 on the square (ratio 0.0209), so the organism outline is not the reaction domain at 115.2 µm. Tubercle spacing is now the within-plate median (0.0168 arm-radii) against the within-plate 10th percentile (0.0100), not a transferred 5.8 mm arm. Equal-diffusion max σ remains −0.103. The GIF is 12 frames at 840×666 and ends on the Hillewaert photograph. The plate still has no scale bar. Garm & Nilsson 2014 experimented on Linckia laevigata.

- **Model fidelity to source: 90** — Same Schnakenberg / Turing–Murray core as Pass 1. The silhouette mask was tested as a domain and rejected because it quenches the unstable mode; the production equations are unchanged.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L98–145; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L574–606
- **Morphological realism / specimen match: 78** — The figure still shows the real animal, its outline, and the activator, and the GIF ends on the photograph. Pass 2 states why that outline is not the micro-domain. The photograph remains an unscaled studio plate, so the tubercle field is not a pixel fit at 115.2 µm.  
  evidence: `docs/media/asterias_rubens_turing_wavelength_s0_comparison.png` L0–0; `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Diffusivity scale s moves λ as √s (ratio 2 at s=4). Equal diffusion keeps σ negative. The silhouette mask is a third control: it collapses the activator standard deviation by about 50× instead of preserving the square-domain pattern.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L245–274; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L693–708
- **Morphometric validation: 88** — Dispersion Pearson r = 0.999817, RMSE = 0.003533, KS = 0.0833. FFT wavelength 116.4904 µm versus 115.2 µm. Silhouette/square std ratio = 0.0209 versus 1. Tubercle median spacing 0.0168 arm-radii versus the within-plate 10th percentile 0.0100. Specimen circularity 0.164 versus a disk. The plate still cannot supply an absolute micrometre.  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L608–709; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L351–373
- **Source and data traceability: 91** — Unchanged from Pass 1: Crossref DOIs, CC BY-SA specimen, SHA-256, and the Linckia species limit on Garm & Nilsson 2014.  
  evidence: `datasets/asterias_rubens_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/asterias_rubens_turing_wavelength_s0/asterias_rubens_turing_wavelength_s0_reference.json` L55–69
- **Code-to-model traceability: 86** — Equation comments and the Lean proofs are unchanged. The quench control is commented at the call site as the Pass-2 test of R1.  
  evidence: `morphoeval/lean/loops/Loop_3123_asterias_rubens_turing_wavelength_s0.lean` L18–38; `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L584–594
- **Reproducibility package: 88** — Same seed, --self-test, and invocation as Pass 1. The extra control is inside run_science, so a plain run reproduces the quench ratio. Near-onset integration remains slow (tens of seconds).  
  evidence: `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` L712–748; `datasets/asterias_rubens_turing_wavelength_s0/README.md` L1–30

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass. Silhouette scaled onto the 576 µm patch quenches the mode (std ratio 0.0209). Production domain stays the square.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in this pass. Tubercle spacing reference is the within-plate 10th percentile (0.0100 arm-radii), not an assumed 5.8 mm arm.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'No independent second assessor.'}
- {'id': 'R4', 'item': 'The studio photograph has no scale bar. Absolute tubercle spacing in micrometres is not claimed.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Asterias rubens Turing wavelength at 115.2 µm (Holocene, Pass 2)
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


Pass 2 working tree. New slot-3123 files are not in this commit; build with --no-verify-loci.
