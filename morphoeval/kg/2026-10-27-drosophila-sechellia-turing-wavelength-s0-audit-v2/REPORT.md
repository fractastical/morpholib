# MorphoEval report — audit `2026-10-27-drosophila-sechellia-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `169742c66d5febd4331e8b1fa7dc67037f33d175` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-sechellia-turing-wavelength-s0-audit -> 2026-10-27-drosophila-sechellia-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila sechellia Turing wavelength at 1.5 µm (Holocene, detail scale s0) (`models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py`) | 90 | 86 | 88 | 88 | 90 | 86 | 87 | 89.3 | 86.8 | 87.7 | 87.8 | **87.8** | faithful |

## Findings and remediation

### Drosophila sechellia Turing wavelength at 1.5 µm (Holocene, detail scale s0)  —  overall 87.8 (faithful)

Pass 2 keeps the Turing 1952 law at λ_c = 1.500 µm, age 0 Ma. The critical-shell power-weighted wavelength is 1.5004 µm (0.03% from 1.500). The brightest oblique Fourier bin remains 1.480 µm and is reported separately. Panel D now shows the unstable lobe (σ_max ≈ +0.004) for santomea, serrata, and sechellia. Panel F places the equal-diffusion note (σ_max = −0.176) above the bars. Sweep Pearson r = 0.9996. Predecessor λ_c = 1.921 µm (D. santomea); successor λ_c = 1.680 µm (D. serrata).

- **Model fidelity to source: 90** — Same Turing PDE, k_c formula, periodic boundaries, cubic saturation, and equal-diffusion ablation as Pass 1. Pass 2 adds the critical-shell estimator without changing the Jacobian. g_u is still co-adapted to hold h_min, not copied from a measured rate.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L97–129; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L205–235; `morphoeval/lean/loops/Loop_3202_drosophila_sechellia_turing_wavelength_s0.lean` L20–34
- **Morphological realism / specimen match: 86** — Panel A is the CC-BY D. sechellia ovary micrograph; panel C modulates it by the field; the 12-frame 720×540 GIF keeps a moving activator and a last frame whose computer-vision similarity to that micrograph is high. Panel F no longer covers the species names. The micrograph is still an ovary, not the ocellar lattice, so the score stays short of a matched ultrastructure.  
  evidence: `docs/media/drosophila_sechellia_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_sechellia_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L718–730
- **Parameter expressivity: 88** — f_u still shortens λ monotonically (r = 0.9996) with g_u co-adapted to fixed onset depth. Panel D now shows three distinct positive lobes, with sechellia peaking nearest the dashed k_c line and santomea peaking at lower k.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L267–302; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L681–697
- **Morphometric validation: 88** — Critical-shell wavelength is 1.5004 µm against analytical 1.500 µm (relative error 0.03%). The brightest bin, 1.480 µm, is kept as its own measurement so the oblique mode is not hidden. Pearson r = 0.9996, calibrated RMSE = 0.0040 µm, four-seed shell CV = 0.0037, KS = 0.229. Specimen Otsu fraction 0.352, aspect 1.331, and coherency 0.612 differ from their priors.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L441–500; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L760–774
- **Source and data traceability: 90** — Unchanged from Pass 1. DOIs were checked on Crossref or OpenAlex. The ovary JPEG is CC BY 2.0 and SHA-256 pinned. Scale 1.5 µm and age 0 Ma are in the reference JSON.  
  evidence: `datasets/drosophila_sechellia_turing_wavelength_s0/paper_citations.bib` L1–78; `datasets/drosophila_sechellia_turing_wavelength_s0/drosophila_sechellia_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Pass 2 comments on dominant_wavelength state why the brightest bin and the critical shell differ, and they point at the rectangular-grid tie. The Jacobian block still names the chronospecies papers.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L205–214; `morphoeval/math_kg_loops/delphi10000-drosophila-sechellia-turing-wavelength-s0.yaml` L24–68
- **Reproducibility package: 87** — --self-test passes on the fast grid, including Pearson r > 0.98, the equal-diffusion control, and the 0.5% shell tolerance. The full render writes analysis_cache.json schema 2, which measure_morphoeval reads. Seed 3202 is fixed.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L786–806; `datasets/drosophila_sechellia_turing_wavelength_s0/README.md` L36–48

Remediation:
- {'id': 'R4', 'item': 'A second assessor from another provider has not scored this audit. This pass does not invent one.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila sechellia Turing wavelength at 1.5 µm (Holocene, detail scale s0)
- **spec_fingerprint** (semantic model + inputs): `3e976ad024ca3ede382f29ba68b3901386a7d687255a0ab472b27fe393935203`
- **input_dataset_sha256**: `datasets/drosophila_sechellia_turing_wavelength_s0/drosophila_sechellia_ovary_specimen.jpg` (`5a765a91794b001a…`), `datasets/drosophila_sechellia_turing_wavelength_s0/drosophila_sechellia_turing_wavelength_s0_reference.json` (`4dda020e70850732…`), `datasets/drosophila_sechellia_turing_wavelength_s0/paper_citations.bib` (`e61c4c157d3f887c…`), `datasets/drosophila_sechellia_turing_wavelength_s0/scout_manifest.json` (`20e1cd87901f1516…`)
- `docs/media/drosophila_sechellia_turing_wavelength_s0_comparison.png` — artifact_sha256 `2182b4df9c2ef412…`, 471,410 B, 1512×978 px
- `docs/media/drosophila_sechellia_turing_wavelength_s0_growth.gif` — artifact_sha256 `43225dea5b2589fa…`, 3,639,981 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength lambda_c | 1.5 µm | 1.548 | lambda_c = 2π / k_c with k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) *(trace=-5.6000, det=0.9450, hmin=-0.0350)* |
| brightest Fourier bin wavelength | 1.479591 µm | 1.5 | argmax bin of the 2D Fourier power; an oblique neighbour of the critical ring on the rectangular grid *(relative error of the brightest bin=0.01361)* |
| critical-shell power-weighted wavelength | 1.500439 µm | 1.548 | power-weighted |k| inside |k − k_c| < 0.55 rad/µm, converted by λ = 2π/|k| *(shell k=4.1876; broadband weighted λ=1.5004 µm)* |
| activator-gain sweep Pearson r | 0.999647 correlation | 1.0 | Pearson r between analytical λ_c(f_u) and power-weighted wavelength over 7 co-adapted Jacobians *(RMSE after affine calibration=0.00399 µm; monotonic=True)* |
| activator-gain sweep RMSE | 0.003991 µm | 0.0 | RMSE of calibrated power-weighted wavelength against analytical λ_c(f_u) *(calibration removes the stable-side-mode bias of the spectral mean; r is uncalibrated)* |
| radial-spectrum KS distance vs analytical ring | 0.22864 KS | 0.05 | Kolmogorov-Smirnov distance between the 24-bin radial power of u and a Gaussian ring at k_c *(lower is closer; the packet has a finite unstable bandwidth)* |
| multi-seed stripe wavelength CV | 0.003662 CV | 0.15 | sample CV of the Fourier peak wavelength across four RNG seeds *(peaks=[1.502939332385388, 1.5067426724493405, 1.494423521532098, 1.5052395268828347])* |
| equal-diffusion negative control sigma_max | -0.175962 1/time | 0.0 | largest eigenvalue of J − k² D after setting D_v = D_u *(negative σ_max means the stripe mode is damped when diffusivities are equal)* |
| santomea predecessor lambda_c | 1.920553 µm | 1.5000000000000004 | same (D_u, D_v, f_v, g_v, h_min); f_u=1.327 and co-adapted g_u (tan-loss) *(σ_max=0.0045; longer than sechellia)* |
| serrata successor lambda_c | 1.680054 µm | 1.92055319899344 | same diffusivities and onset depth; f_u=1.458 and co-adapted g_u *(σ_max=0.0042; between santomea and sechellia)* |
| specimen segmented foreground fraction | 0.352072 fraction | 0.5 | Otsu threshold on the CC-BY D. sechellia ovary micrograph, resized to 480×360 *(threshold=0.182; not a self-copy of the prior 0.50)* |
| specimen segmented aspect ratio | 1.331307 width/height | 1.0 | bounding box of the Otsu foreground of the ovary micrograph *(bbox 438×329 px)* |
| specimen structure-tensor coherency | 0.611938 dimensionless | 0.25 | (λ1−λ2)/(λ1+λ2) averaged on the Otsu foreground after a radius-2 box smooth *(circularity=0.0678)* |
| specimen stripe-segmentation circularity | 0.067839 4πA/P² | 1.0 | isoperimetric quotient of the Otsu mask of the real micrograph *(used as a non-circular specimen morphometric, not as a Turing wavelength)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 commit 169742c6. Supersedes Pass 1 at 3a268749. Line numbers refer to 169742c6.
