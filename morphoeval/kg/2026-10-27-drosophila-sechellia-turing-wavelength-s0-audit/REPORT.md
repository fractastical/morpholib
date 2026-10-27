# MorphoEval report — audit `2026-10-27-drosophila-sechellia-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `3a2687497deaf1d3dae381dbe02b0ce57adcd532` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila sechellia Turing wavelength at 1.5 µm (Holocene, detail scale s0) (`models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py`) | 86 | 80 | 84 | 85 | 90 | 82 | 84 | 85.3 | 81.9 | 85.5 | 84.0 | **84.0** | faithful |

## Findings and remediation

### Drosophila sechellia Turing wavelength at 1.5 µm (Holocene, detail scale s0)  —  overall 84.0 (faithful)

Pass 1 implements the Turing 1952 critical-wavenumber law on an 18 µm × 13.5 µm periodic patch. The sechellia Jacobian is pinned at λ_c = 1.500 µm with trace < 0, det > 0 and h_min = −0.035. Predecessor D. santomea (f_u = 1.327) selects 1.921 µm; successor D. serrata (f_u = 1.458) selects 1.680 µm. The activator-gain sweep has Pearson r = 0.9996. The equal-diffusion control has σ_max = −0.176. The render-grid Fourier peak is 1.480 µm (1.3% short of 1.500). The comparison figure's panel F annotation collides with the tick labels, and the dispersion panel's y-range hides the σ ≈ 0.004 unstable lobe. Age 0 Ma and scale 1.5 µm are stated in the reference record and the figure title.

- **Model fidelity to source: 86** — The script implements the Turing PDE, the classical k_c formula, periodic boundaries, cubic saturation, and a true equal-diffusion ablation. g_u is co-adapted to hold h_min fixed rather than taken from a measured rate, so this is a reconstruction of the dispersion law at the assigned scale, not a transcription of a numbered table from McGregor 2007.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L97–129; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L161–202; `morphoeval/lean/loops/Loop_3202_drosophila_sechellia_turing_wavelength_s0.lean` L26–34
- **Morphological realism / specimen match: 80** — Panel A is the real CC-BY D. sechellia ovary micrograph and panel C modulates it by the field. The growth GIF is 12 frames at 720×540 and its last frame scores CV 96 against that micrograph. Panel F's equal-diffusion annotation is drawn on top of the species names, and the micrograph is ovarian nuclei rather than the ocellar lattice the wavelength refers to.  
  evidence: `docs/media/drosophila_sechellia_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_sechellia_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L698–705
- **Parameter expressivity: 84** — Raising f_u, with g_u co-adapted to the same onset depth, shortens λ monotonically across seven values (r = 0.9996). The dispersion panel plots all three chronospecies but the positive lobe (σ_max ≈ 0.004) is lost on an axis that runs down to −0.5.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L256–291; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L648–666
- **Morphometric validation: 85** — Analytical λ_c is 1.500 µm. The render-grid Fourier peak is 1.480 µm (relative error 1.3%). Sweep RMSE after the stated affine calibration is 0.0040 µm. Four-seed CV is 0.006. Otsu foreground fraction 0.352, aspect and coherency are compared with priors they do not equal. KS distance to a narrow Gaussian ring is 0.229, which is reported rather than forced to zero.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L430–555; `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L306–371
- **Source and data traceability: 90** — Crossref- or OpenAlex-checked DOIs for Turing, McGregor, Sucena, Jeong, Chenoweth & Blows, and Shoup. The ovary JPEG is CC BY 2.0 and SHA-256 pinned. The transfer limits are written in the reference JSON.  
  evidence: `datasets/drosophila_sechellia_turing_wavelength_s0/paper_citations.bib` L1–78; `datasets/drosophila_sechellia_turing_wavelength_s0/drosophila_sechellia_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — The module docstring and critical_wavenumber cite the Turing formula. Chronospecies comments name Jeong, McGregor, and Chenoweth. The Fourier discretisation is not tied to a numbered equation in a methods paper beyond Turing 1952.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L1–40; `morphoeval/math_kg_loops/delphi10000-drosophila-sechellia-turing-wavelength-s0.yaml` L24–68
- **Reproducibility package: 84** — --self-test checks the Turing inequalities, the wavelength order, the ablation, monotonicity, and Pearson r > 0.98. Seed 3202 is fixed. The render-grid peak (1.480 µm) and the self-test grid peak (1.500 µm) are not the same number, so a reader can miss which grid the figure used.  
  evidence: `models/2026-delphi10000-drosophila-sechellia-turing-wavelength-s0.py` L734–776; `datasets/drosophila_sechellia_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'Move the equal-diffusion annotation in render_png off the panel F tick labels.'}
- {'id': 'R2', 'item': 'Show the unstable lobe of σ(k). The near-onset peak is about +0.004 and is invisible on the current y-range.'}
- {'id': 'R3', 'item': 'Bring the render-grid Fourier peak from 1.480 µm to within 0.5% of the analytical 1.500 µm.'}
- {'id': 'R4', 'item': 'A second assessor from another provider has not scored this audit. Do not invent one.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila sechellia Turing wavelength at 1.5 µm (Holocene, detail scale s0)
- **spec_fingerprint** (semantic model + inputs): `3e976ad024ca3ede382f29ba68b3901386a7d687255a0ab472b27fe393935203`
- **input_dataset_sha256**: `datasets/drosophila_sechellia_turing_wavelength_s0/drosophila_sechellia_ovary_specimen.jpg` (`5a765a91794b001a…`), `datasets/drosophila_sechellia_turing_wavelength_s0/drosophila_sechellia_turing_wavelength_s0_reference.json` (`4dda020e70850732…`), `datasets/drosophila_sechellia_turing_wavelength_s0/paper_citations.bib` (`e61c4c157d3f887c…`), `datasets/drosophila_sechellia_turing_wavelength_s0/scout_manifest.json` (`20e1cd87901f1516…`)
- `docs/media/drosophila_sechellia_turing_wavelength_s0_comparison.png` — artifact_sha256 `6ce4f8039da18265…`, 477,513 B, 1512×978 px
- `docs/media/drosophila_sechellia_turing_wavelength_s0_growth.gif` — artifact_sha256 `43225dea5b2589fa…`, 3,639,981 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength lambda_c | 1.5 µm | 1.548 | lambda_c = 2π / k_c with k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) *(trace=-5.6000, det=0.9450, hmin=-0.0350)* |
| simulated stripe peak wavelength | 1.479591 µm | 1.5 | argmax of the 2D Fourier power of the final activator, periodic 18 µm × 13.5 µm patch *(relative error=0.01361; weighted λ=1.5004 µm)* |
| activator-gain sweep Pearson r | 0.999647 correlation | 1.0 | Pearson r between analytical λ_c(f_u) and power-weighted wavelength over 7 co-adapted Jacobians *(RMSE after affine calibration=0.00399 µm; monotonic=True)* |
| activator-gain sweep RMSE | 0.003991 µm | 0.0 | RMSE of calibrated power-weighted wavelength against analytical λ_c(f_u) *(calibration removes the stable-side-mode bias of the spectral mean; r is uncalibrated)* |
| radial-spectrum KS distance vs analytical ring | 0.22864 KS | 0.05 | Kolmogorov-Smirnov distance between the 24-bin radial power of u and a Gaussian ring at k_c *(lower is closer; the packet has a finite unstable bandwidth)* |
| multi-seed stripe wavelength CV | 0.006164 CV | 0.15 | sample CV of the Fourier peak wavelength across four RNG seeds *(peaks=[1.4795908857482158, 1.5, 1.4976905298081187, 1.4948186373673196])* |
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

Pass 1 working tree committed at 3a268749. Self-test and full render both completed. Line numbers refer to that commit.
