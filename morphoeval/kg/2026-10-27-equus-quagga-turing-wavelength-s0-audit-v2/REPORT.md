# MorphoEval report — audit `2026-10-27-equus-quagga-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-equus-quagga-turing-wavelength-s0-audit -> 2026-10-27-equus-quagga-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Equus quagga Holocene Turing wavelength at 1.9 mm (detail s0) (`models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py`) | 86 | 80 | 88 | 90 | 90 | 85 | 86 | 86.7 | 83.8 | 87.1 | 85.6 | **85.6** | faithful |

## Findings and remediation

### Equus quagga Holocene Turing wavelength at 1.9 mm (detail s0)  —  overall 85.6 (faithful)

Pass 2 keeps the Pass-1 Schnakenberg field (lambda(k_fast) pinned at 1.883 mm, age 0 Ma, Pearson r = 0.997, RMSE = 0.067 mm). The nominal FFT is 2.032 mm, quantization ratio 1.079 against the analytical fastest mode. A rump crop from the same Namutoni photograph has gradient -72.5 deg (stripes nearly horizontal where the pattern wraps the hindquarter) while the simulation stays at +2.5 deg. That mismatch is reported, not fitted away. An anisotropy probe gives k_measured/k_fast of 0.93, 1.00, and 0.73 at aniso 1.5, 3.2, and 8. Seeding the field at k_fast made the wavelength exact and dropped GIF-to-flank CV from 92.1 to 83, so the noise-seeded nominal render is unchanged. Equus scotti remains the equal-diffusion control.

- **Model fidelity to source: 86** — Unchanged Schnakenberg PDE, onset numerator, and gamma-scaled growth-rate peak. Pass 2 did not replace the kinetics with a curvature law; the rump failure shows that limit.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L110–176; `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L747–776
- **Morphological realism / specimen match: 80** — Flank GIF versus the Namutoni barrel remains a near-vertical stripe match (CV 92.1, gradients -2.5 and +2.5 deg), with broken dashes. The new rump crop is a counterexample: its stripes run nearly horizontally (-72.5 deg) and the same field does not follow them.  
  evidence: `docs/media/equus_quagga_turing_wavelength_s0_growth.gif` L0–0; `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_rump_crop.jpg` L0–0
- **Parameter expressivity: 88** — Gamma still shortens wavelength monotonically (r = 0.997). The anisotropy probe moves k_measured/k_fast from 0.93 at aniso 1.5 through 1.00 at 3.2 to 0.73 at 8, so the orientation parameter is not decorative. Equal diffusion still kills the pattern.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L408–426; `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 90** — Pearson r = 0.997, RMSE = 0.067 mm, KS D = 0.32. Quantization ratio 1.079 is stored against reference 1, so the 2.032 versus 1.883 mm gap is explicit. Rump angle -72.5 versus simulation 2.5 and rump period 19.9 versus 17.1 px are unequal specimen metrics. Flank period 18.3 versus 32.0 px likewise.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L713–745; `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L778–810
- **Source and data traceability: 90** — Same DOI set and SHA-pinned Namutoni photograph. The rump crop is taken from that same file with an explicit box.  
  evidence: `datasets/equus_quagga_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_turing_wavelength_s0_reference.json` L63–74
- **Code-to-model traceability: 85** — Pass-2 rump and anisotropy functions cite the remediation they answer. Lean still checks the onset sign, the scotti control, and the ten-stripe window, not the rump angle.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L690–694; `morphoeval/lean/loops/Loop_3221_equus_quagga_turing_wavelength_s0.lean` L12–33
- **Reproducibility package: 86** — Same seed, self-test, and command. Pass 2 adds two short integrations and a crop; the self-test still passes.  
  evidence: `models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py` L635–679; `datasets/equus_quagga_turing_wavelength_s0/README.md` L1–24

Remediation:
- {'id': 'R4', 'item': 'Still one assessor. An independent second score was not run.'}
- {'id': 'R5', 'item': 'Rump stripes are nearly horizontal and the straight-patch model does not follow them. A curvature-dependent diffusivity (Staddon 2024) on a body mesh would be the next mechanism, not a higher anisotropy on this rectangle.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Equus quagga Holocene Turing wavelength at 1.9 mm (detail s0)
- **spec_fingerprint** (semantic model + inputs): `c15e608fbebdc00810bdf416a1196b656433143dafaa4373228dabec7a265ec8`
- `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_rump_crop.jpg` — artifact_sha256 `af02fe3ba53ad43b…`, 11,895 B, 174×180 px
- `datasets/equus_quagga_turing_wavelength_s0/equus_quagga_turing_wavelength_s0_metrics.json` — artifact_sha256 `30dee58f693542cb…`, 3,000 B, 
- `docs/media/equus_quagga_turing_wavelength_s0_comparison.png` — artifact_sha256 `63bb5de23ed6cd3e…`, 553,458 B, 1512×888 px
- `docs/media/equus_quagga_turing_wavelength_s0_growth.gif` — artifact_sha256 `70b8320f69bb5b0f…`, 3,625,635 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Pearson r of analytical vs PDE stripe wavelength across gamma | 0.99747 Pearson r | 1.0 | pearsonr(lambda_c(gamma), FFT wavelength of Schnakenberg u) on a fixed 15-wave window *(RMSE 0.0666 mm; p=9.61e-06. Reference 1 is perfect tracking, not the measured r.)* |
| RMSE of PDE wavelength against analytical lambda_c | 0.06664 mm | 0.0 | sqrt mean square residual of the gamma sweep *(reference 0 would be a perfect match; the residual is the discretization error)* |
| nominal measured stripe wavelength | 2.0316 mm | 1.883 | y-averaged FFT of activator at gamma=1.15, scaled by the 1.883 mm pin *(analytical and measured differ by mode quantization on the periodic window)* |
| specimen flank gradient angle | 2.5 deg | -2.5 | Sobel orientation histogram on the rendered GIF field, compared with the Namutoni crop mask *(value is the simulation, reference is the specimen; they are not set equal)* |
| specimen flank stripe period | 32.042 px | 18.284 | autocorrelation peak of the luminance projection inside the comparison frame *(adult photograph period in pixels, not the 1.883 mm fetal wavelength)* |
| luminance KS statistic specimen vs simulation | 0.3203 KS D | 0.0 | ks_2samp on flank-mask luminance vs rendered stripe luminance *(p=0; D=0 would be identical histograms)* |
| equal-diffusion ablation amplitude ratio (quagga / scotti) | 10000.0 amplitude ratio | 1.0 | max(u)-min(u) at d_ratio=18, aniso=3.2 versus d_ratio=1 (Equus scotti control) *(predecessor chronospecies has no selected 1.883 mm mode)* |
| specimen structure-tensor coherency | 0.7075 coherency | 0.8873 | mean structure-tensor coherency of the rendered field *(simulation value versus specimen reference)* |
| periodic-mode quantization ratio (measured / analytical wavelength) | 1.0789 ratio | 1.0 | nominal FFT wavelength divided by lambda(k_fast) *(ratio 1 would be an exact integer mode; the residual is the nonlinear shift)* |
| rump crop gradient angle | 2.5 deg | -72.5 | Sobel orientation on the rendered field versus the Namutoni rump crop *(second body region; value is the simulation, reference is the rump)* |
| rump crop stripe period | 17.123 px | 19.947 | projection period on the rump crop versus the same rendered field *(not set equal to the reference)* |

## Environment at checkpoint


Pass 2 working tree. Slot-3221 files are new relative to the checkpoint commit.
