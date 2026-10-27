# MorphoEval report — audit `2026-10-27-aulacopleura-koninckii-meraspis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-aulacopleura-koninckii-meraspis-turing-wavelength-s0-audit -> 2026-10-27-aulacopleura-koninckii-meraspis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg critical wavelength at 3.125 µm for Aulacopleura koninckii meraspis (Pass 2) (`models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py`) | 90 | 84 | 88 | 90 | 90 | 86 | 88 | 89.3 | 86.2 | 88.1 | 87.7 | **87.7** | faithful |

## Findings and remediation

### Schnakenberg critical wavelength at 3.125 µm for Aulacopleura koninckii meraspis (Pass 2)  —  overall 87.7 (faithful)

Pass 2 keeps the Pass-1 law (λ_c = 3.125 µm, r = 0.99937, equal-diffusion σ_max = -0.458 /s, sweep 3.146–3.993 µm) and adds a 48×36 pleural patch whose dominant wavelength is 3.125 µm (k_dom/k_c = 1.000), distinct from the 1D fastest mode at 3.146 µm. The comparison figure now draws the specimen contour, prints KS = 0.067, and states the holaspis18 copy ratio 1.50 at fixed k_c. GIF is 720×560 and 16 frames; last-frame CV similarity to the Commons photograph is 95.7. Counted furrows remain 11 against the assigned 12.

- **Model fidelity to source: 90** — Pass 1's Schnakenberg PDE and k_c formula remain, with r=0.99937 on the discrete dispersion band. Pass 2 adds the 2D patch: mediolateral noise does not shift the dominant anteroposterior wavenumber off k_c (ratio 1.000). Parameters are still a reconstruction that hits the assigned wavelength, not a measured trilobite kinetic set.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L98–127; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L180–235; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L269–302
- **Morphological realism / specimen match: 84** — The comparison figure now overlays the border-contrast contour on the Commons photograph, and the GIF (16 frames, 720×560) still ends on that photograph (CV 95.7). The 2D field is a short-wavelength lira texture in the same ochre family, not a pixel copy of the exoskeleton. Furrow count 11 is shown against the assigned 12 rather than edited to match.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L640–671; `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — The diffusion-ratio sweep still moves λ_max from 3.146 µm to 3.993 µm, and equal diffusion remains stable (σ_max=-0.458/s). The 2D noise seed is an extra control: the transverse mode survives. a and b stay fixed.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L237–261; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L701–712
- **Morphometric validation: 90** — Pass-1 statistics stand (r=0.99937 vs 1, RMSE=0.0271 vs 0, KS=0.067 vs 0.05, furrows 11 vs 12, circularity 0.191 vs 1, coherency 0.613 vs 0.5). The 2D dominant wavelength is 3.125 µm against the 1D fastest-mode reference 3.146 µm, so the new metric is not self-referential. KS is printed on the figure.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L432–536; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L695–698; `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 90** — Unchanged from Pass 1. DOIs resolved via doi.org on 2026-10-27. Specimen SHA-256 is the Commons original. OpenAlex and Crossref were rate-limited (HTTP 429).  
  evidence: `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/aulacopleura_koninckii_meraspis_turing_wavelength_s0_reference.json` L48–80
- **Code-to-model traceability: 86** — Docstring equations plus the 2D patch docstring map the new panel to the same k_c. The discrete-versus-continuous distinction remains explicit in the probe.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L269–276
- **Reproducibility package: 88** — Self-test now also rejects a 2D wavenumber ratio outside [0.7, 1.35]. Seed, SHA pin, and the README invocation are unchanged. GIF gate is 16 frames at 720×560.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L761–805; `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/README.md` L14–28

Remediation:
- Still one assessor. A second model has not scored these dimensions.
- The Commons photograph has no scale bar, so λ_c = 3.125 µm is not an image measurement.
- Counted axial furrows are 11, not the assigned meraspid degree 12. The dorsal photo is not demonstrated to be an M12 individual.
- The dispersion fit uses the discrete Laplacian symbol on one grid. A continuum-limit refinement study is not included.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg critical wavelength at 3.125 µm for Aulacopleura koninckii meraspis (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `a620df60d6c5b5e97ca4ce3d5911512c60ca0cb80af273a47beb27e51cc495ce`
- **input_dataset_sha256**: `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/aulacopleura_koninckii_meraspis_turing_wavelength_s0_reference.json` (`61b94136e918acb9…`), `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/aulacopleura_koninckii_meraspis_turing_wavelength_s0_specimen.jpg` (`50ac5dbca8261978…`)
- `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/metrics.json` — artifact_sha256 `624bc242a16e1a44…`, 2,358 B, 
- `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_comparison.png` — artifact_sha256 `359856907d3dc084…`, 473,511 B, 1500×1080 px
- `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_growth.gif` — artifact_sha256 `140c626adc27a314…`, 2,942,415 B, 720×560 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_pearson_r | 0.999366 1 | 1.0 | Pearson r between discrete-Laplacian σ(k) and eigenmode growth of the explicit PDE on the band 0.78–1.28 k_c |
| dispersion_rmse | 0.027057 1/s | 0.0 | RMSE of numerical minus analytical growth rate on the same wavenumber grid |
| ks_dispersion_residual | 0.066667 1 | 0.05 | two-sample KS statistic of demeaned σ curves; p=1 *(uncertainty via KS and RMSE)* |
| stripe_wavelength_um | 3.146109 µm | 3.125 | argmax_k σ(k) of the linear dispersion; classical λ_c is the reference |
| equal_diffusion_control_sigma_max | -0.458018 1/s | 0.0 | max Re σ(k) with D_u = D_v (negative control / ablation of differential diffusion) *(ablation control versus the differential-diffusion baseline)* |
| specimen_thoracic_furrow_count | 11.0 count | 12.0 | peak count on a Gaussian-smoothed axial transect of the vendored dorsal photograph |
| specimen_contour_circularity | 0.191 1 | 1.0 | 4πA/P^2 on the border-contrast foreground mask (160×120) |
| specimen_structure_tensor_coherency | 0.6125 1 | 0.5 | mean structure-tensor coherency inside the specimen mask |
| patch2d_dominant_wavelength_um | 3.125 µm | 3.146109 | dominant anteroposterior wavelength of a 48×36 periodic pleural patch seeded with a transverse lira plus mediolateral noise *(2D control versus the 1D transect baseline)* |
| chronoseries_segment_surplus | 6.0 segments | 0.0 | holaspis18 thoracic count minus meraspid degree 12, at fixed λ_c *(successor ratio 18/12 = 1.5 under conserved k_c)* |

## Environment at checkpoint


Pass 2 working tree. Same parent commit as Pass 1; new paths are this loop's files.
