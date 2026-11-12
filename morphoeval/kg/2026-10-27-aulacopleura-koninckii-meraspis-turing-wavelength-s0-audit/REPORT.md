# MorphoEval report — audit `2026-10-27-aulacopleura-koninckii-meraspis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg critical wavelength at 3.125 µm for Aulacopleura koninckii meraspis (Pass 1) (`models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py`) | 84 | 78 | 84 | 82 | 90 | 82 | 86 | 84.0 | 79.5 | 86.3 | 82.9 | **82.9** | faithful |

## Findings and remediation

### Schnakenberg critical wavelength at 3.125 µm for Aulacopleura koninckii meraspis (Pass 1)  —  overall 82.9 (faithful)

Pass 1 integrates a near-onset Schnakenberg system whose classical k_c is 2π/3.125 µm. Eigenmode growth on the pattern-forming band matches the discrete dispersion relation at Pearson r = 0.99937, RMSE = 0.0271 /s. Equal diffusion gives σ_max = -0.458 /s. A diffusion-ratio sweep moves λ_max from 3.146 µm to 3.993 µm. The vendored dorsal photograph yields 11 axial furrows against the assigned meraspid count of 12, circularity 0.191, coherency 0.613. The GIF (720×560, 13 frames at this pass) ends on that photograph (CV similarity 95.7). k_c is held fixed into the holaspis18 successor; the trunk segment surplus is 6.

- **Model fidelity to source: 84** — The script steps the Schnakenberg PDE and the classical k_c formula, and the pattern-band eigenmodes track the discrete dispersion relation (r=0.999). Parameters are reconstructed to hit 3.125 µm; they are not a published trilobite kinetic table. Pass 1 does not yet score a 2D patch.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L8–28; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L98–127; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L180–235
- **Morphological realism / specimen match: 78** — The growth GIF ends on the vendored fossil and the comparison figure places that fossil beside a lira field. The lira panel is a generated stripe texture, not the exoskeleton, and Pass 1 does not draw the segmentation contour. Counted furrows (11) are not forced to the assigned 12.  
  evidence: `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_growth.gif` L0–0; `docs/media/aulacopleura_koninckii_meraspis_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L344–388
- **Parameter expressivity: 84** — Raising D_v/D_u from 2% to 140% above onset moves λ_max monotonically from 3.146 µm to 3.993 µm. Setting D_u = D_v makes σ_max negative. a and b are not swept.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L237–254; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L256–261
- **Morphometric validation: 82** — Dispersion Pearson r=0.99937 against a reference of 1, RMSE=0.0271 against 0, KS=0.067 against 0.05, furrow count 11 against 12, circularity 0.191 against 1, coherency 0.613 against 0.5. None of these values is copied from its reference. The photograph supplies no micrometre scale for λ_c.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L432–520; `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 90** — Fusco 2004, Hughes & Chapman 1995, Hughes et al. 2006, Turing 1952, Schnakenberg 1979, and Murray 1981 were resolved on doi.org. The specimen is the CC BY-SA 3.0 Commons file, SHA-256 pinned. OpenAlex and Crossref returned HTTP 429 during scouting; doi.org succeeded.  
  evidence: `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/aulacopleura_koninckii_meraspis_turing_wavelength_s0_reference.json` L1–40; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L65–66
- **Code-to-model traceability: 82** — The module docstring names the PDE, the steady state, the Jacobian, and k_c, and the probe docstring states that the fit uses the discrete Laplacian symbol. Section-level pins into Turing 1952 are not line-numbered from the paper.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L1–40; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L180–192
- **Reproducibility package: 86** — Seed 3126, --self-test asserts r>0.98, λ_c, the ablation sign, the sweep span, and GIF size. Dependencies are the repo scientific stack. The parent commit does not contain these new paths.  
  evidence: `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L62–63; `models/2026-delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0.py` L761–805; `datasets/aulacopleura_koninckii_meraspis_turing_wavelength_s0/README.md` L14–22

Remediation:
- Add a 2D pleural-patch integration and report its dominant anteroposterior wavenumber against k_c. The Pass-1 law is a 1D transect.
- Draw the specimen segmentation contour on the comparison figure. The mask is computed but not shown.
- Print the KS statistic of the dispersion residuals on the dispersion panel.
- State the M12 to holaspis18 copy-ratio (18/12) on the sweep panel, not only the segment surplus.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg critical wavelength at 3.125 µm for Aulacopleura koninckii meraspis (Pass 1)
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


Pass 1 working tree. Evidence paths are introduced in this loop and are not in the parent commit.
