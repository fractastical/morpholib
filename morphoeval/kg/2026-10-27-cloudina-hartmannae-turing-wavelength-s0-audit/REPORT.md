# MorphoEval report — audit `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cloudina hartmannae wall-normal Turing lamella at 2.397 µm (Pass 1) (`models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py`) | 82 | 74 | 70 | 72 | 88 | 80 | 84 | 78.0 | 73.2 | 84.3 | 78.0 | **78.0** | mostly faithful |

## Findings and remediation

### Cloudina hartmannae wall-normal Turing lamella at 2.397 µm (Pass 1)  —  overall 78.0 (mostly faithful)

Pass 1 implements Schnakenberg kinetics with Du and Dv fixed so the Turing critical wavelength is 2.397 µm on a 19.176 µm wall-normal interval (eight lamellae inside one Grant shell layer). The noise-seeded field produced eight peaks at 2.397 µm. Points are withheld until the diffusivity sweep, equal-diffusion and ablation controls, Corumbá/card segmentation, and the dispersion goodness-of-fit are audited as measurements.

- **Model fidelity to source: 82** — The stepper is Schnakenberg kinetics plus an explicit Laplacian, and k_c is derived from the Jacobian rather than imposed as a drawn cosine. Long-wave eigenvalues of this Jacobian are a complex pair; Pass 1 does not yet credit the decision to keep those modes out of the correlation.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L89–108; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L146–170
- **Morphological realism / specimen match: 74** — The comparison figure shows the species card beside an eight-band activator trace. The card's own scale bar is 100 µm, so a high computer-vision score against that card is not a match to a Nama lamella SEM. Pass 1 holds the score down for that gap.  
  evidence: `docs/media/cloudina_hartmannae_turing_wavelength_s0_comparison.png` L0–0; `docs/media/cloudina_hartmannae_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 70** — Diffusivity scale is the parameter that must move wavelength as sqrt(s). The sweep function is in the script, but Pass 1 does not yet treat its monotonicity or correlation as an audited measurement.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L297–318
- **Morphometric validation: 72** — Peak spacing on the commensurate domain is exactly 2.397 µm with standard deviation 0, which is finite-size locking as much as a test. Dispersion correlation, KS against the Yang band, and the Corumbá segmentation are not yet pinned as measurements.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L235–271
- **Source and data traceability: 88** — Turing, Schnakenberg, Yang, Germs, and Penny resolve by DOI. Grant 1990 has no DOI in OpenAlex or Crossref; the audit uses the PubMed URL instead of inventing one.  
  evidence: `datasets/cloudina_hartmannae_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — Each modelling step is a named function tied to the dispersion relation. The Lean file states the band, stack, and rescaling theorems, but lean is not installed here so those proofs were not executed.  
  evidence: `morphoeval/lean/loops/Loop_3149_cloudina_hartmannae_turing_wavelength_s0.lean` L30–67; `morphoeval/math_kg_loops/delphi10000-cloudina-hartmannae-turing-wavelength-s0.yaml` L1–30
- **Reproducibility package: 84** — Seed 3149, pinned image SHA-256, and a --self-test are present. Pass 1 does not yet record a completed self-test log in the audit trail, and the nanopub profile is absent.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L71–87; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L695–710

Remediation:
- Parameter-regime sweep: audit the diffusivity-scale sweep and require monotone growth of wavelength with sqrt(s).
- Negative and ablation controls: record equal-diffusion max growth rate and the autocatalysis-ablation amplitude ratio.
- Real specimen CV segmentation: segment the species card and the Corumbá fossil and require the foreground fractions to differ.
- Goodness-of-fit: pin Pearson r and RMSE of real-eigenvalue growth rates, and report the KS statistic against the Yang band without treating a rejection as a pass.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cloudina hartmannae wall-normal Turing lamella at 2.397 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `aed8c0ba57f9ef79f97072aa88a12b01ec7259b22f96b35a3a9a2a503fa4efe9`
- **input_dataset_sha256**: `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_corumba_cc_by_2.jpg` (`f20328a5dcb570d1…`), `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_specimen.jpg` (`cbe0233dbf275f4e…`), `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_reference.json` (`dcd1cbc35eecb656…`)
- `docs/media/cloudina_hartmannae_turing_wavelength_s0_comparison.png` — artifact_sha256 `bb3134888de8b71d…`, 379,283 B, 1344×984 px
- `docs/media/cloudina_hartmannae_turing_wavelength_s0_growth.gif` — artifact_sha256 `e890de0a8d59dd17…`, 1,884,937 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion Pearson r (analytical σ vs PDE stepper) | 1.0 1 | 0.98 | corrcoef of σ(k) on periodic modes whose eigenvalues are real (complex long-wave pairs excluded) |
| Turing dispersion RMSE | 0.0312 1/time | 0.0 | sqrt mean square error between analytical and numerical σ(k) |
| specimen-image lamella stripe wavelength | 2.397 µm | 2.75 | mean distance between activator peaks on the noise-seeded wall-normal field |
| Cloudina wall lamella stripe count versus funnel-rung count on the same layer | 8 count | 0.184 | periodic peak count on the 19.176 µm wall-normal domain |
| KS distance of lamella spacings vs Yang-band normal | 0.6232 1 | 0.0 | Kolmogorov-Smirnov statistic of peak spacings against N(2.75 µm, 1.125 µm) *(two-sided p = 0.001581)* |
| specimen-image foreground fraction | 0.3636 1 | 0.6345 | border-contrast mask on the Cloudina hartmannae species card |
| specimen-image structure-tensor coherency | 0.8315 1 | 0.3415 | mean structure-tensor coherency inside the species-card foreground |
| specimen-image contour circularity | 0.057 1 | 0.4489 | 4πA/P^2 of the species-card foreground |
| equal-diffusion negative-control max growth rate | -0.1239 1/time | 0.0 | max σ(k) with Dv reset to Du |
| autocatalysis-ablation activator std vs full pattern std | 0.0 1 | 1.0 | std(u) after removing u^2 v, divided by std(u) of the full model |
| diffusivity-scale sweep Pearson r | 0.9999 1 | 0.98 | corrcoef of analytical λ(s)=2.397√s against the numerical peak wavelength |
| CV similarity of growth GIF last frame vs species card | 91.5 score | 58.0 | verify_image_fitness.compute_sim_vs_specimen_cv_similarity (orientation, radial FFT, coherency, occupancy) |

## Environment at checkpoint


Pass 1 working tree. New slot files are not in this commit. publisher_orcid is the nanopub setup-guide placeholder, not a person: ~/.nanopub/profile.yml is absent, so the graph is unsigned.
