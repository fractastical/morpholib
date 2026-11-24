# MorphoEval report — audit `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit -> 2026-10-27-cloudina-hartmannae-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cloudina hartmannae wall-normal Turing lamella at 2.397 µm (Pass 2) (`models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py`) | 90 | 84 | 90 | 88 | 92 | 86 | 90 | 90.0 | 85.5 | 89.6 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Cloudina hartmannae wall-normal Turing lamella at 2.397 µm (Pass 2)  —  overall 88.1 (faithful)

Re-run of the 2.397 µm wall-normal Schnakenberg model (a=0.0642, b=0.9915, Dv/Du=8.65, seed 3149, Nama Group 549 Ma). Noise-seeded integration on the 19.176 µm layer selects 8 lamellae at 2.397 µm. Real-eigenvalue growth rates match the analytical dispersion with Pearson r=0.99997 and RMSE=0.0312. Scaling both diffusivities by s moves the peak wavelength monotonically (sweep r=0.9999). Equal diffusion gives σmax=-0.124. Removing u^2 v drops the activator standard deviation from 0.164 to ~1e-14. Species-card foreground fraction 0.364 versus 0.635 on the Corumbá fossil photograph. GIF last-frame CV similarity against the species card is 91.5 (12 frames, 640×480). The KS test against N(2.75 µm, 1.125 µm) rejects (statistic 0.623, p=0.0016): the simulator locks to one wavelength inside the Yang band and does not sample the whole 0.5–5 µm range. Spacing standard deviation is 0 because the domain is exactly eight critical wavelengths.

- **Model fidelity to source: 90** — k_c is solved from the Jacobian so the critical wavelength is 2.397 µm, and the same stepper supplies the nonlinear field, the ablation, and the growth-rate probe. Complex long-wave eigenvalues are plotted and excluded from r, which matches the oscillation those modes actually have. The near-onset discriminant (0.010) is why the fastest real mode is n=8 rather than a longer wave.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L100–134; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L196–233; `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_metrics.json` L0–0
- **Morphological realism / specimen match: 84** — The growth GIF is a 12-frame 640×480 animation whose last frame scores 91.5 against the assigned Cloudina hartmannae species card on orientation, radial FFT, coherency, and occupancy. The activator panel shows eight wall-normal bands at the lamella wavelength. The card remains a 100 µm picture, so the score does not treat this as an SEM overlay of Nama lamellae.  
  evidence: `docs/media/cloudina_hartmannae_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L353–402; `docs/media/cloudina_hartmannae_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Multiplying both diffusivities by s in {0.64, 0.81, 1, 1.21, 1.44, 1.69, 1.96} moves the numerical peak wavelength with the analytical curve λ=2.397√s. Sweep Pearson r=0.9999 and the sequence is monotone. That is the expressivity check Pass 1 asked for. Equal diffusion, a separate axis, shuts the instability off.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L284–318; `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 88** — Real-eigenvalue σ(k) agrees with the stepper at r=0.99997, RMSE=0.0312. Equal-diffusion σmax=-0.124. Ablation std is ~1e-14 against a patterned std of 0.164. Species-card foreground 0.364, coherency 0.832, circularity 0.057 versus Corumbá fossil 0.635, 0.342, 0.449; none of those pairs are copies. KS against the band-wide normal rejects (0.623, p=0.0016), which is reported as a spike at one wavelength rather than a fit to the whole 0.5–5 µm range. Spacing std is 0 on the commensurate domain, so the score stays below 90.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L273–282; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L320–351; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L562–678; `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Crossref confirmed Turing 1952, Schnakenberg 1979, Kondo and Miura 2010, Yang 2020, Germs 1972, and Penny 2014. Grant 1990 is cited by its PubMed URL because no DOI was returned. Image SHA-256 pins cover the species card, the CC BY-SA life restoration, and the CC BY 2.0 Corumbá photograph, and the restoration is labeled as not a fossil.  
  evidence: `datasets/cloudina_hartmannae_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/cloudina_hartmannae_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 86** — Dispersion, sweep, controls, segmentation, and rendering are separate functions, and the Lean module proves the band membership, the eight-lamella stack, the 43–44× funnel contrast, diffusion-rescaling of the discriminant, and the integer wavenumber balance. lean is not installed in this environment, so the proofs were not executed.  
  evidence: `morphoeval/lean/loops/Loop_3149_cloudina_hartmannae_turing_wavelength_s0.lean` L30–67; `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L1–40
- **Reproducibility package: 90** — Self-test passed on this run: r>0.98, monotone sweep, equal-D σmax<0, ablation ratio <0.05, spacing within 8% of 2.397 µm, 12 frames, min side 480 px, CV>=58, stack inside 8–50 µm, age 549 Ma. Seed and image hashes are pinned. Nanopub signing was not available (~/.nanopub/profile.yml absent) and was not faked.  
  evidence: `models/2026-delphi10000-cloudina-hartmannae-turing-wavelength-s0.py` L695–710; `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_reference.json` L70–109

Remediation:
- Still open: a Nama Group thin-section of C. hartmannae lamellae would replace the species-card CV target and the band-midpoint comparison.
- Still open: spacing standard deviation is 0 on a domain of exactly eight critical wavelengths; an incommensurate domain would separate mode selection from grid locking.
- Still open: lean binary was not available, and no second assessor scored this audit.
- Closed from Pass 1: diffusivity sweep (r=0.9999, monotone), equal-diffusion and ablation controls, card-versus-Corumbá segmentation, and dispersion r/RMSE with the KS rejection reported as a rejection.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cloudina hartmannae wall-normal Turing lamella at 2.397 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `aed8c0ba57f9ef79f97072aa88a12b01ec7259b22f96b35a3a9a2a503fa4efe9`
- **input_dataset_sha256**: `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_corumba_cc_by_2.jpg` (`f20328a5dcb570d1…`), `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_specimen.jpg` (`cbe0233dbf275f4e…`), `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_reference.json` (`dcd1cbc35eecb656…`)
- `datasets/cloudina_hartmannae_turing_wavelength_s0/cloudina_hartmannae_turing_wavelength_s0_metrics.json` — artifact_sha256 `34a24c4d8ec75173…`, 1,175 B, 
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


Pass 2 working tree. publisher_orcid is the nanopub setup-guide placeholder, not a person. No profile, so these nanopubs are unsigned and were not given npx:supersedes trusty URIs.
