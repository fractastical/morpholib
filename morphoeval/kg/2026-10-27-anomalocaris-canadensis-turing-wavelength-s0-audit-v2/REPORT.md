# MorphoEval report — audit `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `63ace896bba72dd95b1823861b5b9936be9cf066` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit -> 2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Anomalocaris canadensis Turing wavenumber k_c at 24 µm (508 Ma) (`models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py`) | 90 | 84 | 88 | 91 | 92 | 86 | 90 | 89.3 | 86.6 | 89.6 | 88.3 | **88.3** | faithful |

## Findings and remediation

### Anomalocaris canadensis Turing wavenumber k_c at 24 µm (508 Ma)  —  overall 88.3 (faithful)

Pass 2 accepts the Du sweep (lambda_c from 5.93 to 10.73 cells, monotonic), the equal-diffusion ablation (std ratio about 0.0026), the Tyrrell-cast segmentation (foreground 0.589, circularity 0.303, coherency 0.559), and the stencil-versus-numerical growth-rate fit (Pearson r = 0.999989, RMSE = 0.00626, KS D = 0.071 against a zero-growth null of 0.714). The 508 Ma wavelength remains 2.397e-05 m, below the reconstructed 266 µm Emu Bay facet spacing. What changes at the Ordovician successor is body size, not this cellular wavenumber.

- **Model fidelity to source: 90** — The code integrates the Schnakenberg PDE, evaluates the Turing discriminant, and checks the discrete growth rate against the stencil symbol at 14 wavenumbers (r = 0.999989). The kinetics are the standard spot system, not a rate law measured on this fossil.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L83–172; `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L180–214
- **Morphological realism / specimen match: 84** — The comparison plate and the 12-frame GIF use the CC BY-SA Tyrrell cast as the organism image, with the activator grown in a footer strip and as a light registration on the slab. The shale itself does not display 24 µm ocelli, so the match is the animal plus an explicit scale-separated field, not a photographic twin of a pigment pattern.  
  evidence: `docs/media/anomalocaris_canadensis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/anomalocaris_canadensis_turing_wavelength_s0_growth.gif` L0–0; `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg` L0–0
- **Parameter expressivity: 88** — Scaling both diffusivities from 0.55 to 1.80 moves analytical lambda_c monotonically from 5.93 to 10.73 cells, and every scale stays Turing-unstable. Setting Du = Dv makes the discriminant negative and the field std collapses (ratio about 0.0026).  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L216–232; `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L567–604
- **Morphometric validation: 91** — Pearson r = 0.999989 against a perfect-agreement reference of 1, RMSE = 0.00626 against 0, and KS D = 0.071 against a zero-growth null of 0.714. The cast mask is not a circle (circularity 0.303 versus 1) and the slot wavelength is 0.090 of the reconstructed 266 µm facet spacing versus a ratio of 1.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L149–172; `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L234–266; `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L466–565
- **Source and data traceability: 92** — Seven Crossref-checked DOIs, a SHA-256 pin of the Commons file, catalogue TMP 2023.003.0003 / ROMIP 51211, CC BY-SA 4.0, artist Chris Woodrich. The scout note records the OpenAlex 429.  
  evidence: `datasets/anomalocaris_canadensis_turing_wavelength_s0/paper_citations.bib` L1–79; `datasets/anomalocaris_canadensis_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/anomalocaris_canadensis_turing_wavelength_s0/README.md` L1–37
- **Code-to-model traceability: 86** — Each scored closure is a named function with a docstring that points at the law. The Lean file proves the integer forms of the trace sign, the diffusion-ratio gate, the sub-facet inequality, and the age order. There is no line-by-line map onto Turing's 1952 section numbers.  
  evidence: `morphoeval/lean/loops/Loop_3113_anomalocaris_canadensis_turing_wavelength_s0.lean` L1–61; `morphoeval/math_kg_loops/delphi10000-anomalocaris-canadensis-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 90** — python3 script --self-test checks trace, determinant, discriminant, slot length, equal-diffusion stability, Pearson r > 0.98, monotonic sweep, sub-facet inequality, age order, segmentation, and ablation std. Seed 3113. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L567–649

Remediation:
- R5 (open): only one photographed object, and it is a museum cast rather than the original Burgess surface. A second specimen would test whether the mask statistics are about this mount.
- R6 (open): a and b are a spot-capable regime, not rates measured on anomalocaridid cuticle. No 24 µm pigment map exists to fit them to.
- R7 (open): single assessor. A second provider has not scored this audit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Anomalocaris canadensis Turing wavenumber k_c at 24 µm (508 Ma)
- **spec_fingerprint** (semantic model + inputs): `6e60ae9de13486d021af674ae24c3fe415cc4c359fddd14174f41704efc81a34`
- `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_TMP2023_003_0003_tyrrell.jpg` — artifact_sha256 `0dcc1da801bddc9b…`, 6,921,317 B, 2820×4010 px
- `docs/media/anomalocaris_canadensis_turing_wavelength_s0_comparison.png` — artifact_sha256 `8c788c0b684071e1…`, 661,168 B, 1344×1008 px
- `docs/media/anomalocaris_canadensis_turing_wavelength_s0_growth.gif` — artifact_sha256 `fdc6e92b6f8b2b81…`, 3,313,375 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_growth_rate_pearson_r | 0.999989 Pearson r | 1.0 | 1D Schnakenberg cosine projected for 120 steps versus stencil growth rate σ(q), 14 wavenumbers *(Reference 1 is perfect numerical-analytic agreement. The measured r is lower.)* |
| dispersion_growth_rate_rmse | 0.006263 1/time | 0.0 | RMSE of numerical minus stencil growth rates on q in [0.25, 1.25] *(Reference 0 would be a perfect residual. RMSE is the KS companion scale.)* |
| dispersion_growth_rate_ks_statistic | 0.071429 KS D | 0.714286 | Two-sample KS of numerical versus stencil growth rates on the same 14 wavenumbers *(The reference D is the no-growth null. The fit D is the stencil comparison and is smaller.)* |
| specimen_image_foreground_fraction | 0.5885 fraction | 0.45 | border-median contrast mask on the Tyrrell cast, 160x120 *(0.45 is the mask prior used only as a non-equal comparator, not a published area.)* |
| specimen_contour_circularity | 0.3026 4πA/P² | 1.0 | erosion perimeter of the border-contrast mask *(The slab-plus-animal silhouette is not a circle.)* |
| specimen_stripe_coherency | 0.5589 (λ1-λ2)/(λ1+λ2) | 0.0 | mean structure-tensor coherency inside the specimen mask *(Shale cracks and flap margins are oriented, so coherency is above the isotropic null.)* |
| specimen_subfacet_to_facet_spacing_ratio | 0.0901 ratio | 1.0 | slot λ 23.97 µm divided by reconstructed 2.5 cm hemisphere spacing at 16000 lenses *(A ratio of 1 would mean this slot's wavelength was the facet spacing. It is not.)* |
| equal_diffusion_ablation_std_ratio | 0.0026 ratio | 1.0 | std(u | Du=Dv) / std(u | Dv/Du=20) after 700 steps on a 64x48 Neumann patch *(Reference 1 would mean the ablation still formed the Turing pattern.)* |
| du_sweep_lambda_monotonic | 1.0 boolean | 0.0 | eight diffusivity scales from 0.55 to 1.80; analytical λ_c must increase *(0 would record a non-monotonic sweep.)* |
| specimen_vs_modulated_luminance_ks | 0.1044 KS D | 0.0233 | KS of cast luminance versus the 0.16-alpha activator modulation *(The reference is the scrambled image, not a copy of the modulated KS.)* |

## Environment at checkpoint


Slot 3113 Pass 2. Same commit as Pass 1; the closures are the sweep, ablation, segmentation, and dispersion fit already in that script, now scored.
