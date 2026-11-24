# MorphoEval report — audit `2026-10-27-cornus-controversa-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-cornus-controversa-turing-wavelength-s0-audit -> 2026-10-27-cornus-controversa-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cornus controversa Turing wavenumber at 1.939 mm (Holocene, 0 Ma) (`models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py`) | 90 | 74 | 90 | 88 | 90 | 86 | 88 | 90.0 | 79.2 | 88.1 | 85.1 | **85.1** | faithful |

## Findings and remediation

### Cornus controversa Turing wavenumber at 1.939 mm (Holocene, 0 Ma)  —  overall 85.1 (faithful)

Pass 2, superseding the Pass 1 audit (overall 82.7). The linear critical wavelength remains 1.939 mm with Pearson r = 0.9999998 and RMSE = 2.13e-4. The unbiased saturated wavelength is 2.216 mm on both the 8-wavelength patch (Pass 1) and the 16-wavelength patch (Pass 2), so the shift is not a coarse-bin artifact: the 16-wavelength box could have peaked at 2.068 mm or 1.939 mm and peaked at 2.216 mm. A midrib template of amplitude 0.004 at 1.939 mm entrains the field to 1.939 mm and raises the horizontal-edge fraction from 0.286 to 0.934, against 0.433 on the crown photograph. The photograph has no scale bar. Equal-diffusion ablation is unchanged (sigma_max -0.0615, std ratio 0.018). Weighted score 85.1/100. R4 and R5 remain open.

- **Model fidelity to source: 90** — The linear law is unchanged from Pass 1 and still matches equations (1)–(6): lambda_c = 1.939000 mm, trace -0.120, det 1, and eleven real modes with r = 0.9999998. Pass 2 does not retune the kinetics to hide the nonlinear shift. The template is reported as a separate forcing, not as a replacement for k_c.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L90–172; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L434–505; `morphoeval/lean/loops/Loop_3155_cornus_controversa_turing_wavelength_s0.lean` L10–44
- **Morphological realism / specimen match: 74** — The GIF is still 12 frames at 480x640 and the last frame is the Meise crown (CV similarity 96.0). Pass 2 keeps that anchor and states, in the metrics, that the frame cannot support a millimetre reading. The generative picture remains a spot field, now with the honest nonlinear wavelength 2.216 mm rather than a template-forced stripe field presented as the organism. Horizontal-edge fraction is 0.286 on that field and 0.433 on the crown. The +2 from Pass 1 is the resolution statement and the refusal to paint the template over the selected pattern.  
  evidence: `docs/media/cornus_controversa_turing_wavelength_s0_comparison.png` L0–0; `docs/media/cornus_controversa_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L417–432
- **Parameter expressivity: 90** — The Dv sweep is unchanged and still monotonic: 1.998 mm down to 1.881 mm as Dv/Dv* goes from 0.65 to 2.40, with the nominal point at 1.939 mm. Pass 2 additionally runs the midrib template at amplitude 0.004. It moves the saturated wavelength from 2.216 mm to 1.939 mm and the horizontal-edge fraction from 0.286 to 0.934. That is a large, directional response. It also shows the template dominates, which is why it is not used as the primary field.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L318–325; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L82–83; `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 88** — Linear fit is unchanged: r = 0.9999998, RMSE = 2.13e-4, KS = 0.514 (p = 0.003) against a normal residual model that the truncation error does not obey. Unbiased nonlinear wavelength is 2.216 mm on the 16-wavelength patch, the same peak Pass 1 found on the 8-wavelength patch, against the linear reference 1.939 mm. The larger box has a bin at 2.068 mm between those two values, and the field did not use it. Ablation std ratio remains 0.018. Specimen foreground 0.866 versus null 0.50, foliage 0.212 versus null 0.25, coherency 0.363 versus the field orientation 0.286, horizontal-edge fraction 0.433 versus 0.286. scale_bar_present is 0 against a reference of 1. The KS miss and the 14 percent nonlinear shift keep this under 90.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L236–283; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L381–415; `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Same Crossref-checked bibliography and the same pinned specimen digest as Pass 1. The reference JSON states 0.001939 m and 0 Ma. The scout manifest records why --slot 3155 could not be used. Pass 2 does not add citations, and it does not need to: the new numbers are measurements, not new sources.  
  evidence: `datasets/cornus_controversa_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_reference.json` L1–40; `datasets/cornus_controversa_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 86** — Pass 1 mapping stands. Pass 2 adds run_core's split between the unbiased integrator and the template integrator, and resolution_statement() is the scale-bar claim. Lean witnesses are still integer `decide` proofs of the scaled Jacobian, not a PDE formalization.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L197–224; `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L434–505; `morphoeval/math_kg_loops/delphi10000-cornus-controversa-turing-wavelength-s0.yaml` L1–61
- **Reproducibility package: 88** — --self-test still passes on the linear law, the sweep, and the ablation. The default run now executes Pass 2 and rewrites the PNG, the GIF, and the metrics JSON from seed 3155. The dependency set is still the environment stack rather than a per-script lock.  
  evidence: `models/2026-delphi10000-cornus-controversa-turing-wavelength-s0.py` L803–842; `datasets/cornus_controversa_turing_wavelength_s0/README.md` L1–31

Remediation:
- {'id': 'R1', 'item': 'Resolved. The 2.216 mm peak survives on a 16-wavelength patch, which has a bin at 2.068 mm between 2.216 and 1.939. It is a nonlinear shift, not a coarse-bin accident.'}
- {'id': 'R2', 'item': 'Resolved. Bias 0.004 is executed as a separate template. It entrains lambda to 1.939 mm and is not substituted for the unbiased field.'}
- {'id': 'R3', 'item': 'Resolved. resolution_limit records scale_bar_present false and mm_per_px null.'}
- {'id': 'R4', 'item': "Open. No second assessor. This score is still the builder's own."}
- {'id': 'R5', 'item': 'Open. ~/.nanopub/profile.yml is absent, so the nanopubs are not signed.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cornus controversa Turing wavenumber at 1.939 mm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `889436d47b7f6a20b524144198f645dcccbc348d99fd9ce666f715dfafc2a48a`
- **input_dataset_sha256**: `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_specimen.jpg` (`be3da2bb8b817bec…`), `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_reference.json` (`61581ec4249a6ec3…`)
- `datasets/cornus_controversa_turing_wavelength_s0/cornus_controversa_turing_wavelength_s0_metrics.json` — artifact_sha256 `7b630848827e99ff…`, 4,769 B, 
- `docs/media/cornus_controversa_turing_wavelength_s0_comparison.png` — artifact_sha256 `03cd6470ad1f2e0c…`, 482,774 B, 1260×840 px
- `docs/media/cornus_controversa_turing_wavelength_s0_growth.gif` — artifact_sha256 `7903bfcca2a12b30…`, 1,971,011 B, 480×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (PDE vs analytic sigma) | 1.0 1 | 0.98 | early-time Fourier growth of 12 discrete Laplacian modes vs eigenvalue σ(k) *(RMSE=0.00021, KS=0.5144, p_pearson=4.21e-30)* |
| dispersion RMSE | 0.0002 1/time | 0.0 | root mean square of σ_measured − σ_analytic over the probed band |
| dispersion KS statistic | 0.5144 1 | 0.0 | Kolmogorov–Smirnov statistic of σ residuals against Normal(0, s) *(ks_p=0.0031, n_modes=11)* |
| nonlinear dominant wavelength | 2.216 mm | 1.939 | radial FFT peak of the unbiased saturated activator |
| ablation equal-diffusion std ratio | 0.0183 1 | 1.0 | std(u_final)/std(u_initial) with D_u = D_v = D_u*, noise seed 3162 *(sigma_max=-0.0615 (must be negative))* |
| specimen foreground fraction | 0.8659 1 | 0.5 | border-contrast mask on cornus_controversa_specimen.jpg (crown photograph, not the 1.939 mm patch) |
| specimen structure-tensor coherency | 0.3634 1 | 0.286 | mean structure-tensor coherency inside the foreground mask *(reference is the simulated orientation fraction, not a copy of the coherency)* |
| specimen horizontal-edge fraction | 0.4328 1 | 0.286 | share of Sobel energy within 30 degrees of a horizontal edge on the crown photograph |
| specimen foliage bright fraction | 0.2118 1 | 0.25 | luminance > 0.45 above the lower 20 percent lawn band |
| midrib-template entrained wavelength | 1.939 mm | 2.216 | radial FFT peak with production bias 0.004 at 1.939 mm on the 16-wavelength patch *(A cosine production template at 1.939 mm entrains the saturated wavelength. The unbiased field on the same 16-wavelength patch keeps the longer nonlinear peak, so the Pass-1 shift is not a finite-box binning artifact.)* |
| specimen image scale-bar present | 0 boolean | 1 | visual inspection of the vendored crown frame: no scale bar *(The vendored crown photograph has no scale bar, so a 1.939 mm pigment wavelength cannot be measured on it. Wavelength claims use the dispersion relation on the lamina patch. Crown pixels support only orientation, occupancy, and coherency.)* |

## Environment at checkpoint


Pass 2 working tree on top of 0c76d15d. PASS2_REMEDIATION is True. The unbiased field is integrated on a 16-wavelength patch and a separate midrib template is reported. No nanopub profile is present.
