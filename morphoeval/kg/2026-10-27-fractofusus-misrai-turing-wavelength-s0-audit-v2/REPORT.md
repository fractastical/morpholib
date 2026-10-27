# MorphoEval report — audit `2026-10-27-fractofusus-misrai-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-fractofusus-misrai-turing-wavelength-s0-audit -> 2026-10-27-fractofusus-misrai-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing wavelength for Fractofusus misrai at 63 µm (565 Ma) (`models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py`) | 90 | 84 | 90 | 88 | 90 | 86 | 88 | 90.0 | 85.5 | 88.1 | 87.6 | **87.6** | faithful |

## Findings and remediation

### Schnakenberg–Turing wavelength for Fractofusus misrai at 63 µm (565 Ma)  —  overall 87.6 (faithful)

Pass 2 of the 63 µm (6.33e-05 m, log10 = -4.199) detail-scale model for Fractofusus misrai at Mistaken Point, 565 Ma (chronoseries 1/1). The linear dispersion fit is unchanged (Pearson r = 0.999999999, RMSE = 0.00375). The saturated Fourier wavelength now lengthens with D_u: 69.8 µm, 82.2 µm, 94.0 µm at D_u = 0.08, 0.12, 0.18, above the analytical curve and in the same direction. The nominal saturated peak remains 73.7 µm against the linear 63.3 µm. The render's interior is an oblique chevron of vanes on the siltstone spindle rather than the Pass 1 spot lattice. GIF versus the CC BY photograph scores 84.6 on the orientation / radial-FFT / coherency comparison. Specimen spindle aspect 1.70 versus render 1.12; gradient-angle KS D = 0.121. Equal diffusion and the 2×2 subcritical domain still decay. Fifty-one detail wavelengths fit in the 3.275 mm organ module and fifty-two do not.

- **Model fidelity to source: 90** — Same Schnakenberg–Turing system as Pass 1. The saturated field's Fourier peak now moves with D_u in the direction the dispersion relation predicts, which is a check on the nonlinear integration and not only on the Jacobian formula.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L108–134; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L266–292
- **Morphological realism / specimen match: 84** — Pass 2 replaces the spot lattice with oblique vanes inside the spindle, which is the rangeomorph reading of the Mistaken Point mould. Bedding colour and the elongate body remain. The fossil's vanes are irregular and cracked; the model's are periodic. Computer-vision similarity of the 12-frame 720×540 GIF against the CC BY photograph is 84.6, up from 79.4.  
  evidence: `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` L0–0; `docs/media/fractofusus_misrai_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L415–460
- **Parameter expressivity: 90** — Analytical λ_c rises from about 40 µm to about 93 µm as D_u goes from 0.05 to 0.25. The saturated Fourier wavelength at D_u = 0.08, 0.12, 0.18 is 69.8, 82.2, 94.0 µm, also increasing. Equal diffusion and a domain shorter than λ_c both decay.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L245–311; `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Dispersion Pearson r = 0.999999999 against reference threshold 0.98, RMSE = 0.00375. Saturated nominal wavelength 73.7 µm is not forced to 63.3 µm. Specimen aspect 1.70 versus render 1.12. Structure-tensor coherency 0.369 versus 0.454. Gradient-angle KS D = 0.121 on one photograph.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L176–190; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L354–396; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L640–737
- **Source and data traceability: 90** — Unchanged from Pass 1. Crossref-checked DOIs and the SHA-256-pinned CC BY 2.0 Commons photograph by djpmapleferryman.  
  evidence: `datasets/fractofusus_misrai_turing_wavelength_s0/paper_citations.bib` L1–86; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L75–85
- **Code-to-model traceability: 86** — Pass 2 adds measured_wavelength_sweep next to the analytical sweep, and the renderer comment states that the vane angle is display while the wavelength is the model's. The module docstring still carries the PDEs and the 51-versus-52 scale split.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L1–36; `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L266–292
- **Reproducibility package: 88** — Self-test now also requires the saturated D_u sweep to be monotonic and to grow. Seed, hashes, frame count, and the 480 px resolution floor are unchanged and still enforced.  
  evidence: `models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py` L760–776

Remediation:
- {'id': 'R3', 'item': 'Still one assessor. A second provider has not scored Pass 2.'}
- {'id': 'R4', 'item': 'The nonlinear wavelength (73.7 µm at nominal D_u, 82.2 µm on the sweep grid) is not brought back to 63.3 µm by a weakly nonlinear correction. The shift is reported rather than removed.'}
- {'id': 'R5', 'item': 'Specimen comparison is one unscaled bedding-plane photograph. Branch spacing on the fossil is not in micrometres.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing wavelength for Fractofusus misrai at 63 µm (565 Ma)
- **spec_fingerprint** (semantic model + inputs): `b7aed75670ec58eb2e24b6b33d4964b84814f0e27b8d16305477ab4160932f43`
- **input_dataset_sha256**: `datasets/fractofusus_misrai_turing_wavelength_s0/README.md` (`749222e73e20900f…`), `datasets/fractofusus_misrai_turing_wavelength_s0/fractofusus_misrai_specimen.jpg` (`e423fba39900f0b8…`), `datasets/fractofusus_misrai_turing_wavelength_s0/fractofusus_misrai_turing_wavelength_s0_reference.json` (`6b331d726b68779e…`), `datasets/fractofusus_misrai_turing_wavelength_s0/paper_citations.bib` (`365f45ca2b96a3cb…`), `datasets/fractofusus_misrai_turing_wavelength_s0/scout_manifest.json` (`00bdda7655784bfa…`)
- `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` — artifact_sha256 `1693932873ca663e…`, 731,414 B, 1344×1008 px
- `docs/media/fractofusus_misrai_turing_wavelength_s0_growth.gif` — artifact_sha256 `f5b296c5e085e725…`, 3,709,878 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion Pearson r of spectral σ(k) versus analytical σ(k) | 0.999999999192 Pearson r | 0.98 | k = 1..8 on a 2π ring; spectral diffusion of the linearised Schnakenberg Jacobian versus eig(J - k² D) *(RMSE = 0.003753; linear dispersion, not a copy of the reference series)* |
| dispersion RMSE between numerical and analytical growth rates | 0.003753 1/time | 0.0 | root mean square of σ_numerical(k) - σ_analytic(k) for integer k = 1..8 *(uncertainty on the linear wavenumber fit)* |
| nonlinear dominant wavelength on the saturated 63 µm field | 73.6621 µm | 63.3 | argmax of the 2D power spectrum of the saturated activator, times dx with analytical λ_c fixed at 6.33e-05 m *(nonlinear peak is not forced equal to the linear k_c; the reference is the assigned linear wavelength)* |
| specimen spindle aspect ratio (height over width) from the CC BY mould | 1.7 aspect | 1.1217 | Otsu threshold on smoothed Sobel magnitude of fractofusus_misrai_specimen.jpg; bounding support of the core mask *(reference_value is the simulated spindle aspect, which is a different number; KS and CV are separate)* |
| specimen structure-tensor coherency | 0.3686 coherency | 0.4541 | (λ1-λ2)/(λ1+λ2) averaged on the Sobel structure tensor of the Mistaken Point photograph *(KS statistic of gradient angles versus the GIF = 0.12075000000000002)* |
| equal-diffusion ablation final activator std | 0.000348 concentration std | 0.009975 | Schnakenberg with D_u = D_v = 2.40; Turing h = D*trace < 0 so the pattern must decay *(negative control versus the patterned D_v/D_u = 20 run)* |
| subcritical-domain ablation final activator std | 0.000159 concentration std | 0.00535 | 2 by 2 domain, both axes shorter than λ_dx, so k_min = π sits above the unstable band *(σ(k_min) = -0.4421)* |
| KS statistic of specimen versus simulation gradient-angle samples | 0.1208 KS D | 0.0 | two-sample KS on 4000 Sobel-magnitude-weighted gradient angles from the photograph and the GIF *(p = 8.189814844249964e-26; value is not the null reference 0)* |
| saturated Fourier wavelength increase from D_u = 0.08 to D_u = 0.18 | 24.2568 µm | 0.0 | dominant 2D Fourier wavelength of three saturated Schnakenberg fields at D_u = 0.08, 0.12, 0.18 *(null reference is a sweep with no D_u effect; wavelengths µm = [69.77, 82.22, 94.02])* |

## Environment at checkpoint


Pass 2 working tree. New slot files are not in 0c76d15d.
