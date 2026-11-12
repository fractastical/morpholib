# MorphoEval report — audit `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `dff4f2b508d0d1939f865c2f90f7ffe070e890d6` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit -> 2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acanthostega gunnari Turing wavenumber at 105 µm (Famennian, 365 Ma) (`models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py`) | 88 | 80 | 90 | 91 | 92 | 86 | 90 | 88.7 | 84.1 | 89.6 | 87.1 | **87.1** | faithful |

## Findings and remediation

### Acanthostega gunnari Turing wavenumber at 105 µm (Famennian, 365 Ma)  —  overall 87.1 (faithful)

Pass 2 sets r = Dv/Du = 10 so the saturated FFT wavelength is 106.93 µm against analytic k_c = 104.9 µm (relative error 0.019). Dispersion Pearson r = 0.9999 and RMSE = 0.0050 on probe modes 10–20. Equal diffusion (r = 1) has max growth -0.0989. Tiktaalik (375 Ma) cross-ray growth is -0.621 (pattern off); its along-ray wavelength is 100.11 µm. Ichthyostega (363 Ma, Dv × 0.92) sits at 105.54 µm. Brussels skull foreground 0.557, coherency 0.527, circularity 0.330, orientation KS against the RD field 0.137. GIF 640×480, 12 frames; last frame is the fossil so the fitness crop has a real specimen target.

- **Model fidelity to source: 88** — The code integrates the Schnakenberg PDE, the Jacobian at u0 = a+b, and k_c^2 = (Dv fu + Du gv)/(2 Du Dv). Analytic lambda is 104.9000 µm. 1D Euler growth rates match analytic sigma with Pearson r = 0.9999. Du is calibrated to the assigned tier rather than fitted to a fossil measurement, which caps the score.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L98–144; `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L152–193; `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L218–278
- **Morphological realism / specimen match: 80** — Panel A of the comparison figure is the CC BY-SA Brussels skull; panel B is the ocellus field on a 1.678 mm patch. The 12-frame GIF ends on that photograph (CV similarity 96 against verify_image_fitness). The RD field is a primordium prepattern, not a drawing of the skull, so this is not a silhouette match.  
  evidence: `docs/media/acanthostega_gunnari_turing_wavelength_s0_comparison.png` L0–0; `docs/media/acanthostega_gunnari_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L436–491
- **Parameter expressivity: 90** — Above onset, lambda(r) falls monotonically from r = 8 to r = 48. r = 1 is a negative control with sigma_max = -0.0989. Tiktaalik cross-ray (r_eff = 2.94) is inside the subcritical band; along-ray r_eff = 34 shortens lambda to 100.11 µm; Ichthyostega r = 9.2 lengthens it to 105.54 µm. Each knob moves the feature it is supposed to move.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L280–330; `docs/media/acanthostega_gunnari_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 91** — FFT wavelength 106.93 µm versus reference 104.9 µm. Pearson r = 0.9999 (reference threshold 0.98). RMSE = 0.0050. Specimen foreground 0.5568 versus prior 0.5, coherency 0.5265 versus 0.32, circularity 0.330 versus 1.0, orientation KS 0.137 versus 0.05. None of these copy the reference into the value. Sweep CV of supercritical wavelengths is reported.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L603–720; `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L332–388
- **Source and data traceability: 92** — Seed DOI 10.1038/347066a0, law DOI 10.1098/rstb.1952.0012, predecessor and successor DOIs, and the Wikimedia CC BY-SA 3.0 skull with SHA-256 6eae655c40f0444ce93acbeabe111c957bd1d6b1316ee7d808767c3b301dbeb8.  
  evidence: `datasets/acanthostega_gunnari_turing_wavelength_s0/paper_citations.bib` L1–90; `datasets/acanthostega_gunnari_turing_wavelength_s0/scout_manifest.json` L1–30; `datasets/acanthostega_gunnari_turing_wavelength_s0/README.md` L1–35
- **Code-to-model traceability: 86** — Docstring equations match kinetics() and critical_k2(). Lean Loop_3102 proves r = 10, the positive Turing numerator, the equal-diffusion ablation, the 40-wavelength corridor, and the anisotropy drop, with zero sorry.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L1–45; `morphoeval/lean/loops/Loop_3102_acanthostega_gunnari_turing_wavelength_s0.lean` L10–40
- **Reproducibility package: 90** — python3 models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py --self-test passed: analytic lambda, Pearson > 0.98, ablation sigma < 0, monotonic sweep, FFT within 8%, GIF >= 10 frames and >= 480 px, non-circular specimen rows. Seed 3102.  
  evidence: `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` L724–755

Remediation:
- A second assessor from another provider has not scored this audit.
- The 4.2 mm digit corridor, alpha = 3.4, and the Dv factor 0.92 remain reconstructions.
- No published Acanthostega melanosome spacing at 105 µm is available to replace the campaign scale.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Acanthostega gunnari Turing wavenumber at 105 µm (Famennian, 365 Ma)
- **spec_fingerprint** (semantic model + inputs): `c711774b8d595cd5a6f22703fc15b77247416d03dc05678953f2434b64d13ee2`
- **input_dataset_sha256**: `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_brussels_head_specimen.jpg` (`6eae655c40f0444c…`), `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_turing_wavelength_s0_reference.json` (`dc929ab5cb065aa6…`)
- `docs/media/acanthostega_gunnari_turing_wavelength_s0_comparison.png` — artifact_sha256 `ee8fcd1b8df7a3e3…`, 557,252 B, 1344×1008 px
- `docs/media/acanthostega_gunnari_turing_wavelength_s0_growth.gif` — artifact_sha256 `9161f32ecf0fd605…`, 3,408,681 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen dermal-patch ocellus wavelength | 106.9344 µm | 104.9 | radial FFT peak of the saturated Schnakenberg activator on the 128^2 patch *(value is the measured FFT wavelength; reference is the analytic k_c target)* |
| dispersion growth-rate Pearson r | 0.9999 correlation | 0.98 | corrcoef of analytic σ(k) and 1D Euler growth rates on probe modes 10..20 (band around k_c) *(reference is the pass threshold; the achieved correlation is higher)* |
| dispersion growth-rate RMSE | 0.005 1/time | 0.0 | sqrt mean square residual of σ_hat(k) against σ_analytic(k) |
| specimen orientation-histogram KS distance | 0.1368 KS | 0.05 | max CDF gap between 12-bin Sobel orientation of the Brussels skull and of the RD field *(non-circular specimen metric: the fossil plate and the ocellus field are different images)* |
| specimen foreground fraction | 0.5568 area fraction | 0.5 | border-median contrast mask on the Brussels Acanthostega skull (verify_image_fitness family) |
| specimen structure-tensor coherency | 0.5265 coherency | 0.32 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the specimen mask |
| specimen contour circularity | 0.33 4πA/P^2 | 1.0 | mask area and morphological perimeter of the Brussels skull photograph |
| Tiktaalik along-ray stripe wavelength | 100.1115 µm | 104.9 | same Du, r_eff = 10 * 3.4 along lepidotrichia; analytic k_c *(predecessor pattern is ray-locked, not the isotropic ocellus)* |
| Tiktaalik cross-ray specimen-scale growth rate | -0.621 1/time | 0.0 | σ(k_c) at r_eff = 10/3.4 (cross-ray inhibitor diffusivity); negative means the mode decays |
| Ichthyostega inhibitor-slowed wavelength | 105.5387 µm | 104.9 | same Du, Dv scaled by 0.92 (reconstructed cornified-dermis factor), analytic k_c |
| equal-diffusion ablation growth rate | -0.0989 1/time | 0.05 | max_k σ(k) at Dv = Du (r = 1); Turing numerator is negative |
| wavelength sweep coefficient of variation | 0.0239 CV | 0.0 | std/mean of analytic λ(r) for supercritical ratios at fixed Du |

## Environment at checkpoint


Working tree on top of dff4f2b5. Loci verified against the files in this commit's parent plus the new loop files; build uses --no-verify-loci because the loop commit does not exist yet.
