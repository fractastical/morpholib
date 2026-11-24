# MorphoEval report — audit `2026-10-27-drosophila-teissieri-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-teissieri-turing-wavelength-s0-audit -> 2026-10-27-drosophila-teissieri-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila teissieri Turing wavelength at 1.7 µm (detail scale s0, Holocene 0 Ma) (`models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py`) | 91 | 84 | 88 | 90 | 86 | 87 | 90 | 90.0 | 86.2 | 87.9 | 87.9 | **87.9** | faithful |

## Findings and remediation

### Drosophila teissieri Turing wavelength at 1.7 µm (detail scale s0, Holocene 0 Ma)  —  overall 87.9 (faithful)

Pass 2 keeps the same Schnakenberg patch (λ_c formula = 1.72 µm, dispersion Pearson r = 0.9935, RMSE = 0.030 1/s) and adds the two Pass-1 measurements that were missing. The growth-rate peak is at 1.985 µm; the saturated sub-bin wavelength is 1.903 µm, 4.2% shorter than that peak and 10.6% longer than the determinant-minimising formula wavelength. Those are reported as separate numbers. The rendered-frame KS against the specimen is 0.147 (reference 0.05); the raw-activator KS is 0.450. Canonical CV similarity is 88.4 on the 12-frame 800×600 GIF. R3 stays open: one assessor, and no Commons bitmap exists for this species.

- **Model fidelity to source: 91** — Schnakenberg kinetics and the Turing wavenumber formula are implemented directly, with D_u solved so λ_c = 1.72 µm. Trace < 0, det > 0, σ(0) < 0, σ(k_c) > 0. Linearised growth matches continuum σ(k) at Pearson r = 0.9935. Pass 2 separates that formula wavelength from argmax σ(k) = 1.985 µm instead of forcing them to agree.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L59–98; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L433–445
- **Morphological realism / specimen match: 84** — The GIF is the vendored micrograph plus a magnified 20.64 µm inset that grows across 12 frames at 800×600. Canonical CV similarity is 88.4. The rendered-frame luminance KS is 0.147 against a 0.05 reference, so the inset changes the histogram; the frame is not a copy of the specimen and not a pure stripe chart. Commons still has no CC bitmap of this species (R3).  
  evidence: `docs/media/drosophila_teissieri_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L447–483; `docs/media/drosophila_teissieri_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — Raising D_v/D_u from 12.2 to 32 with D_u fixed lengthens analytical wavelength monotonically, and the numerical wavelength follows at Pearson r = 0.9995. Equal diffusion and removal of u^2 v both collapse spatial standard deviation.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L270–320; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L197–268
- **Morphometric validation: 90** — Dispersion RMSE is 0.030 1/s. The saturated wavelength (1.903 µm) is 4.2% from the growth-rate peak and is not substituted for the 1.72 µm formula value. Specimen Otsu area, circularity, aspect and coherency differ from the activator mask. Rendered-frame KS and activator-vs-micrograph KS are both reported against a 0.05 reference they do not equal. An 18-wavelength domain selects a higher mode (17) than the 12-wavelength patch (11).  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L593–708; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L322–389
- **Source and data traceability: 86** — Crossref-checked DOIs for Turing, Schnakenberg, Kondo & Miura, Crampin & Maini, and Tamura. GBIF 5074619. Specimen SHA-256 pinned. Gallery fallback and the rejected yeast DOI are written in the README and the reference JSON.  
  evidence: `datasets/drosophila_teissieri_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/drosophila_teissieri_turing_wavelength_s0/README.md` L1–33
- **Code-to-model traceability: 87** — Each modelling step is a function whose docstring names the Turing or Schnakenberg relation. Pass 2 adds sigma_peak_wavelength and rendered_frame_ks beside the original dispersion and renderer. Lean theorems pin the trace sign, the drive sign, the 12-wavelength domain, and the predecessor and successor inequalities.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3209_drosophila_teissieri_turing_wavelength_s0.lean` L17–62
- **Reproducibility package: 90** — --self-test checks stability signs, Pearson r > 0.98, both controls, the longer domain, GIF length and size, and non-equal specimen metrics. Seed 3209. SHA-256 pins. Pass 2 self-test passed.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L710–756; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L47–57

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass. Growth-rate peak is 1.985 µm; saturated wavelength is 1.903 µm (relative gap 0.042). Formula λ_c remains 1.72 µm and is not overwritten.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in this pass. rendered_frame_ks compares the composite frame to the specimen. Activator-vs-micrograph KS is retained as a separate, larger distance.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Still open. One builder-assessor, and Wikimedia Commons had no CC bitmap of Drosophila teissieri on 2026-10-27, so the gallery micrograph remains the specimen.', 'status': 'open'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.5]** `independent_assessor` → `specimen-match`: A second assessor and a CC-licensed taxonomic photograph, if one is later released, would be the only honest way to raise specimen-match further.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila teissieri Turing wavelength at 1.7 µm (detail scale s0, Holocene 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `eae547f9e47b2ea29de70862c97589eafea17d3c75744f5144aa5ef1d4f3d6d0`
- `docs/media/drosophila_teissieri_turing_wavelength_s0_comparison.png` — artifact_sha256 `3afee39d65af76a2…`, 475,296 B, 1512×984 px
- `docs/media/drosophila_teissieri_turing_wavelength_s0_growth.gif` — artifact_sha256 `b26546f4b1e7b7ae…`, 4,963,910 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (real-eigenvalue band) | 0.99349 correlation | 1.0 | linearised 1D PDE growth vs continuum σ(k), modes with disc>=0 |
| dispersion RMSE | 0.03008 1/s | 0.0 | RMSE of measured vs continuum growth rates on the same band |
| formula critical wavelength | 1.72 µm | 97.6 | lambda_c = 2 pi / k_c; reference is the takahashii cell-tier atlas stripe, not a copy of this value |
| determinant-minimizer relative error | 1.2156786782482774e-16 fraction | 0.0 | abs(k_argmin h(k) - k_c) / k_c |
| growth-rate peak wavelength | 1.98538 µm | 1.72 | argmax_k Re sigma(k) on the continuum dispersion; differs from determinant k_c when trace depends on k |
| saturated wavelength versus growth-rate peak | 0.04152 fraction | 0.0 | relative gap between the nonlinear selected wavelength and argmax sigma(k) |
| KS statistic rendered frame vs specimen image | 0.147 KS distance | 0.05 | ks_2samp of the last composite frame against the 800x600 specimen |
| saturated stripe wavelength (sub-bin) | 1.90295 µm | 1.72 | parabolic refinement of the dominant Fourier cycle on the final activator |
| specimen image foreground area fraction | 0.84527 fraction | 0.40213 | Otsu mask on the vendored micrograph |
| specimen image contour circularity | 0.01278 4πA/P^2 | 0.12561 | largest Otsu component, perimeter from binary erosion |
| specimen image aspect ratio | 1.33333 width/height | 0.04167 | bounding box of the largest Otsu component |
| specimen image structure-tensor coherency | 0.09192 index | 0.17708 | mean structure-tensor coherency of the specimen mask |
| KS statistic specimen luminance vs activator | 0.4495 KS distance | 0.05 | scipy.stats.ks_2samp on 4000 pixels, both scaled to [0,1] |
| equal-diffusion control final/initial std | 0.19257 ratio | 1.0 | Du=Dv integration; ratio < 1 means the pattern decays |
| autocatalysis-ablation final/initial std | 2e-05 ratio | 1.0 | u^2 v exchange removed; spatial std must fall |
| diffusivity-sweep Pearson r | 0.99949 correlation | 1.0 | analytical argmax σ(k) vs 1D numerical dominant wavelength across six ratios |
| predecessor wavenumber ratio takahashii/teissieri | 56.7442 ratio | 50.0 | 97.6 µm atlas stripe / 1.72 µm detail wavelength |
| successor floor over teissieri detail | 1.5814 ratio | 1.0 | 2.72 µm virilis atlas floor / 1.72 µm |
| homogeneous trace | -0.3316 1/s (reaction scale) | 0.0 | f_u + g_v at the Schnakenberg steady state |
| 18-wavelength domain dominant mode | 17 cycles | 11 | same PDE on an 18 λ_c patch versus the 12 λ_c patch |

## Environment at checkpoint


Pass 2 working tree. publisher_orcid matches morphoeval/kg/2026-10-19-turing-audit. Unsigned: no ~/.nanopub/profile.yml.
