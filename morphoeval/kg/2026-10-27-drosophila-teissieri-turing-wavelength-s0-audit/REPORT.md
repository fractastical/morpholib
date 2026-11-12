# MorphoEval report — audit `2026-10-27-drosophila-teissieri-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila teissieri Turing wavelength at 1.7 µm (detail scale s0, Holocene 0 Ma) (`models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py`) | 90 | 82 | 88 | 84 | 86 | 86 | 90 | 89.3 | 82.8 | 87.6 | 86.2 | **86.2** | faithful |

## Findings and remediation

### Drosophila teissieri Turing wavelength at 1.7 µm (detail scale s0, Holocene 0 Ma)  —  overall 86.2 (faithful)

Schnakenberg kinetics on a periodic 20.64 µm patch with diffusivities solved so Turing's formula gives λ_c = 1.72 µm. Dispersion Pearson r = 0.9935 (RMSE 0.030 1/s) on the real-eigenvalue band. Equal-diffusion and autocatalysis-ablation controls both decay. A six-point diffusivity sweep is monotonic in the analytical wavelength (numerical Pearson r = 0.9995). The saturated sub-bin wavelength is 1.903 µm, 10.6% longer than the formula wavelength: the growth-rate peak is not the determinant minimizer, and that comparison is not yet a scored measurement (R1). Specimen CV similarity of the 12-frame 800×600 GIF is 88.4. The KS statistic (0.450) compares the raw activator to the micrograph, not the rendered frame the fitness test sees (R2).

- **Model fidelity to source: 90** — The script implements Schnakenberg kinetics and solves D_u from Turing's wavenumber formula so λ_c = 1.72 µm. Homogeneous trace is negative, det is positive, σ(0) < 0 and σ(k_c) > 0. The linearised PDE recovers continuum σ(k) at Pearson r = 0.9935. The formula wavelength and the growth-rate peak are not the same number; the code pins the formula, which is the cited law, and the offset is reported rather than tuned away.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L59–98; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L111–191
- **Morphological realism / specimen match: 82** — The 800×600 GIF keeps the vendored micrograph as the field and grows a magnified 20.64 µm activator inset. Canonical CV similarity against that micrograph is 88.4 (12 frames, 600 px). The gallery frame is a 100 µm-bar micrograph, not a taxonomic photograph from Commons (none was found), and the inset is a magnification because 1.72 µm is ~1.9 px on the 100 µm bar.  
  evidence: `docs/media/drosophila_teissieri_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_teissieri_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L401–431
- **Parameter expressivity: 88** — Holding D_u fixed and raising D_v/D_u from 12.2 to 32 lengthens the analytical wavelength monotonically. The 1D numerical wavelength tracks it at Pearson r = 0.9995. Equal diffusion and deletion of the u^2 v term both kill the pattern, so the two signs the law cares about are exercised, not only the forward parameter.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L270–320; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L197–268
- **Morphometric validation: 84** — Dispersion RMSE is 0.030 1/s and the determinant minimizer sits on the formula k_c (relative error under 5% by the self-test, and in practice far tighter). Otsu area, circularity, aspect and coherency are measured on the specimen and compared with the activator mask; none is a copy of its reference. The KS statistic is 0.450 against a 0.05 reference because it contrasts the raw activator with the micrograph rather than the rendered frame (R2). Saturated wavelength 1.903 µm versus formula 1.72 µm is recorded, not hidden (R1).  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L593–650; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L322–389
- **Source and data traceability: 86** — Turing, Schnakenberg, Kondo & Miura, Crampin & Maini, and Tamura DOIs were checked on Crossref. GBIF 5074619 is the taxon record. The specimen SHA-256 is pinned. Commons had no CC bitmap; the gallery fallback is stated. DOI 10.1093/molbev/msx210 is explicitly rejected because Crossref resolves it to a yeast paper.  
  evidence: `datasets/drosophila_teissieri_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/drosophila_teissieri_turing_wavelength_s0/drosophila_teissieri_turing_wavelength_s0_reference.json` L1–40; `datasets/drosophila_teissieri_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 86** — Jacobian, wavenumber solve, dispersion, sweep and controls are separate functions whose docstrings name the Turing or Schnakenberg relation they implement. The module docstring states the periodic boundary condition and the chronoseries contrast.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3209_drosophila_teissieri_turing_wavelength_s0.lean` L14–62
- **Reproducibility package: 90** — --self-test checks the Turing inequalities, Pearson r > 0.98, both negative controls, the longer-domain mode count, GIF frame count and size, and non-circular specimen metrics. RNG seed 3209 and SHA-256 pins are in the reference JSON. One self-test run completed in about 7 s.  
  evidence: `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L710–756; `models/2026-delphi10000-drosophila-teissieri-turing-wavelength-s0.py` L47–57

Remediation:
- {'id': 'R1', 'item': 'Saturated sub-bin wavelength is 1.903 µm versus formula λ_c = 1.72 µm. Score the growth-rate peak explicitly and report how close the selected mode is to that peak, not only to the determinant minimizer.'}
- {'id': 'R2', 'item': 'KS (0.450) compares the raw activator histogram to the micrograph. Add a KS between the rendered last GIF frame and the specimen, which is the image the fitness test scores.'}
- {'id': 'R3', 'item': 'Single builder-assessor. Gallery micrograph is a fallback (no Commons bitmap), so specimen-match cannot rise to a verified species photograph.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=3.0]** `morphometric_validation` → `morphometric-validation`: Compare saturated wavelength to the σ(k) peak and KS the rendered frame against the specimen.

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
| formula critical wavelength | 1.72 µm | 1.72 | lambda_c = 2 pi / k_c with k_c from the wavenumber formula |
| determinant-minimizer wavelength | 1.72 µm | 1.72 | numerical argmin of det(gamma J - k^2 D) on a k grid |
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


Pass 1 working tree. New slot files are not in this commit. publisher_orcid is the dct:creator already used on morphoeval/kg/2026-10-19-turing-audit. No ~/.nanopub/profile.yml, so these nanopubs are unsigned.
