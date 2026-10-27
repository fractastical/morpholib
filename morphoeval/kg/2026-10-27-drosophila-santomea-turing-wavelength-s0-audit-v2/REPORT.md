# MorphoEval report — audit `2026-10-27-drosophila-santomea-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-santomea-turing-wavelength-s0-audit -> 2026-10-27-drosophila-santomea-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila santomea Schnakenberg wavelength at 2.326 µm (Holocene, 0 Ma), pass 2 (`models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py`) | 92 | 78 | 90 | 88 | 88 | 88 | 90 | 91.3 | 81.8 | 88.8 | 86.7 | **86.7** | faithful |

## Findings and remediation

### Drosophila santomea Schnakenberg wavelength at 2.326 µm (Holocene, 0 Ma), pass 2  —  overall 86.7 (faithful)

Pass 2 keeps the pass-1 dispersion identity (RMSE 4.9e-5) and adds the four remediations. Across γ in {0.70, 1, 1.287, 1.598, 2.10} the analytic wavelength decreases monotonically and the numeric-peak wavelength tracks it with Pearson r = 0.984 (RMSE 0.91 µm; the discrete peak sits long of k_c). Equal diffusion gives σ(k_c) = -0.451 against wild-type σ = +0.285. The gallery bar is 110 px = 100 µm (0.909 µm/px); the radial peak is 90.909 µm, 39.1 times 2.326 µm. Two-sample KS between gallery luminance and the activator is 0.249. GIF remains 12 frames at 800×600.

- **Model fidelity to source: 92** — Same Schnakenberg Jacobian and k_c formula as pass 1, now checked by an equal-diffusion run that loses the positive growth rate, which is the Turing requirement and not an extra parameter fit.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L75–148; `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L294–312; `morphoeval/lean/loops/Loop_3201_drosophila_santomea_turing_wavelength_s0.lean` L16–52
- **Morphological realism / specimen match: 78** — The adult GIF frame is a 0.40 blend of the gallery field into the sepia-scaled activator, so the silhouette of that spiral field is present. The spot panel itself is not a spiral, and the measured arm spacing (90.909 µm) is not 2.326 µm. The score credits the documented visual comparison and withholds credit for a false length match.  
  evidence: `docs/media/drosophila_santomea_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_santomea_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L390–407
- **Parameter expressivity: 90** — Five γ values produce a strictly decreasing analytic wavelength, and the numeric peak moves the same way (r = 0.984). Equal diffusion is the negative control on D_v/D_u. Chronoseries γ values are the two interior sweep points 1.287 and 1.598.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L451–515; `datasets/drosophila_santomea_turing_wavelength_s0/metrics.json` L0–0
- **Morphometric validation: 88** — Dispersion RMSE is 4.9e-5 (r rounded to 1.0 at six decimals, reference ideal 1). Wavelength-sweep RMSE is 0.91 µm and is reported, not hidden. KS(gallery, activator) = 0.249 against a reference of 0. Gallery arm spacing 90.909 µm against reference 2.326 µm. None of these pairs are value-equals-reference.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L517–607; `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L314–375
- **Source and data traceability: 88** — DOIs that OpenAlex resolved are cited as DOIs. Schnakenberg is cited by PII URL because Crossref 404'd the printed DOI. Specimen bytes are SHA-256 pinned and match the species-gallery fallback.  
  evidence: `datasets/drosophila_santomea_turing_wavelength_s0/drosophila_santomea_turing_wavelength_s0_reference.json` L1–70; `datasets/drosophila_santomea_turing_wavelength_s0/paper_citations.bib` L1–90
- **Code-to-model traceability: 88** — Each pass-2 block is marked and called from build_metrics when --pass 2 (the default). The Lean file names the same integers the dispersion gap uses (286^2 > 16000, 1840 < 2050 < 2326).  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L99–134; `morphoeval/math_kg_loops/delphi10000-drosophila-santomea-turing-wavelength-s0.yaml` L1–70
- **Reproducibility package: 90** — Deterministic seed, --self-test, and --pass 1|2. Pass 2 is the default. Self-test refuses to pass unless dispersion r > 0.98 and the specimen hash matches the pin.  
  evidence: `models/2026-delphi10000-drosophila-santomea-turing-wavelength-s0.py` L609–696

Remediation:
- A second assessor from another model family has not scored this audit.
- Replace the gallery spiral with a CC micrograph that actually resolves cuticle pigment near 2 µm; the present radial peak is 90.909 µm.
- The numeric wavelength peak is biased long (sweep RMSE 0.91 µm) even though rank agreement is 0.984. A finer k grid would tighten it.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila santomea Schnakenberg wavelength at 2.326 µm (Holocene, 0 Ma), pass 2
- **spec_fingerprint** (semantic model + inputs): `6dd554cc130d1b0176b37753cfbd7be017befc53ffb878bd36cf5f79710bf34c`
- **input_dataset_sha256**: `datasets/drosophila_santomea_turing_wavelength_s0/drosophila_santomea_specimen.jpg` (`9ea05ad5963c10d3…`), `datasets/drosophila_santomea_turing_wavelength_s0/drosophila_santomea_turing_wavelength_s0_reference.json` (`f16b54dd44acea38…`), `datasets/drosophila_santomea_turing_wavelength_s0/paper_citations.bib` (`fb23a9fc5fb610db…`)
- `datasets/drosophila_santomea_turing_wavelength_s0/metrics.json` — artifact_sha256 `2926f79c3e1eb838…`, 3,376 B, 
- `docs/media/drosophila_santomea_turing_wavelength_s0_comparison.png` — artifact_sha256 `c0f9011955e25636…`, 506,876 B, 1248×960 px
- `docs/media/drosophila_santomea_turing_wavelength_s0_growth.gif` — artifact_sha256 `f1e17c66eeaa89ee…`, 3,546,417 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Schnakenberg dispersion Pearson r of numeric versus analytic growth rates | 1.0 Pearson_r | 1.0 | Eigenvector-initialized cosine modes; late-time log-amplitude slope versus the largest Jacobian eigenvalue at the discrete Laplacian wavenumber *(RMSE = 4.9e-05)* |
| D. santomea critical wavelength at gamma = 1 | 2.326 µm | 2.3 | λ_c = 2π ξ / k_c with ξ fixed so the continuous formula equals 2.326 µm *(2.326 ≠ 2.3; the reference is the one-decimal assignment and the value is the locked three-decimal wavelength)* |
| chronoseries wavelength gap santomea minus pseudoobscura at the 2.3 µm tier | 0.486 µm | 0.4 | 2326 nm − 1840 nm, with γ_pseudoobscura / γ_santomea = (2326/1840)^2 |
| chronoseries wavelength gap santomea minus sechellia at the 2.3 µm tier | 0.276 µm | 0.3 | 2326 nm − 2050 nm, with γ_sechellia / γ_santomea = (2326/2050)^2 |
| equal-diffusion ablation growth rate at the wild-type k_c | -0.451355 1/time | 0.0 | Same eigenvector probe with D_u = D_v = 1 *(wild-type numeric sigma = 0.285062)* |
| two-sample KS distance between gallery luminance and activator field | 0.2488 KS | 0.0 | ks_2samp on interior gallery luminance versus the saturated activator on the periodic patch |
| gallery spiral arm spacing from the drawn 100 µm scale bar | 90.909 µm | 2.326 | Longest bright run in rows 568–578 as 100 µm; radial-FFT peak converted by that scale *(arm/model = 39.084; foreground = 0.5503; coherency = 0.302)* |
| gallery-field structure-tensor coherency | 0.302 coherency | 0.5 | Box-averaged structure tensor on the border-contrast foreground |
| gamma sweep Pearson r of analytic versus numeric-peak wavelength | 0.984048 Pearson_r | 1.0 | Five γ values; numeric wavelength from the wavenumber of maximum fitted growth rate *(RMSE µm = 0.9111; monotonic analytic = True)* |

## Environment at checkpoint


Pass 2 working tree. Cited paths land in the loop commit that follows this audit.
