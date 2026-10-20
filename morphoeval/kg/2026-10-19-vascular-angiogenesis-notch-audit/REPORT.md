# MorphoEval report — audit `2026-10-19-vascular-angiogenesis-notch-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/60

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature & Jakobsson et al. (2010) Nat. Cell Biol. VEGF-VEGFR2-Dll4-Notch1 tip/stalk salt-and-pepper lateral inhibition and angiogenic sprouting (`2021-ibrahim-vascular-angiogenesis-notch.py`) | 54 | 44 | 49 | 51 | 68 | 55 | 53 | 52.3 | 46.6 | 58.5 | 51.9 | **51.9** | partially faithful |

## Findings and remediation

### Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature & Jakobsson et al. (2010) Nat. Cell Biol. VEGF-VEGFR2-Dll4-Notch1 tip/stalk salt-and-pepper lateral inhibition and angiogenic sprouting  —  overall 51.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2021-ibrahim-vascular-angiogenesis-notch.py` (weighted score: 53.0/100). The Pass 1 baseline integrates only an isolated two-cell Dll4-Notch1 ODE at two trans-Notch gains (k_t = 1.0 and 0.0) without solving the multicellular M=16 endothelial vessel wall coupled VEGFR2-Dll4-Notch1 + 2D filopodial angiogenic sprout elongation network across the 5 canonical biological regimes (Physiological VEGF gradient salt-and-pepper, High uniform VEGF synchronous oscillations, DAPT Notch blockade all-tip hypersprouting, Dll4+/- haploinsufficiency, and Vegfr2+/- mosaic chimeric tip competition Jakobsson et al. 2010), omits VEGFR1/sFlt1 decoy sequestration and the 8-point Notch trans-gain sweep, and does not segment the vendored Homo sapiens HUVEC angiogenic sprouting vascular plexus micrograph.

- **Model fidelity to source: 54** — Integrates a minimal two-cell Dll4-Notch1 ODE (L26-66), but omits the M=16 multicellular endothelial vessel wall network, 2D filopodial angiogenic sprout elongation and anastomosis/syncytial fusion, VEGFR1 (sFlt1) decoy sequestration, and Brillouin k=pi Fourier mode decomposition.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L26–66
- **Morphological realism / specimen match: 44** — Vendors `datasets/vascular_angiogenesis_notch/huvec_angiogenic_sprouting_plexus_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract vascular plexus area density, branch junctions, or sprouting filopodial tip counts.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L94–138
- **Parameter expressivity: 49** — Evaluates only two scalar k_t values (1.0 and 0.0) in a 2-cell ODE without simulating the 5 canonical angiogenic regimes (High VEGF oscillations, Dll4+/- haploinsufficiency, Vegfr2+/- mosaic competition) or an 8-point Notch gain sweep.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L69–91
- **Morphometric validation: 51** — Exposes only 2 scalar measurements in `measure_morphoeval()` without Brillouin k=pi period-2 alternation index, temporal oscillation CV, chimeric WT tip occupancy, Pearson correlation p-values, or specimen CV morphometrics.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L69–91
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `vascular_angiogenesis_notch_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/vascular_angiogenesis_notch/vascular_angiogenesis_notch_reference.json` L1–145
- **Code-to-model traceability: 55** — Module docstring cites Bentley et al. (2008) and Hellström et al. (2007), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L1–9
- **Reproducibility package: 53** — Includes a minimal 2-assertion `--self-test` flag, but lacks SHA-256 dataset integrity checks, Brillouin mode eigenvalue checks, Flt1 rescue checks, and specimen segmentation verification.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L141–146

Remediation:
- Implement the coupled multicellular M=16 Bentley-Gerhardt-Bates VEGFR2-Dll4-Notch1 lateral-inhibition ODE network + 2D filopodial angiogenic sprout elongation solver with soluble VEGFR1 (sFlt1) decoy sequestration.
- Simulate all 5 canonical angiogenic regimes (Physiological VEGF gradient salt-and-pepper, High uniform VEGF synchronous oscillations, DAPT Notch blockade hypersprouting, Dll4+/- haploinsufficiency, and Vegfr2+/- mosaic chimeric tip competition Jakobsson et al. 2010) plus an 8-point Notch trans-activation gain sweep.
- Segment the vendored Homo sapiens HUVEC angiogenic sprouting vascular network micrograph (`huvec_angiogenic_sprouting_plexus_specimen.jpg`, CC BY 4.0) to extract vascular plexus area density, branch junctions, and sprouting tip count, backed by runtime SHA-256 verification and an 8-suite `--self-test`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature & Jakobsson et al. (2010) Nat. Cell Biol. VEGF-VEGFR2-Dll4-Notch1 tip/stalk salt-and-pepper lateral inhibition and angiogenic sprouting
- **spec_fingerprint** (semantic model + inputs): `75ad27375139b901d6c87702d5f02c739a8e698db79579ceae73bdab1d021c1b`
- **input_dataset_sha256**: `datasets/vascular_angiogenesis_notch/huvec_angiogenic_sprouting_plexus_specimen.jpg` (`9bf4f3e2692bd2ac…`), `datasets/vascular_angiogenesis_notch/paper_citations.bib` (`a96e44f99de8a756…`), `datasets/vascular_angiogenesis_notch/scout_manifest.json` (`2b97b38c418e4c44…`), `datasets/vascular_angiogenesis_notch/vascular_angiogenesis_notch_reference.json` (`8821743569ceaf2e…`)
- `docs/media/vascular_angiogenesis_notch_comparison.png` — artifact_sha256 `752052d83f74a51a…`, 81,701 B, 1365×572 px

| measurement | value | reference | method |
|---|---:|---|---|
| Physiological salt-and-pepper (k_t = 1.0) tip-cell fraction | 0.5 tip cell fraction | 0.5 | Two-cell Dll4-Notch1 ODE at k_t=1.0 *(two-cell tip/stalk split with |D1 - D2| = 0.5162)* |
| DAPT Notch blockade (k_t = 0.0) tip-cell fraction | 1.0 tip cell fraction | 1.0 | Two-cell Dll4-Notch1 ODE at k_t=0.0 *(both cells become tip cells with |D1 - D2| = 0.0000)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2021-ibrahim-vascular-angiogenesis-notch.py` prior to Step 12 closed-loop remediation.
