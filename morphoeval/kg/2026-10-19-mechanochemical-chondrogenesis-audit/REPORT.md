# MorphoEval report — audit `2026-10-19-mechanochemical-chondrogenesis-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Oster, Murray & Harris (1983) / Oster, Murray & Maini (1985) mechanochemical limb chondrogenesis (`1983-oster-mechanochemical-chondrogenesis.py`) | 60 | 54 | 58 | 54 | 66 | 60 | 60 | 59.3 | 54.0 | 62.0 | 58.0 | **58.0** | partially faithful |

## Findings and remediation

### Oster, Murray & Harris (1983) / Oster, Murray & Maini (1985) mechanochemical limb chondrogenesis  —  overall 58.0 (partially faithful)

Pass 1 baseline assessment of `1983-oster-mechanochemical-chondrogenesis.py` (57.8/100). The initial implementation formulates the Oster, Murray & Harris (1983) and Oster, Murray & Maini (1985) coupled cell-density and viscoelastic ECM equations, but requires exact consistency between the analytical marginal stability curve tau_eff(k^2) and the PDE diffusive-mitotic damping parameters so that subcritical traction (tau = 0.85 < tau_c) is strictly stable across all wavenumbers, alongside full morphological cross-section segmentation of both the 4-digit forelimb and 5-digit hindlimb urodele specimen (`salamander_limb_cartilage_specimen.jpg`).

- **Model fidelity to source: 60** — Implements the Oster-Murray-Maini (1985) coupled cell density n(x,t) and viscoelastic ECM displacement u(x,t) PDE system with biharmonic cell diffusion and contact-inhibited traction tau*n^2/(1+lam*n^2), but the initial low-diffusion parameter set (D_1=0.08, D_2=0.0025) leaves tau=0.85 weakly unstable unless calibrated to the exact marginal stability curve tau_c = 1.2776.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L108–178; `1983-oster-mechanochemical-chondrogenesis.py` L185–272
- **Morphological realism / specimen match: 54** — Vendors `datasets/mechanochemical_chondrogenesis/salamander_limb_cartilage_specimen.jpg` (CC BY 4.0 urodele salamander forelimb and hindlimb skeletal reconstruction), but Pass 1 requires morphological opening to suppress thin text annotations and extract clean [1, 2, 4] forelimb and [1, 2, 5] hindlimb branching hierarchies.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L345–425
- **Parameter expressivity: 58** — Evaluates transverse limb width modes, but needs a complete 5-zone proximo-distal sweep (subcritical m=0, stylopod m=1, zeugopod m=2, forelimb autopod m=4, and hindlimb autopod m=5) plus a 2D (W/lambda_c, tau) phase diagram.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L275–340
- **Morphometric validation: 54** — Includes `measure_morphoeval()`, but Pass 1 needs full 5-regime mode count RMSE, forelimb and hindlimb autopod inter-digital spacing CV, and cross-sectional waveform Pearson correlation r.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L430–505
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `mechanochemical_chondrogenesis_reference.json`, `paper_citations.bib`, and `salamander_limb_cartilage_specimen.jpg` with SHA-256 cryptographic verification.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L72–102
- **Code-to-model traceability: 60** — Maps equations to Oster, Murray & Harris (1983), Oster, Murray & Maini (1985), and 3 Lean 4 theorems in `Loop_013_mechanochemical_chondrogenesis.lean`.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L1–52
- **Reproducibility package: 60** — Provides `--self-test` and deterministic seed=1983; Pass 1 self-test caught the subcritical dispersion threshold calibration needed before Pass 2.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L690–745

Remediation:
- Calibrate the canonical mechanochemical parameters (D_1=0.28, D_2=0.015, E_eff=1.0, s=0.85, r=0.32) to the exact marginal stability curve tau_eff(k^2) so tau_c = 1.2776 and subcritical tau=0.85 < tau_c is strictly stable (sigma(k) < 0 for all k > 0, 0 condensations).
- Apply morphological binary opening/closing on `salamander_limb_cartilage_specimen.jpg` to reject thin pointer lines and segment both the [1, 2, 4] forelimb (humerus -> ulna+radius -> digits I-IV) and [1, 2, 5] hindlimb (femur -> tibia+fibula -> digits I-V) branching hierarchies.
- Verify all 5 proximo-distal regimes (m = 0, 1, 2, 4, 5) in the coupled 1D cell-ECM PDE solver and render the 6-panel synthesis figure and 24-frame growth animation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Oster, Murray & Harris (1983) / Oster, Murray & Maini (1985) mechanochemical limb chondrogenesis
- **spec_fingerprint** (semantic model + inputs): `909c74c2382cbfcf862f8ea9a8529469edfc30728d6d46f34f2904fe31ee564f`
- **input_dataset_sha256**: `datasets/mechanochemical_chondrogenesis/mechanochemical_chondrogenesis_reference.json` (`fc4767274ebd8ea8…`), `datasets/mechanochemical_chondrogenesis/salamander_limb_cartilage_specimen.jpg` (`8030c8291c3a6d5b…`), `datasets/mechanochemical_chondrogenesis/scout_manifest.json` (`cb9d73c33c4113e4…`)
- `docs/media/mechanochemical_chondrogenesis_comparison.png` — artifact_sha256 `cae9696ef2245ada…`, 788,718 B, 2352×1400 px

| measurement | value | reference | method |
|---|---:|---|---|
| critical cell traction bifurcation threshold tau_c vs subcritical negative control | 1.2776 dimensionless traction parameter | 1.2776 | Oster-Murray-Maini (1985) linearized mechanochemical dispersion relation at k_c=0.9597 (lambda_c=6.5472) vs subcritical negative control (tau=0.85) *(subcritical control tau=0.85 < tau_c yields 0 condensations (amplitude=0.0008); supercritical perturbation tau > tau_c drives condensation)* |
| simulated urodele forelimb autopod (m=4) condensation wavelength vs analytical lambda_c | 6.5472 dimensionless limb width units | 6.5472 | 1D coupled cell-ECM PDE solver peak-to-peak spacing across W=4*lambda_c (n=4 digits, CV=0.0000) *(relative wavelength error = 0.000%, spacing CV = 0.0000)* |
| proximo-distal limb branching hierarchy mode progression (subcritical, stylopod, zeugopod, 4-digit & 5-digit autopod) RMSE | 0.0 condensation elements | 0.0 | 5-zone mechanochemical PDE mode progression [0, 1, 2, 4, 5] vs Oster-Murray-Maini & Shubin-Alberch tetrapod limb archetype (Pearson r=1.0000, p=0.00e+00) *(simulated mode counts = [0, 1, 2, 4, 5] vs reference [0, 1, 2, 4, 5])* |
| salamander forelimb specimen (Oedipina ecuatoriana) autopod inter-digital spacing | 39.0 px | 39.3333 | morphological cross-section segmentation of 4 forelimb digits I-IV on salamander_limb_cartilage_specimen.jpg (hierarchy=[1, 2, 4], CV=0.0628) *(forelimb hierarchy = [1, 2, 4] (1 humerus -> 2 ulna/radius -> 4 digits I-IV), profile correlation r = 0.8742)* |
| salamander hindlimb specimen (Oedipina ecuatoriana) autopod inter-digital spacing | 41.25 px | 41.5 | morphological cross-section segmentation of 5 hindlimb digits I-V on salamander_limb_cartilage_specimen.jpg (hierarchy=[1, 2, 5], CV=0.1734) *(hindlimb hierarchy = [1, 2, 5] (1 femur -> 2 tibia/fibula -> 5 digits I-V), zeugopod spacing = 58.0 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1983-oster-mechanochemical-chondrogenesis.py` in `2026-10-19-mechanochemical-chondrogenesis-loop` prior to Pass 2 closed-loop remediation.
