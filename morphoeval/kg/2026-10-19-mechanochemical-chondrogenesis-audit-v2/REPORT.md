# MorphoEval report — audit `2026-10-19-mechanochemical-chondrogenesis-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-mechanochemical-chondrogenesis-audit -> 2026-10-19-mechanochemical-chondrogenesis-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Oster, Murray & Harris (1983) / Oster, Murray & Maini (1985) mechanochemical limb chondrogenesis (`1983-oster-mechanochemical-chondrogenesis.py`) | 93 | 88 | 90 | 90 | 93 | 91 | 90 | 92.0 | 88.8 | 91.3 | 90.5 | **90.5** | faithful |

## Findings and remediation

### Oster, Murray & Harris (1983) / Oster, Murray & Maini (1985) mechanochemical limb chondrogenesis  —  overall 90.5 (faithful)

Pass 2 post-remediation evaluation of `1983-oster-mechanochemical-chondrogenesis.py` (90.7/100, up from 57.8/100 in Pass 1; +32.9 overall). The remediated implementation integrates the Oster, Murray & Harris (1983) and Oster, Murray & Maini (1985) coupled mesenchymal cell-density and viscoelastic ECM PDE system with biharmonic cell flux regularization and contact-inhibited traction sigma_cell(n) = tau*n^2/(1+lam*n^2), verifying the critical traction bifurcation threshold tau_c = 1.2776 (subcritical tau = 0.85 -> 0 condensations) and exact proximo-distal mode progression [0, 1, 2, 4, 5] (RMSE = 0.0, Pearson r = 1.0000), paired with automated segmentation of the urodele salamander forelimb ([1, 2, 4]: 1 humerus -> 2 ulna/radius -> 4 digits I-IV, spacing 39.00 px, r = 0.8744) and hindlimb ([1, 2, 5]: 1 femur -> 2 tibia/fibula -> 5 digits I-V, spacing 41.50 px) on `salamander_limb_cartilage_specimen.jpg`, backed by 3 machine-checked Lean 4 theorems in `Loop_013_mechanochemical_chondrogenesis.lean`.

- **Model fidelity to source: 93** — Faithfully implements the Oster, Murray & Harris (1983) and Oster, Murray & Maini (1985) coupled cell conservation PDE (with biharmonic diffusion -D_2*nabla^4 n and passive ECM advection -div(n*v_ecm)) and viscoelastic Kelvin-Voigt ECM force balance with contact-inhibited cell traction tau*n^2/(1+lam*n^2) (L108-278), reproducing the exact critical traction bifurcation threshold tau_c = 1.2776 at k_c = 0.9602 (lambda_c = 6.5435) and verified by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_013_mechanochemical_chondrogenesis.lean`.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L108–182; `1983-oster-mechanochemical-chondrogenesis.py` L185–278; `morphoeval/lean/loops/Loop_013_mechanochemical_chondrogenesis.lean` L22–97
- **Morphological realism / specimen match: 88** — Vendors and SHA-256-verifies the CC BY 4.0 urodele salamander (*Oedipina ecuatoriana*, PeerJ 9934 Fig. 13) forelimb and hindlimb skeletal reconstruction (`datasets/mechanochemical_chondrogenesis/salamander_limb_cartilage_specimen.jpg`) and applies morphological opening/closing (`segment_salamander_limb_specimen`, L348-436) to extract the exact [1, 2, 4] forelimb hierarchy (humerus -> ulna+radius -> 4 digits I-IV, spacing 39.00 px, CV = 0.0628, cross-section fit r = 0.8744) and [1, 2, 5] hindlimb hierarchy (femur -> tibia+fibula -> 5 digits I-V, spacing 41.50 px, zeugopod spacing 55.0 px).  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L348–436; `docs/media/mechanochemical_chondrogenesis_comparison.png` L0–0
- **Parameter expressivity: 90** — Evaluates 5 proximo-distal mechanochemical regimes across domain width W/lambda_c and traction tau (subcritical tau=0.85 -> m=0, stylopod W=lambda_c -> m=1, zeugopod W=2*lambda_c -> m=2, forelimb autopod W=4*lambda_c -> m=4, hindlimb autopod W=5*lambda_c -> m=5), maps the 2D (W/lambda_c, tau) phase diagram (Panel D), simulates the 2D limb paddle condensation and ECM displacement vector field (Panel C), and exposes CLI flags (`--tau`, `--mode`, `--seed`).  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L280–345; `1983-oster-mechanochemical-chondrogenesis.py` L580–645
- **Morphometric validation: 90** — Extracts 5 reference-anchored quantitative morphometric measurements via `measure_morphoeval()` (L440-515): critical traction threshold tau_c = 1.2776 (subcritical control = 0 condensations), simulated 4-digit autopod wavelength vs analytical lambda_c = 6.5469 (CV = 0.0000), 5-zone limb branching mode progression [0, 1, 2, 4, 5] RMSE = 0.0000 (Pearson r = 1.0000), salamander forelimb autopod spacing 39.00 px (hierarchy [1, 2, 4], r = 0.8744), and salamander hindlimb autopod spacing 41.50 px (hierarchy [1, 2, 5]).  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L440–515
- **Source and data traceability: 93** — Vendors `datasets/mechanochemical_chondrogenesis/` with `scout_manifest.json`, `mechanochemical_chondrogenesis_reference.json`, `salamander_limb_cartilage_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1242/dev.78.1.83`, `10.1242/dev.89.1.93`, `10.1093/imammb/1.1.51`, `10.1007/978-1-4615-6983-1_6`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L72-98).  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L72–98; `datasets/mechanochemical_chondrogenesis/mechanochemical_chondrogenesis_reference.json` L1–145
- **Code-to-model traceability: 91** — Every computational section and docstring maps directly to numbered equations in Oster, Murray & Harris (1983) Sections 2-4, Oster, Murray & Maini (1985) Sections 2-3 and Figs. 4-9, Shubin & Alberch (1986), and the 3 Lean 4 theorems in `Loop_013_mechanochemical_chondrogenesis.lean` and `morphoeval/math_kg_loops/oster-mechanochemical-chondrogenesis.yaml`.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L1–52; `morphoeval/math_kg_loops/oster-mechanochemical-chondrogenesis.yaml` L1–102
- **Reproducibility package: 90** — Deterministic RNG (`seed=1983`), standalone `--self-test` suite verifying SHA-256 dataset integrity, subcritical stability (sigma < 0 at tau=0.85, 0 condensations), contact-inhibited traction bound, bitwise two-run PDE determinism (`atol=0.0`), exact [0, 1, 2, 4, 5] mode progression, and specimen [1, 2, 4] / [1, 2, 5] hierarchy segmentation (`run_self_test`, L695-748), plus rendered PNG and GIF outputs in `docs/media/`.  
  evidence: `1983-oster-mechanochemical-chondrogenesis.py` L695–748; `docs/media/mechanochemical_chondrogenesis_growth.gif` L0–0

Remediation:
- Couple the 2D viscoelastic ECM displacement solver to a moving free-boundary apical ectodermal ridge (AER) outgrowth domain.
- Hybridize mechanochemical cell traction with Bmp-Sox9-Wnt reaction-diffusion Turing kinetics (Raspopovic et al. 2014, Slot 017) to test mechanochemical-biochemical synergy.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Oster, Murray & Harris (1983) / Oster, Murray & Maini (1985) mechanochemical limb chondrogenesis
- **spec_fingerprint** (semantic model + inputs): `909c74c2382cbfcf862f8ea9a8529469edfc30728d6d46f34f2904fe31ee564f`
- **input_dataset_sha256**: `datasets/mechanochemical_chondrogenesis/mechanochemical_chondrogenesis_reference.json` (`fc4767274ebd8ea8…`), `datasets/mechanochemical_chondrogenesis/salamander_limb_cartilage_specimen.jpg` (`8030c8291c3a6d5b…`), `datasets/mechanochemical_chondrogenesis/scout_manifest.json` (`cb9d73c33c4113e4…`)
- `docs/media/mechanochemical_chondrogenesis_comparison.png` — artifact_sha256 `cae9696ef2245ada…`, 788,718 B, 2352×1400 px
- `docs/media/mechanochemical_chondrogenesis_growth.gif` — artifact_sha256 `bdf2d29aaa10d3f9…`, 813,332 B, 1019×480 px, 24 frames

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

Pass 2 (v2) post-remediation evaluation of `1983-oster-mechanochemical-chondrogenesis.py` after completing the 12-step closed remediation loop (`2026-10-19-mechanochemical-chondrogenesis-loop`).
