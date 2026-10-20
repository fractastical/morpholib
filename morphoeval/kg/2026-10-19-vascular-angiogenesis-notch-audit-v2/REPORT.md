# MorphoEval report — audit `2026-10-19-vascular-angiogenesis-notch-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/60 · supersession chain `2026-10-19-vascular-angiogenesis-notch-audit -> 2026-10-19-vascular-angiogenesis-notch-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature & Jakobsson et al. (2010) Nat. Cell Biol. VEGF-VEGFR2-Dll4-Notch1 tip/stalk salt-and-pepper lateral inhibition and angiogenic sprouting (`2021-ibrahim-vascular-angiogenesis-notch.py`) | 95 | 93 | 93 | 94 | 95 | 93 | 93 | 94.3 | 93.4 | 93.7 | 93.8 | **93.8** | faithful |

## Findings and remediation

### Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature & Jakobsson et al. (2010) Nat. Cell Biol. VEGF-VEGFR2-Dll4-Notch1 tip/stalk salt-and-pepper lateral inhibition and angiogenic sprouting  —  overall 93.8 (faithful)

Pass 2 (v2) evaluation of `2021-ibrahim-vascular-angiogenesis-notch.py` (overall weighted score: 93.8/100, up from 53.0/100 in v1). The remediated script solves the coupled M = 16 multicellular Bentley-Gerhardt-Bates VEGFR2-Dll4-Notch1 ODE network and 2D filopodial angiogenic sprout elongation across all 5 canonical regimes — Physiological VEGF salt-and-pepper (tip fraction = 0.5000, Brillouin I_pi = 0.9472), High uniform VEGF oscillations (temporal CV = 0.3808, rescued by soluble Flt1 decoy to I_pi = 0.9480), DAPT Notch blockade all-tip hypersprouting (tip fraction = 1.0000, 0 stalk cells), Dll4+/- haploinsufficiency (tip fraction = 0.7500), and WT : Vegfr2+/- mosaic competition (WT tip occupancy = 0.8571); reproduces the 8-point Notch gain sweep (Pearson r = 0.99781, RMSE = 0.0136, p = 2.61e-8); and segments the SHA-256-locked Homo sapiens HUVEC angiogenic sprouting vascular network micrograph (plexus density = 0.1967, 69 branch junctions, 58 sprouting tips, tip/junction ratio = 0.8406, inter-tip NND CV = 0.7003).

- **Model fidelity to source: 95** — Implements the Brillouin k = pi linear stability & sFlt1 decoy verifier (`verify_brillouin_instability_and_flt1_rescue`, L136-181), coupled M = 16 multicellular VEGFR2-Dll4-Notch1 + 2D filopodial sprout outgrowth solver (`simulate_vessel_notch_sprouting`, L202-395), 5-regime + sFlt1 rescue evaluator (`evaluate_angiogenic_regimes`, L398-429), and 8-point Notch trans-gain sweep (`evaluate_notch_gain_sweep`, L436-499), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean`.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L136–429; `2021-ibrahim-vascular-angiogenesis-notch.py` L436–499; `morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean` L43–139
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies the CC BY 4.0 Homo sapiens HUVEC angiogenic sprouting vascular network fluorescence micrograph (`huvec_angiogenic_sprouting_plexus_specimen.jpg`, 1224x904 px), segments the calcein-stained vascular plexus area density (0.1967), extracts the medial-axis skeleton to quantify N_junc = 69 branch junctions (degree >= 3) and N_tip = 58 blind-ending filopodial sprouting tips (degree == 1, tip-to-junction ratio = 0.8406), and computes inter-tip NND CV = 0.7003 (`segment_angiogenic_plexus_specimen`, L506-581).  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L506–581; `docs/media/vascular_angiogenesis_notch_comparison.png` L0–0
- **Parameter expressivity: 93** — Evaluates all 5 canonical angiogenic regimes plus sFlt1 decoy rescue (Physiological WT, High uniform VEGF oscillations, DAPT k_t=0 blockade, Dll4+/- haploinsufficiency, WT:Vegfr2+/- mosaic chimera, and High VEGF + sFlt1 rescue), sweeps 8 Notch trans-activation gains k_t in [0.00, 1.25], and exposes `--vegf`, `--kt`, `--dll4-dosage`, `--flt1-decoy`, and `--seed` CLI overrides (L950-976).  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L398–499; `2021-ibrahim-vascular-angiogenesis-notch.py` L950–976
- **Morphometric validation: 94** — `measure_morphoeval()` (L798-899) computes 9 reference-anchored morphometric measurements with uncertainty, Pearson p-values, RMSE, and negative controls: Physiological tip-cell fraction = 0.5000 (ref 0.5000), Brillouin k=pi alternation index = 0.9472 (ref 0.9450, sFlt1 rescue delta +0.8858), DAPT Notch blockade tip fraction = 1.0000 (ref 1.0000, 0 stalk cells), Dll4+/- haploinsufficiency tip fraction = 0.7500 (ref 0.7500), High-VEGF synchronous oscillation CV = 0.3808 (ref 0.4150, sFlt1 rescue CV = 0.0134), WT:Vegfr2+/- chimera WT tip occupancy = 0.8571 (ref 0.8750), 8-point Notch gain sweep Pearson r = 0.99781 (RMSE = 0.0136, p = 2.61e-8), HUVEC specimen plexus density = 0.1967 (ref 0.1967), and HUVEC specimen tip-to-junction ratio = 0.8406 (58 tips / 69 junctions, inter-tip NND = 31.11 +- 21.79 px, CV = 0.7003).  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L798–899
- **Source and data traceability: 95** — Vendors `datasets/vascular_angiogenesis_notch/` with 5 BibTeX entries and verified DOIs (`10.1371/journal.pcbi.1000052`, `10.1038/nature05571`, `10.1038/ncb2103`, `10.1016/j.devcel.2009.06.004`, `10.1006/jtbi.1996.0233`), `scout_manifest.json`, `vascular_angiogenesis_notch_reference.json`, and `huvec_angiogenic_sprouting_plexus_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L99-129).  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L99–129; `datasets/vascular_angiogenesis_notch/vascular_angiogenesis_notch_reference.json` L1–145
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation to Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature, Jakobsson et al. (2010) Nat. Cell Biol., Chappell et al. (2009) Dev. Cell, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_060_vascular_angiogenesis_notch.lean` (L1-60, L136-499).  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L1–60
- **Reproducibility package: 93** — 8-suite `self_test()` (`L905-947`) verifying SHA-256 dataset integrity, Brillouin k=pi instability & sFlt1 decoy rescue, physiological salt-and-pepper alternation, DAPT all-tip hypersprouting collapse, Dll4+/- haploinsufficiency & high-VEGF oscillations, Vegfr2+/- mosaic competition & 8-point k_t sweep, HUVEC specimen CV segmentation, and bitwise determinism.  
  evidence: `2021-ibrahim-vascular-angiogenesis-notch.py` L905–947

Remediation:
- Couple the multicellular tip/stalk selection network to lumenized blood flow and wall shear stress to model post-anastomosis vascular pruning and arteriovenous specification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bentley, Gerhardt & Bates (2008) PLoS Comput. Biol., Hellström et al. (2007) Nature & Jakobsson et al. (2010) Nat. Cell Biol. VEGF-VEGFR2-Dll4-Notch1 tip/stalk salt-and-pepper lateral inhibition and angiogenic sprouting
- **spec_fingerprint** (semantic model + inputs): `75ad27375139b901d6c87702d5f02c739a8e698db79579ceae73bdab1d021c1b`
- **input_dataset_sha256**: `datasets/vascular_angiogenesis_notch/huvec_angiogenic_sprouting_plexus_specimen.jpg` (`9bf4f3e2692bd2ac…`), `datasets/vascular_angiogenesis_notch/paper_citations.bib` (`a96e44f99de8a756…`), `datasets/vascular_angiogenesis_notch/scout_manifest.json` (`2b97b38c418e4c44…`), `datasets/vascular_angiogenesis_notch/vascular_angiogenesis_notch_reference.json` (`8821743569ceaf2e…`)
- `docs/media/vascular_angiogenesis_notch_comparison.png` — artifact_sha256 `467bd8172c349e7c…`, 739,170 B, 2490×1530 px

| measurement | value | reference | method |
|---|---:|---|---|
| Physiological VEGF salt-and-pepper regime tip-cell fraction | 0.5 tip cell fraction | 0.5 | 16-cell coupled VEGFR2-Dll4-Notch1 vessel wall ODE at VEGF=1.0, k_t=1.0 (N_tip=8/16, spacing=36.0 um) *(period-2 Tip-Stalk alternation with Brillouin k=pi index I_pi = 0.9472 and sprout spacing = 36.0 um)* |
| Physiological salt-and-pepper Brillouin k=pi period-2 alternation index | 0.9472 Brillouin k=pi alternation index | 0.945 | Normalized staggered Fourier mode amplitude at k=pi across M=16 endothelial cells (sFlt1 rescue delta = +0.8858) *(DAPT blockade collapses alternation to 0.0000; sFlt1 decoy rescues high-VEGF alternation from 0.0622 to 0.9480)* |
| DAPT Notch blockade (k_t = 0.0) all-tip hypersprouting fraction | 1.0 tip cell fraction | 1.0 | Negative Notch ablation control with k_t=0.0 => N_i=0 => V_i=V_max (N_tip=16/16, N_stalk=0, delta = +0.5000) *(100% all-tip hypersprouting collapse with 0 stalk cells and alternation index = 0.0000)* |
| Dll4+/- haploinsufficiency heterozygote tip-cell fraction | 0.75 tip cell fraction | 0.75 | 16-cell vessel wall ODE with Dll4 ligand dosage = 0.5 (N_tip=12/16, 1.5x WT tip density) *(Dll4+/- haploinsufficiency increases tip-cell fraction from 0.5000 to 0.7500 (+50% hyperbranching))* |
| High uniform VEGF overload synchronous Dll4 oscillation CV | 0.3808 temporal CV (SD / mean) | 0.415 | Temporal Dll4 coefficient of variation across M=16 cells under 3.4x uniform VEGF overload (sFlt1 rescue reduces CV to 0.0134) *(excessive uniform VEGF triggers synchronous anti-phase/in-phase oscillations; soluble Flt1 decoy rescues stable salt-and-pepper (CV = 0.0134))* |
| WT : Vegfr2+/- mosaic chimera wild-type tip-cell occupancy | 0.8571 WT fraction of tip cells | 0.875 | 1:1 mosaic vessel wall pairing WT (V_max=1.0) and Vegfr2+/- (V_max=0.5) endothelial cells across M=16 positions *(WT cells win 85.7% of tip positions vs 14.3% for Vegfr2+/- cells)* |
| Notch trans-activation gain sweep tip-cell fraction correlation | 0.99781 Pearson r | 1.0 | 8-point Notch gain sweep k_t in [0.00, 1.25] comparing simulated tip fraction against Hellström et al. (2007) benchmark (RMSE = 0.0136, p = 2.61e-08) *(RMSE = 0.0136, p = 2.61e-08; alternation sweep r = 0.99809 (RMSE = 0.0227))* |
| HUVEC angiogenic sprouting plexus specimen vascular area density | 0.1967 vascular area fraction | 0.1967 | Calcein green fluorescence morphological segmentation on huvec_angiogenic_sprouting_plexus_specimen.jpg (N_junc=69, N_tip=58) *(vascular cord density = 0.1967 with 69 branch junctions)* |
| HUVEC angiogenic sprouting plexus specimen sprouting tip-to-junction ratio | 0.8406 sprouting tips / branch junctions | 0.8406 | Medial-axis skeleton degree-1 endpoint vs degree>=3 junction detection (N_tip=58, N_junc=69, inter-tip NND CV=0.7003) *(N_tip = 58 sprouting tips, N_junc = 69 junctions, inter-tip NND = 31.11 +- 21.79 px (CV = 0.7003))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2021-ibrahim-vascular-angiogenesis-notch.py` after completing Step 12 closed-loop remediation in `2026-10-19-vascular-angiogenesis-notch-loop`.
