# MorphoEval report — audit `2026-10-19-butterfly-eyespot-focus-rings-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/98 · supersession chain `2026-10-19-butterfly-eyespot-focus-rings-audit -> 2026-10-19-butterfly-eyespot-focus-rings-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Nijhout (1990) & Brakefield et al. (1996) butterfly wing-vein focus determination and concentric French-flag eyespot ring morphogenesis (`2001-nijhout-butterfly-eyespot-focus-rings.py`) | 93 | 91 | 91 | 92 | 94 | 92 | 91 | 92.3 | 91.4 | 92.3 | 91.9 | **91.9** | faithful |

## Findings and remediation

### Nijhout (1990) & Brakefield et al. (1996) butterfly wing-vein focus determination and concentric French-flag eyespot ring morphogenesis  —  overall 91.9 (faithful)

Pass 2 (v2) remediated evaluation of `2001-nijhout-butterfly-eyespot-focus-rings.py` (consensus score 92.0/100, up from 51.5/100 in Pass 1). The remediated script couples Stage-1 2D Meinhardt wing-compartment vein-boundary reaction-diffusion focus determination (`solve_stage1_vein_focus_rd`, isolating the organizer focus on the compartment midline `y_f = 0.0000 mm`, Nijhout 1990) to Stage-2 2D focal morphogen decay and concentric *Spalt* / *Engrailed* / *Distal-less* threshold induction across Wet-Season Wild-Type (`c0 = 1.00`), Dry-Season Ecdysone Polyphenism (`c0 = 0.42`, invariant gold annulus `Delta_R = 0.2648 mm`), *Goldeneye* Spalt-threshold shift (`theta_black = 0.44`, `Delta_R = 0.6941 mm`), Twin-Focus Hourglass Fusion (`d_12 = 1.50 mm`, bridge overlap `0.6765 mm`), and Early Pupal Focal Microcautery Ablation (`c0 = 0.05 -> R_gold = 0`, 6-point timing series `r = 1.0000`), and performs CV DoG focus detection and azimuthal radial luminance/chroma ring segmentation on the SHA-256-locked *Bicyclus anynana* ventral wing specimen photograph (`R_black / R_gold = 0.7569`), backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Couples Stage-1 2D Meinhardt activator-inhibitor reaction-diffusion inside a longitudinal wing-vein-bounded cell compartment (`solve_stage1_vein_focus_rd`, L114-187) to Stage-2 2D focal morphogen decay and concentric *Spalt* / *Engrailed* / *Distal-less* threshold induction (`solve_stage2_concentric_rings_2d` and `evaluate_all_eyespot_regimes`, L192-398) across all 5 experimental regimes plus the 6-point pupal microcautery timing series. Backed by 4 machine-checked Lean 4 theorems in `Loop_098_butterfly_eyespot_focus_rings.lean`.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L114–298; `2001-nijhout-butterfly-eyespot-focus-rings.py` L301–398; `morphoeval/lean/loops/Loop_098_butterfly_eyespot_focus_rings.lean` L24–169
- **Morphological realism / specimen match: 91** — Vendors, SHA-256-verifies, and performs Difference-of-Gaussians focus localization and azimuthal radial luminance + golden-chroma concentric ring segmentation (`segment_bicyclus_eyespot_specimen`, L403-496) on the CC BY-SA 4.0 *Bicyclus anynana* ventral wing specimen photograph (`datasets/butterfly_eyespot_focus_rings/bicyclus_anynana_eyespot_specimen.jpg`), measuring primary Cu1 focus `(596, 231)`, central white pupil radius `R_pupil = 3.663 px` (`0.1952 mm`), black *Spalt* disc radius `R_black = 15.442 px` (`0.8231 mm`), outer *Engrailed* gold ring radius `R_gold = 20.402 px` (`1.0874 mm`), and black-to-gold ratio `0.7569` (vs theoretical `0.7567`).  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L403–496; `docs/media/butterfly_eyespot_focus_rings_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates all 5 seasonal, genetic, and surgical perturbation regimes (`wet_season_wildtype`, `dry_season_polyphenism`, `goldeneye_mutant`, `twin_focus_hourglass`, `focal_cautery_ablation`), validates the 6-point pupal cautery timing curve (`1-24 hr`), and exposes `--c0`, `--lambda-mm`, `--theta-black`, `--theta-gold`, `--d12`, `--seed`, and `--skip-gif` CLI overrides (L755-785).  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L301–398; `2001-nijhout-butterfly-eyespot-focus-rings.py` L755–785
- **Morphometric validation: 92** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L624-715), covering Stage-1 midline focus offset (`0.0000 mm`), wet-season `R_gold` (`1.0882 mm`) and `R_black / R_gold` ratio (`0.7567`), seasonal polyphenism invariant gold annulus width (`0.2648 mm`), *Goldeneye* expanded gold annulus (`0.6941 mm`), early pupal cautery ablation (`0.0000 mm`, timing series `r = 1.0000`), twin-focus bridge overlap (`0.6765 mm`), and empirical *Bicyclus anynana* Cu1 specimen `R_black / R_gold` (`0.7569`) and calibrated `R_gold` (`1.0874 mm`).  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L624–715
- **Source and data traceability: 94** — Vendors `datasets/butterfly_eyespot_focus_rings/` with `butterfly_eyespot_focus_rings_reference.json`, `bicyclus_anynana_eyespot_specimen.jpg` (CC BY-SA 4.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (DOIs `10.1098/rspb.1990.0048`, `10.1006/dbio.1995.1065`, `10.1038/384236a0`, `10.1016/S0960-9822(01)00429-8`, `10.1038/416844a`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L92-109).  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L75–109; `datasets/butterfly_eyespot_focus_rings/butterfly_eyespot_focus_rings_reference.json` L1–165
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and solver step directly to Nijhout (1990) Proc. R. Soc. B, French & Brakefield (1995) Dev. Biol., Brakefield et al. (1996) Nature, Brunetti et al. (2001) Curr. Biol., Beldade et al. (2002) Nature, and the 4 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_098_butterfly_eyespot_focus_rings.lean`.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L1–48
- **Reproducibility package: 91** — Deterministic execution (`seed=2001`), 8-check `self_test()` suite (L718-752) verifying SHA-256 provenance, JSON parity, Stage-1 midline focus localization, concentric threshold ordering, seasonal gold annulus invariance and *Goldeneye* expansion, focal cautery collapse and twin-focus hourglass fusion, specimen Cu1 ring segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 2-panel GIF outputs.  
  evidence: `2001-nijhout-butterfly-eyespot-focus-rings.py` L718–752

Remediation:
- Couple Stage-1 focus reaction-diffusion directly to curved wing-margin parafocal element (chevron) formation across all 7 hindwing compartments simultaneously (Nijhout 1990, 2017).
- Incorporate explicit temperature-dependent 20-hydroxyecdysone (20E) receptor kinetics during the wandering-to-pupal transition window.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Nijhout (1990) & Brakefield et al. (1996) butterfly wing-vein focus determination and concentric French-flag eyespot ring morphogenesis
- **spec_fingerprint** (semantic model + inputs): `0a2086f2436043ce9998603ef36e23feb6c621834d7d960a99310b296156772b`
- **input_dataset_sha256**: `datasets/butterfly_eyespot_focus_rings/bicyclus_anynana_eyespot_specimen.jpg` (`f1826cee44602782…`), `datasets/butterfly_eyespot_focus_rings/butterfly_eyespot_focus_rings_reference.json` (`b19590626bdaff79…`), `datasets/butterfly_eyespot_focus_rings/scout_manifest.json` (`7cfe9d2221ea5b8b…`)
- `docs/media/butterfly_eyespot_focus_rings_comparison.png` — artifact_sha256 `119a30775931cd53…`, 505,778 B, 2237×1424 px

| measurement | value | reference | method |
|---|---:|---|---|
| Stage-1 2D wing-compartment focus midline offset |y_focus| | 0.0 mm | 0.0 | Center-of-mass anteroposterior offset |y_f| of the Stage-1 Meinhardt activator focal organizer inside [0, 3.60] x [-0.90, 0.90] mm *(focus_x_mm = 2.028 mm (x/L_x = 0.5634))* |
| Wet-season wild-type outer Engrailed gold ring radius R_gold (c0 = 1.00) | 1.0883 mm | 1.0882 | Analytical & 2D threshold radius R_gold = lambda * ln(c0 / theta_gold) with lambda = 0.480 mm, c0 = 1.00, theta_gold = 0.1036 *(R_pupil = 0.1951 mm < R_black = 0.8234 mm < R_gold = 1.0883 mm)* |
| Wet-season wild-type black-to-gold diameter ratio R_black / R_gold | 0.7566 radius ratio | 0.7567 | Ratio of Spalt/Dll black melanin disc radius R_black = 0.8234 mm to outer Engrailed gold radius R_gold = 1.0882 mm *(pupil_to_gold_ratio = 0.1793)* |
| Seasonal ecdysone polyphenism invariant gold annulus width Delta_R (wet vs dry season) | 0.2649 mm | 0.2648 | Outer gold annulus width Delta_R = R_gold(c0) - R_black(c0) = lambda * ln(theta_black / theta_gold) at dry-season c0 = 0.42 vs wet-season c0 = 1.00 *(|Delta_R_wet - Delta_R_dry| = 2.22e-16 mm)* |
| Goldeneye mutant expanded Engrailed gold annulus width Delta_R_Goldeneye (theta_black = 0.44) | 0.6942 mm | 0.6941 | Outer gold annulus width when Spalt activation threshold shifts upward from theta_black = 0.1799 to 0.4400 *(Goldeneye black_to_gold_ratio = 0.3621 (vs WT 0.7567))* |
| Early pupal focal microcautery ablation outer radius R_gold (c0 = 0.05 <= theta_gold) | 0.0 mm | 0.0 | Complete eyespot ablation negative control when focal source strength c0 = 0.05 falls below theta_gold = 0.1036; 6-point timing series r = 1.0000 *(6-point pupal cautery timing series Pearson r = 1.0000, RMSE = 0.0001 mm)* |
| Twin-focus hourglass fusion inter-focal bridge overlap width w_bridge (d_12 = 1.50 mm) | 0.6765 mm | 0.6765 | Inter-focal gold-ring overlap width 2*R_gold - d_12 and midpoint superposition c_mid = 2*c0*exp(-d_12/(2*lambda)) = 0.4192 > theta_black *(midpoint_c = 0.4192, bridge_fused_gold = True, bridge_fused_black = True)* |
| Bicyclus anynana specimen Cu1 eyespot black-to-gold radius ratio R_black / R_gold (bicyclus_anynana_eyespot_specimen.jpg) | 0.7571 radius ratio | 0.7569 | Azimuthal-mean radial luminance segmentation around primary Cu1 focus (596, 231) (R_black = 15.441 px / R_gold = 20.395 px) *(pupil_to_gold_ratio = 0.1796 (R_pupil = 3.662 px))* |
| Bicyclus anynana specimen Cu1 calibrated outer gold ring radius R_gold (bicyclus_anynana_eyespot_specimen.jpg) | 1.0871 mm | 1.0874 | Calibrated outer Engrailed gold ring radius R_gold_px * 0.0533 mm/px from azimuthal radial luminance crossing L(r) = 0.61 *(calibrated gold annulus width = 0.2641 mm (vs theoretical 0.2649 mm))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2001-nijhout-butterfly-eyespot-focus-rings.py` superseding `2026-10-19-butterfly-eyespot-focus-rings-audit`.
