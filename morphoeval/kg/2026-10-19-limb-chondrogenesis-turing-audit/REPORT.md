# MorphoEval report — audit `2026-10-19-limb-chondrogenesis-turing-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Newman & Frisch (1979) Science 3D Helmholtz/Turing standing-wave eigenmodes and proximodistal 1 -> 2 -> 3 skeletal chondrogenesis in developing Gallus gallus wing bud (`1979-newman-limb-chondrogenesis-turing.py`) | 48 | 28 | 44 | 40 | 60 | 56 | 68 | 46.7 | 32.5 | 62.1 | 45.6 | **45.6** | partially faithful |

## Findings and remediation

### Newman & Frisch (1979) Science 3D Helmholtz/Turing standing-wave eigenmodes and proximodistal 1 -> 2 -> 3 skeletal chondrogenesis in developing Gallus gallus wing bud  —  overall 45.6 (partially faithful)

Pass 1 (v1) baseline evaluation of `1979-newman-limb-chondrogenesis-turing.py` (overall weighted score: 47.4 / 100). The v1 baseline script computes the analytical critical mode number m = sqrt((Dv*fu + Du*gv)/(2*Du*Dv)) for stylopod (m=1) and autopod (m=3), but synthesizes spatial condensations via a closed-form trigonometric formula (`synthesize_standing_wave_box`) rather than numerically integrating the nonlinear cubic-saturated 2-component fibronectin/TGF-beta Turing reaction-diffusion PDE coupled to the shrinking apical progress zone L_PD(t) and frozen proximal chondrogenic compartments; omits the 4-digit chick hindlimb stage (HH28, m=4), AER surgical excision truncation controls (HH20 and HH24 excisions), and equal-diffusion / hyaluronate-excess negative controls; does not load or segment the vendored Gallus gallus wing skeleton specimen; and returns only 2 analytical scalar measurements without PDE peak counts or specimen morphometrics.

- **Model fidelity to source: 48** — Evaluates the analytical wavenumber formula `compute_newman_mode_number` (L25-28), but `synthesize_standing_wave_box` (L31-36) plots a static `-cos(2*m*pi*X)*sin(pi*Z)` formula instead of numerically integrating the nonlinear 2-component Turing reaction-diffusion PDE with Neumann/Dirichlet boundary conditions inside the shrinking apical progress zone L_PD(t) or assembling the composite 2D proximodistal limb bud (`1 -> 2 -> 3` skeletal compartments).  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L25–28; `1979-newman-limb-chondrogenesis-turing.py` L31–36
- **Morphological realism / specimen match: 28** — Although `datasets/limb_chondrogenesis_turing/chick_wing_skeleton_specimen.jpg` is vendored on disk, the v1 baseline script never loads, segments, or plots the Gallus gallus wing skeleton specimen, and performs no quantitative comparison of stylopod (1 humerus), zeugopod (2 radius+ulna), and autopod (3 digits) cross-sectional bone profiles against the PDE.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L39–64; `docs/media/limb_chondrogenesis_turing_comparison.png` L0–0
- **Parameter expressivity: 44** — Evaluates only 3 static mode numbers without simulating the full 4-stage proximodistal/hindlimb series (HH20 stylopod m=1 -> HH24 zeugopod m=2 -> HH27 wing autopod m=3 -> HH28 leg autopod m=4), omits Saunders (1948) / Summerbell et al. (1973) AER excision truncation controls (HH20 and HH24 excision) and equal-diffusion (Du=Dv) / hyaluronate-excess (sub-Turing) negative controls, and exposes only `--self-test` on the CLI (L99-106).  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L39–64; `1979-newman-limb-chondrogenesis-turing.py` L99–106
- **Morphometric validation: 40** — `measure_morphoeval()` (L67-90) returns only 2 analytical scalar endpoints (m=1.0 and m=3.0) without measuring nonlinear PDE skeletal element peak counts across stylopod/zeugopod/autopod, AER truncation counts, negative-control amplitude suppression, or segmented Gallus gallus wing bone counts and transverse spacing ratios.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L67–90
- **Source and data traceability: 60** — Header cites Newman & Frisch (1979) DOI `10.1126/science.462174`, and `datasets/limb_chondrogenesis_turing/` is present on disk, but the v1 script does not load `limb_chondrogenesis_turing_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L1–8
- **Code-to-model traceability: 56** — Basic docstrings are present, but functions lack explicit equation and table mappings to Newman & Frisch (1979) Eqs. (1)-(7) & Table 1, Saunders (1948) AER excision experiments, and the 3 Lean 4 theorems in `Loop_012_limb_chondrogenesis_turing.lean`.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L25–36
- **Reproducibility package: 68** — Includes `--self-test` (L93-97), but `self_test()` checks only 2 scalar equalities without verifying dataset SHA-256 locks, nonlinear PDE condensation counts across all 4 stages (1, 2, 3, 4 elements), AER truncation hierarchy, negative-control quenching, specimen 1->2->3 bone segmentation, or bitwise two-run determinism.  
  evidence: `1979-newman-limb-chondrogenesis-turing.py` L93–97

Remediation:
- Replace the static cosine-sine heuristic with a genuine nonlinear 2-component cubic-saturated Turing reaction-diffusion PDE solver (du/dt = f_u(L_PD, D_eff)*u + f_v*v - u^3 + D_u*Lap(u), dv/dt = g_u(L_PD, D_eff)*u + g_v*v + D_v*Lap(v)) with zero-flux Neumann anteroposterior boundaries (x in {0, L_AP}) and pinned/apical proximodistal boundaries (z in [0, L_PD(t)]), plus a 2D composite proximodistal limb bud solver that freezes stylopod (m=1) -> zeugopod (m=2) -> autopod (m=3) compartments sequentially.
- Simulate all 4 developmental regimes from Newman & Frisch (1979) Table 1 (HH20 stylopod L_PD=500 um -> 1 humerus; HH24 zeugopod L_PD=250 um -> 2 radius+ulna; HH27 wing autopod L_PD=140 um -> 3 digits II-IV; HH28 hindlimb autopod L_PD=115 um -> 4 pedal digits I-IV), the 3-step AER surgical excision truncation series ([1,0,0] -> [1,2,0] -> [1,2,3] elements), and dual negative controls (equal diffusion Du=Dv=0.5 and hyaluronate-excess sub-Turing fu=1.5).
- Load and SHA-256-verify `datasets/limb_chondrogenesis_turing/limb_chondrogenesis_turing_reference.json`, `chick_wing_skeleton_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib` at runtime; segment the humerus (1 bone), radius+ulna (2 bones), and carpometacarpus/digits II-IV (3 elements) directly from the Gallus gallus wing specimen; and compare specimen bone counts and transverse spacing ratios against the nonlinear Turing PDE.
- Expose full CLI parameters (--stage, --L-PD-um, --Du, --Dv, --fv, --gv, --steps, --seed), expand `measure_morphoeval()` to 8 reference-backed morphometric measurements, and upgrade `docs/media/limb_chondrogenesis_turing_comparison.png` (8-panel synthesis figure) and `docs/media/limb_chondrogenesis_turing_growth.gif` (sequential proximodistal outgrowth + AER excision animation).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Newman & Frisch (1979) Science 3D Helmholtz/Turing standing-wave eigenmodes and proximodistal 1 -> 2 -> 3 skeletal chondrogenesis in developing Gallus gallus wing bud
- **spec_fingerprint** (semantic model + inputs): `381dc7883b80c95da9d476a5865f816bf3bb5de908cfdb8ef8c6bbef3a244a82`
- **input_dataset_sha256**: `datasets/limb_chondrogenesis_turing/chick_wing_skeleton_specimen.jpg` (`771e26f9fe8c813d…`), `datasets/limb_chondrogenesis_turing/limb_chondrogenesis_turing_reference.json` (`c30d742bbf2a990a…`), `datasets/limb_chondrogenesis_turing/scout_manifest.json` (`7f3605aa11f5e077…`)
- `docs/media/limb_chondrogenesis_turing_comparison.png` — artifact_sha256 `9508d38c1641902f…`, 198,805 B, 1307×429 px

| measurement | value | reference | method |
|---|---:|---|---|
| HH20 stylopod mode number m(L_PD=500 um) | 1.0 elements | 1.0 | analytical wavenumber formula at fu=3.0 |
| HH27 chick wing autopod mode number m(L_PD=140 um) | 3.0 elements | 3.0 | analytical wavenumber formula at fu=11.0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1979-newman-limb-chondrogenesis-turing.py` prior to Step 12 remediation in `2026-10-19-limb-chondrogenesis-turing-loop`.
