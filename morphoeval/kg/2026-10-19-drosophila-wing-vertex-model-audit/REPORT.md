# MorphoEval report — audit `2026-10-19-drosophila-wing-vertex-model-audit`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Farhadifar et al. (2007) Drosophila wing disc apical vertex model & Lewis's law epithelial packing (`2007-farhadifar-drosophila-wing-vertex-model.py`) | 68 | 35 | 46 | 54 | 64 | 68 | 69 | 60.7 | 42.1 | 67.1 | 55.2 | **55.2** | partially faithful |

## Findings and remediation

### Farhadifar et al. (2007) Drosophila wing disc apical vertex model & Lewis's law epithelial packing  —  overall 55.2 (partially faithful)

Pass 1 (v1) evaluation of `2007-farhadifar-drosophila-wing-vertex-model.py` (57.7/100). The v1 baseline implements the Farhadifar et al. (2007) dimensionless apical vertex energy functional and simulates 2D periodic Voronoi-vertex relaxation in the proliferating wing disc vs. annealed pupal wing states, but omits analytical ground-state phase diagram classification (Case I rigid hexagonal vs. Case II soft network vs. Case III cell collapse), omits explicit vertex gradient forces -dE/dr_i and simulated UV laser junction ablation recoil kinetics, does not load or segment the vendored apical junction-stained polygonal epithelium specimen (`drosophila_wing_epithelium_specimen.jpg`), and exposes no CLI parameter flags beyond `--self-test`.

- **Model fidelity to source: 68** — Evaluates the Farhadifar et al. (2007) Eq. (1)-(2) dimensionless single-cell apical energy (`compute_single_cell_vertex_energy`, L32-52) and Lewis's Law (1928) linear area-side scaling (`lewis_law_ratio`, L55-57), and contrasts proliferating wing disc division disorder against post-mitotic pupal annealing (`simulate_vertex_packing_v1`, L85-150). However, v1 uses centroidal Voronoi relaxation rather than explicit tri-cellular vertex force gradients `-dE/dr_i` with T1 topological edge swaps, omits Case I / II / III ground-state phase boundaries (`Lambda_bar_c(Gamma_bar)`), and omits UV laser ablation junction recoil simulation (Farhadifar et al. 2007 Fig. 4).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L32–52; `2007-farhadifar-drosophila-wing-vertex-model.py` L85–150
- **Morphological realism / specimen match: 35** — Although `datasets/drosophila_wing_vertex_model/drosophila_wing_epithelium_specimen.jpg` is vendored in the repository, the v1 script never loads, verifies, or segments the apical junction-stained epithelial micrograph (`render_outputs_v1`, L153-201) and renders only a 3-panel synthetic plot.  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L153–201; `docs/media/drosophila_wing_vertex_model_comparison.png` L0–0
- **Parameter expressivity: 46** — Simulates only a single parameter point `(Lambda_bar=0.12, Gamma_bar=0.04)` in two states (`proliferating=True/False`) without mapping the 2D `(Lambda_bar, Gamma_bar)` ground-state phase diagram across Cases I, II, and III, without a continuous sweep across `Lambda_bar` or `Gamma_bar`, and without CLI flags for `--lambda-bar`, `--gamma-bar`, `--K`, `--A0`, or `--steps` (L243-252).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L156–157; `2007-farhadifar-drosophila-wing-vertex-model.py` L243–252
- **Morphometric validation: 54** — Computes simulated hexagonal fraction `P(6)`, mean side number `<n>`, and theoretical Lewis's law ratio for `n=6` (`measure_morphoeval`, L204-233), but does not compare simulated `P(n)` (`n=4..9`) against Farhadifar et al. (2007) Fig. 4B / Gibson et al. (2006) larval wing disc and Classen et al. (2005) pupal wing distributions via Pearson `r` / RMSE, does not compare `<A_n>/<A>` against Lewis's law across `n=4..9`, and does not segment the biological specimen micrograph to extract empirical cell areas and polygon classes.  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L204–233
- **Source and data traceability: 64** — Module docstring cites Farhadifar et al. (2007) and Lewis (1928) DOIs (L5-7) and `datasets/drosophila_wing_vertex_model/` exists on disk, but the v1 script does not load `drosophila_wing_vertex_model_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L5–13
- **Code-to-model traceability: 68** — Functions correspond to the Farhadifar vertex energy and Lewis's Law, and `morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean` proves 4 theorems, but v1 docstrings lack equation-by-equation mappings to Farhadifar et al. (2007) Eqs. 1-3 & Figs. 2-4, Gibson et al. (2006), and Classen et al. (2005).  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L32–150; `morphoeval/lean/loops/Loop_043_drosophila_wing_vertex_model.lean` L1–125
- **Reproducibility package: 69** — Uses deterministic `np.random.default_rng(2007)` and provides `--self-test` and `measure_morphoeval()` (L204-240), but `self_test()` checks only 2 basic assertions without SHA-256 dataset verification, Case I/II/III ground-state parity, or two-run bitwise determinism checks.  
  evidence: `2007-farhadifar-drosophila-wing-vertex-model.py` L236–241

Remediation:
- Implement explicit tri-cellular vertex force relaxation (-dE/dr_i) with T1 topological neighbor exchanges, analytical Case I / II / III ground-state phase diagram classification in the (Lambda_bar, Gamma_bar) plane, and simulated UV laser junction ablation recoil velocity profiles across all 4 regimes (Proliferating Wing Disc, Annealed Pupal Wing, High-Contractility Case I, and Soft Case II).
- Load and SHA-256-verify `datasets/drosophila_wing_vertex_model/drosophila_wing_vertex_model_reference.json` and `drosophila_wing_epithelium_specimen.jpg`, segment the apical green junction network and nuclear seeds via watershed tessellation to extract empirical polygon classes P(n) and normalized cell areas <A_n>/<A>, and compute Pearson r and RMSE against the vertex model and Lewis's Law.
- Execute a 7-point continuous parameter sweep across normalized line tension Lambda_bar spanning Case II and Case I, validate Gibson et al. (2006) Markov cleavage and Classen et al. (2005) pupal hexagonal repacking, and expose full CLI flags.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Farhadifar et al. (2007) Drosophila wing disc apical vertex model & Lewis's law epithelial packing
- **spec_fingerprint** (semantic model + inputs): `00f70e92aaa338b22956554153bb93fbccd372cdaae2e3f47f4744bcadfffe52`
- `docs/media/drosophila_wing_vertex_model_comparison.png` — artifact_sha256 `ab67d29234ea2884…`, 209,107 B, 1890×588 px

| measurement | value | reference | method |
|---|---:|---|---|
| wing_disc_proliferating_hexagonal_fraction_P6 | 0.6111 fraction | 0.455 | Simulated hexagonal cell fraction P(6) in proliferating larval wing disc (Farhadifar et al. 2007 Fig. 4B) |
| pupal_wing_annealed_hexagonal_fraction_P6 | 0.7222 fraction | 0.78 | Simulated hexagonal cell fraction P(6) in post-mitotic annealed pupal wing (Classen et al. 2005 / Farhadifar et al. 2007) |
| euler_mean_polygon_side_number | 6.0 sides | 6.0 | Mean polygon side number <n> across periodic 3-valent vertex tessellation (Euler 3-valent theorem <n>=6) |
| lewis_law_hexagon_normalized_area | 1.0 normalized_area | 1.0 | Lewis (1928) linear area-side formula <A_6>/<A> = (6 - 2)/4 = 1.0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2007-farhadifar-drosophila-wing-vertex-model.py` prior to Pass 2 remediation in `2026-10-19-drosophila-wing-vertex-model-loop`.
