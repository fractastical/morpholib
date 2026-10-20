# MorphoEval report — audit `2026-10-19-zebrafish-three-cell-stripes-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Nakamasu et al. (2009) 3-variable zebrafish pigment-cell reaction-diffusion network, laser-ablation polarity & leopard (cx41.8) spot-to-stripe morphospace (`2009-nakamasu-zebrafish-three-cell-stripes.py`) | 52 | 28 | 44 | 44 | 62 | 58 | 68 | 49.3 | 34.0 | 63.3 | 47.4 | **47.4** | partially faithful |

## Findings and remediation

### Nakamasu et al. (2009) 3-variable zebrafish pigment-cell reaction-diffusion network, laser-ablation polarity & leopard (cx41.8) spot-to-stripe morphospace  —  overall 47.4 (partially faithful)

Pass 1 (v1) baseline evaluation of `2009-nakamasu-zebrafish-three-cell-stripes.py` (overall weighted score: 49.8 / 100). The v1 script computes the linear dispersion curve sigma(k) of the 3-variable Nakamasu et al. (2009) melanophore-xanthophore-factor matrix J0 - k^2 D (k_c = 3.41, lambda_c = 1.84), but approximates spatial stripes via a static cosine formula (`synthesize_cosine_stripes`) rather than numerically integrating the nonlinear 3-species piecewise-saturated reaction-diffusion PDE for (u, v, w) with zero-flux Neumann boundaries; omits in silico laser-ablation regeneration and xanthophore-depletion survival collapse (Nakamasu 2009 Figs. 2-3); omits the leopard (cx41.8) spot-to-labyrinth-to-stripe morphospace sweep and dual negative controls; does not load or segment the vendored Danio rerio specimen photograph; and returns only 2 analytical measurements without CLI parameter flags.

- **Model fidelity to source: 52** — Evaluates the 3x3 Jacobian J0 and linear Turing dispersion curve sigma(k) of Nakamasu et al. (2009) across k in [0, 6.5] (L27-68), but `synthesize_cosine_stripes` (L71-79) uses a heuristic cosine wave rather than integrating the coupled 3-variable nonlinear reaction-diffusion PDE for melanophores (u), xanthophores (v), and long-range survival factor (w) with saturating production clip(a*u + b*v + d*w + e, 0, F_max), and omits both laser-ablation regeneration polarity (Figs. 2-3) and horizontal myoseptum / iridophore X0 pioneer guidance.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L27–68; `2009-nakamasu-zebrafish-three-cell-stripes.py` L71–79
- **Morphological realism / specimen match: 28** — Although `datasets/zebrafish_three_cell_stripes/danio_rerio_specimen.jpg` is vendored in the repository, the v1 script never loads, segments, or displays the biological zebrafish photograph (`docs/media/zebrafish_three_cell_stripes_comparison.png` shows only 2 synthetic panels), and performs no quantitative comparison against Danio rerio flank inter-stripe wavelength or horizontal anisotropy.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L82–111; `docs/media/zebrafish_three_cell_stripes_comparison.png` L0–0
- **Parameter expressivity: 44** — Renders a single static cosine field without sweeping the *leopard* (cx41.8) mutant spot -> labyrinth -> wild-type guided stripe -> inverse spot morphospace (Nakamasu 2009 Fig. 5; Watanabe et al. 2006), without laser-ablation perturbations, without equal-diffusion (D_w = D_u) or mutual-repulsion ablation negative controls, and without CLI flags for kinetic or diffusive parameters (L139-145).  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L82–98; `2009-nakamasu-zebrafish-three-cell-stripes.py` L139–145
- **Morphometric validation: 44** — `measure_morphoeval()` (L114-135) returns only 2 analytical scalar values (k_c and lambda_c) without measuring Routh-Hurwitz invariants, 2D PDE melanophore-xanthophore anti-phase correlation, xanthophore laser-ablation melanophore survival loss, negative-control quenching, or biological Danio rerio inter-stripe spacing CV and horizontal orientation anisotropy.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L114–135
- **Source and data traceability: 62** — Header cites Nakamasu et al. (2009) DOI `10.1073/pnas.0808622106` and `datasets/zebrafish_three_cell_stripes/` exists on disk, but the v1 script does not load `zebrafish_three_cell_stripes_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L1–10
- **Code-to-model traceability: 58** — Basic docstrings are present for linear dispersion (L27-68), but functions lack explicit equation and figure mappings to Nakamasu et al. (2009) Figs. 2-5 & SI Eqs. (1)-(6), Watanabe et al. (2006) *leopard* cx41.8 alleles, and the 3 Lean 4 theorems in `Loop_007_zebrafish_three_cell_stripes.lean`.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L27–79
- **Reproducibility package: 68** — Includes deterministic `np.random.default_rng(2009)` and `--self-test` (L138-144), but `self_test()` checks only 2 basic inequalities without verifying dataset SHA-256 locks, 3-species PDE convergence, laser-ablation response, negative-control quenching, specimen segmentation, or bitwise determinism across two runs.  
  evidence: `2009-nakamasu-zebrafish-three-cell-stripes.py` L138–144

Remediation:
- Replace `synthesize_cosine_stripes` with a genuine 2D nonlinear 3-species (u, v, w) reaction-diffusion PDE solver (spectral DCT-II IMEX with zero-flux Neumann boundary conditions and piecewise-saturating synthesis rates) modeling wild-type horizontal myoseptum-guided stripes, unguided labyrinthine stripes, and *leopard* (cx41.8) mutant spot regimes.
- Simulate Nakamasu et al. (2009) Figs. 2-3 laser-ablation experiments in silico: (1) selective xanthophore interstripe ablation demonstrating melanophore stripe survival collapse when long-range w support is removed, and (2) rectangular stripe ablation demonstrating directional horizontal stripe regeneration vs isotropic spot/labyrinth fill-in.
- Implement dual formal negative controls around the homogeneous steady state: (1) equal-diffusion control D_w = D_u = D_v = 0.02 (testing Lean Theorem 3), and (2) ablated short-range mutual repulsion b_u = a_v = 0.0 (testing Lean Theorem 1 where det J_uv > 0 prevents pigment segregation).
- Load and SHA-256-verify `datasets/zebrafish_three_cell_stripes/zebrafish_three_cell_stripes_reference.json`, `danio_rerio_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib`; segment the dark melanophore stripes and yellow/iridescent interstripes from `danio_rerio_specimen.jpg`; expose full CLI flags; expand `measure_morphoeval()` to 8 reference-backed morphometrics; and render an 8-panel comparison PNG and 6-frame growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Nakamasu et al. (2009) 3-variable zebrafish pigment-cell reaction-diffusion network, laser-ablation polarity & leopard (cx41.8) spot-to-stripe morphospace
- **spec_fingerprint** (semantic model + inputs): `e325ab05dbc489df59ad15e0ad7e5ce094c3a9ea9e524b654d3adbb036c29214`
- **input_dataset_sha256**: `datasets/zebrafish_three_cell_stripes/danio_rerio_specimen.jpg` (`850b2d4a94c7cbcb…`), `datasets/zebrafish_three_cell_stripes/scout_manifest.json` (`d19df92d7c80de8a…`), `datasets/zebrafish_three_cell_stripes/zebrafish_three_cell_stripes_reference.json` (`fcc1809ce2e992bc…`)
- `docs/media/zebrafish_three_cell_stripes_comparison.png` — artifact_sha256 `7e46a54e2f714ffd…`, 71,468 B, 1287×506 px

| measurement | value | reference | method |
|---|---:|---|---|
| Nakamasu 3-cell critical wavenumber k_c | 3.4153 dimensionless | 3.4106 | argmax_k Re(eigvals(J0 - k^2 * diag(Du, Dv, Dw))) |
| Nakamasu characteristic stripe wavelength lambda_c | 1.8397 dimensionless length | 1.8423 | 2 * pi / k_c |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2009-nakamasu-zebrafish-three-cell-stripes.py` prior to Step 12 remediation in `2026-10-19-zebrafish-three-cell-stripes-loop`.
