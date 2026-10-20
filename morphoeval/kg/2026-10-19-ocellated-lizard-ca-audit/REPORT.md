# MorphoEval report — audit `2026-10-19-ocellated-lizard-ca-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Manukyan et al. (2017) Nature hexagonal skin-scale border thickness reduction & outer-totalistic cellular automaton (`2015-manukyan-ocellated-lizard-ca.py`) | 50 | 28 | 44 | 42 | 62 | 58 | 68 | 48.0 | 33.2 | 63.3 | 46.7 | **46.7** | partially faithful |

## Findings and remediation

### Manukyan et al. (2017) Nature hexagonal skin-scale border thickness reduction & outer-totalistic cellular automaton  —  overall 46.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2015-manukyan-ocellated-lizard-ca.py` (overall weighted score: 48.8 / 100). The v1 script implements the discrete 2D hexagonal outer-totalistic cellular automaton transition probabilities from Manukyan et al. (2017), but omits the underlying continuous Kondo-Asai reaction-diffusion PDE on a hexagonal skin-scale domain with inter-scale border thickness reduction D_border = P * D_interior (the central physical mechanism of the Nature paper showing how PDEs discretize into a scale-level CA); lacks negative controls (P = 0.0 insulated scales, P = 1.0 smooth continuum skin without scale borders, and equal-diffusion d = 1.0); never loads or segments the vendored Timon lepidus dorsal scale photograph; and exposes only 2 scalar measurements without CLI parameter flags.

- **Model fidelity to source: 50** — Implements the 6-neighbor hexagonal outer-totalistic cellular automaton (`count_hex_green_neighbors` and `simulate_hex_ca`, L27-74), but completely omits the continuous reaction-diffusion PDE with periodic scale-border thickness reduction D_border = P * D_interior (Manukyan et al. 2017 Fig. 4 & Methods Eqs. 1-8) that explains why continuous morphogen diffusion discretizes into monochromatic hexagonal skin scales.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L27–74
- **Morphological realism / specimen match: 28** — Although `datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or displays the biological *Timon lepidus* specimen photograph (`docs/media/ocellated_lizard_ca_comparison.png` shows only a 2-panel synthetic CA grid and histogram) and performs no quantitative comparison against empirical dorsal green/black scale lattices.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L77–108; `docs/media/ocellated_lizard_ca_comparison.png` L0–0
- **Parameter expressivity: 44** — Runs only a single fixed CA realization without sweeping the scale-border thickness reduction factor P in {0.0, 0.08, 0.17, 0.45, 1.0}, without negative controls (P=0.0 insulated scales, P=1.0 unattenuated continuum Turing waves, or d=1.0 equal diffusion), and without CLI flags for P, diffusivity ratio d, lattice size, steps, or seed (L136-143).  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L42–74; `2015-manukyan-ocellated-lizard-ca.py` L136–143
- **Morphometric validation: 42** — `measure_morphoeval()` (L111-129) returns only 2 CA summary numbers (mean green neighbors and green fraction) without measuring PDE vs CA histogram Jensen-Shannon / L1 convergence, intra-scale vs inter-scale concentration variance (bimodality), negative-control suppression, or segmented *Timon lepidus* dorsal scale neighbor statistics.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L111–129
- **Source and data traceability: 62** — Module docstring cites Manukyan et al. (2017) DOI `10.1038/nature22031`, but the v1 script does not load `ocellated_lizard_ca_reference.json` or verify SHA-256 digests of `datasets/ocellated_lizard_ca/` at runtime.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L1–10
- **Code-to-model traceability: 58** — Functions have brief docstrings (L27-74) but lack explicit mappings to Manukyan et al. (2017) Figs. 2-4, Methods Eqs. (1)-(8), Kondo & Asai (1995) reaction kinetics, and the 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_009_ocellated_lizard_ca.lean`.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L27–74
- **Reproducibility package: 68** — Uses deterministic `np.random.default_rng(2017)` and `--self-test` (L132-137), but `self_test()` checks only 2 coarse bounds without verifying dataset SHA-256 locks, PDE-to-CA convergence, negative controls, specimen segmentation, or bitwise two-run determinism.  
  evidence: `2015-manukyan-ocellated-lizard-ca.py` L132–137

Remediation:
- Implement the continuous reaction-diffusion PDE on a 2D hexagonal skin-scale domain with spatially modulated scale-border diffusivity D_s(x) = P * D_s,interior at inter-scale borders (Manukyan et al. 2017 Fig. 4 & Methods Eqs. 1-8), demonstrating how P = 0.17 discretizes continuous Turing waves into monochromatic hexagonal scales whose neighbor histogram converges to the outer-totalistic CA.
- Implement formal negative and ablation controls: (1) insulated scale borders P = 0.0 (testing Lean Theorem 2 where zero border flux freezes juvenile noise), (2) homogeneous continuum skin P = 1.0 without scale border thinning (where smooth Turing waves cut across scale boundaries, degrading intra-scale uniformity), and (3) equal-diffusion control d = Dv/Du = 1.0 (abolishing Turing instability).
- Load and SHA-256-verify `datasets/ocellated_lizard_ca/ocellated_lizard_ca_reference.json`, `timon_lepidus_specimen.jpg`, `scout_manifest.json`, and `paper_citations.bib` at runtime; segment individual green and black dorsal scales from `timon_lepidus_specimen.jpg`; and compare biological Delaunay nearest-neighbor green/black degree distributions against both the scale-border PDE and the hexagonal CA.
- Sweep all 5 border thickness reduction regimes P in {0.0, 0.08, 0.17, 0.45, 1.0}, expose full CLI parameters (--P, --d, --rows, --cols, --steps, --seed), expand `measure_morphoeval()` to 8 reference-backed morphometric measurements, and upgrade `docs/media/ocellated_lizard_ca_comparison.png` (8 panels) and `docs/media/ocellated_lizard_ca_growth.gif`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Manukyan et al. (2017) Nature hexagonal skin-scale border thickness reduction & outer-totalistic cellular automaton
- **spec_fingerprint** (semantic model + inputs): `242b9a7c86c1f9293dfbf14c7b4f7718e95fac3e748d9d6c90e4cc1c23174b29`
- **input_dataset_sha256**: `datasets/ocellated_lizard_ca/ocellated_lizard_ca_reference.json` (`ed1472207d9c79bb…`), `datasets/ocellated_lizard_ca/scout_manifest.json` (`e5bca2d8766b49e7…`), `datasets/ocellated_lizard_ca/timon_lepidus_specimen.jpg` (`bc8e149849e6729d…`)
- `docs/media/ocellated_lizard_ca_comparison.png` — artifact_sha256 `9450c57e57e5c19b…`, 32,694 B, 1213×559 px

| measurement | value | reference | method |
|---|---:|---|---|
| adult green scale mean green neighbor count | 2.3854 neighbors (out of 6) | 2.665 | Hexagonal CA steady-state mean green neighbors around green scales |
| adult green scale fraction | 0.4846 fraction | 0.44 | Mean binary state of hexagonal CA grid at step 40 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2015-manukyan-ocellated-lizard-ca.py` prior to Step 12 remediation in `2026-10-19-ocellated-lizard-ca-loop`.
