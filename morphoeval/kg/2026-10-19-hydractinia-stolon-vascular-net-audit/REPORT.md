# MorphoEval report — audit `2026-10-19-hydractinia-stolon-vascular-net-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/100

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. & Blackstone (1999) Syst. Biol. colonial hydroid pulsatile gastrovascular stolon network, anastomosis loop closure, and runner-to-sheet morphospace (`2020-badin-hydractinia-stolon-vascular-net.py`) | 54 | 44 | 49 | 51 | 68 | 55 | 53 | 52.3 | 46.6 | 58.5 | 51.9 | **51.9** | partially faithful |

## Findings and remediation

### Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. & Blackstone (1999) Syst. Biol. colonial hydroid pulsatile gastrovascular stolon network, anastomosis loop closure, and runner-to-sheet morphospace  —  overall 51.9 (partially faithful)

Pass 1 (v1) baseline evaluation of `2020-badin-hydractinia-stolon-vascular-net.py` (weighted score: 53.0/100). The Pass 1 baseline integrates only a single-stolon lumen radius ODE at two pump gain values without solving the multi-node Kirchhoff gastrovascular pressure network driven by active polyp contraction oscillators V_p(t) = V_0 + Delta_V sin(omega_p t + phi_p), omits spatial stolon tip outgrowth, tip-to-stolon anastomosis loop closure (b_1 = E - V + 1), and low-shear stolon regression across the Blackstone (1999) / Buss (2015) runner-to-sheet continuum (Podocoryna runner, 2,4-DNP uncoupled shift, Hydractinia web, Hydractinia dense sheet, and polyp-excision / MgCl2 paralysis control), and does not segment the vendored Hydractinia symbiolongicarpus colony micrograph.

- **Model fidelity to source: 54** — Integrates a single-edge lumen radius ODE under prescribed sinusoidal pressure (L25-63), but omits the multi-node Kirchhoff Laplacian pressure solver driven by pulsatile polyp contraction oscillators, spatial stolon tip outgrowth, tip-to-stolon anastomosis loop closure (b_1 = E - V + 1), and low-shear regression.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L25–63
- **Morphological realism / specimen match: 44** — Vendors `datasets/hydractinia_stolon_vascular_net/hydractinia_symbiolongicarpus_specimen.jpg` on disk, but Pass 1 does not load, segment, or extract colony tissue coverage, gastrovascular polyp pump cavities, planar meshedness alpha, or Euler loop count b_1.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L90–134
- **Parameter expressivity: 49** — Evaluates only two scalar pump gains (0.55 and 1.65) in a single-tube ODE without sweeping the runner-to-sheet morphospace continuum, 2,4-DNP redox perturbation, or MgCl2 polyp paralysis regression control.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L66–87
- **Morphometric validation: 51** — Exposes only 2 scalar measurements in `measure_morphoeval()` without planar meshedness alpha = (E - V + 1)/(2V - 5), cyclomatic loop count b_1, stolon mat area coverage, Pearson correlation p-values, or specimen CV morphometrics.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L66–87
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `hydractinia_stolon_vascular_net_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/hydractinia_stolon_vascular_net/hydractinia_stolon_vascular_net_reference.json` L1–175
- **Code-to-model traceability: 55** — Module docstring cites Buss et al. (2015) and Dudgeon et al. (1999), but lacks explicit equation-to-code and Lean 4 theorem mappings.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L1–9
- **Reproducibility package: 53** — Includes a minimal 1-assertion `--self-test` flag, but lacks SHA-256 dataset integrity checks, Kirchhoff conservation checks, and polyp-excision negative ablation controls.  
  evidence: `2020-badin-hydractinia-stolon-vascular-net.py` L137–141

Remediation:
- Implement the coupled pulsatile polyp oscillator + multi-node Hagen-Poiseuille Kirchhoff gastrovascular pressure solver sum_j G_ij*(p_i(t) - p_j(t)) = -dV_p/dt with quartic lumen conductance G_ij = pi*r_ij^4/(8*mu*L_ij) and wall shear stress tau_w = 4*mu*|Q_ij|/(pi*r_ij^3).
- Simulate spatial stolon tip outgrowth, tip-to-stolon anastomosis loop closure (b_1 = E - V + 1), and low-shear stolon regression across the 5 canonical Blackstone (1999) / Buss (2015) morphospace regimes (Podocoryna runner, 2,4-DNP redox uncoupling shift, Hydractinia intermediate web, Hydractinia dense stolonal sheet, and polyp-excision / MgCl2 contraction arrest control) plus an 8-point pump-amplitude sweep.
- Segment the vendored Yale Peabody Museum Hydractinia symbiolongicarpus colony micrograph (`hydractinia_symbiolongicarpus_specimen.jpg`, YPM IZ 080458) to extract coenosarc mat area coverage, endodermal gastrovascular polyp pump centers, planar meshedness alpha, and Euler cyclomatic loops b_1, backed by runtime SHA-256 verification and an 8-suite `--self-test`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Buss, Anderson & Perry (2015) Biol. Bull., Dudgeon et al. (1999) Am. Nat. & Blackstone (1999) Syst. Biol. colonial hydroid pulsatile gastrovascular stolon network, anastomosis loop closure, and runner-to-sheet morphospace
- **spec_fingerprint** (semantic model + inputs): `89c0f33a7d88261f666404a05447dcbcee682836ef33db2cdc94324f94c96aa2`
- **input_dataset_sha256**: `datasets/hydractinia_stolon_vascular_net/hydractinia_stolon_vascular_net_reference.json` (`db4c51ae3bdeb00a…`), `datasets/hydractinia_stolon_vascular_net/hydractinia_symbiolongicarpus_specimen.jpg` (`059b83cd797d0516…`), `datasets/hydractinia_stolon_vascular_net/paper_citations.bib` (`44b19ab180b9e27a…`), `datasets/hydractinia_stolon_vascular_net/scout_manifest.json` (`626e17690a5a1cd7…`)
- `docs/media/hydractinia_stolon_vascular_net_comparison.png` — artifact_sha256 `08ee845e9749730b…`, 100,328 B, 1200×504 px

| measurement | value | reference | method |
|---|---:|---|---|
| Runner regime (eta = 0.55) steady-state stolon lumen radius | 10.5287 um | 13.4 | Single-stolon ODE at pump gain eta = 0.55 |
| Sheet regime (eta = 1.65) steady-state stolon lumen radius | 19.1574 um | 26.6 | Single-stolon ODE at pump gain eta = 1.65 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2020-badin-hydractinia-stolon-vascular-net.py` prior to Step 12 closed-loop remediation.
