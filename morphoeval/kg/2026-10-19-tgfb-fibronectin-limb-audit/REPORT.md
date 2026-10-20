# MorphoEval report — audit `2026-10-19-tgfb-fibronectin-limb-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hentschel, Glimm, Glazier & Newman (2004) Proc. R. Soc. B / Kiskowski et al. (2004) Dev. Biol. multiscale TGF-beta / FGF-inhibitor reaction-diffusion and fibronectin haptotaxis PDE in developing Gallus gallus limb bud (`2004-hentschel-tgfb-fibronectin-limb.py`) | 49 | 28 | 45 | 42 | 60 | 58 | 68 | 47.7 | 33.2 | 62.7 | 46.4 | **46.4** | partially faithful |

## Findings and remediation

### Hentschel, Glimm, Glazier & Newman (2004) Proc. R. Soc. B / Kiskowski et al. (2004) Dev. Biol. multiscale TGF-beta / FGF-inhibitor reaction-diffusion and fibronectin haptotaxis PDE in developing Gallus gallus limb bud  —  overall 46.4 (partially faithful)

Pass 1 (v1) baseline evaluation of `2004-hentschel-tgfb-fibronectin-limb.py` (overall weighted score: 48.2 / 100). The v1 baseline script evaluates the analytical TGF-beta / inhibitor wavenumber formula `m = sqrt(fa - 2)` for stylopod (`m=1`) and wing autopod (`m=3`), but approximates fibronectin and mesenchymal cell density via static algebraic scaling of a cosine wave rather than numerically integrating the coupled 4-field `(c_a, c_i, F, R)` reaction-diffusion-haptotaxis PDE (`dR/dt = D_R*Lap(R) - chi_F*div(R*grad(F)) + r_R*R*(1-R)`); omits the 2D composite proximodistal wing bud solver, the 2D micromass nodular-spot vs TGF-beta2-stripe morphospace (Miura & Shiota 2000; Kiskowski et al. 2004), and anti-fibronectin (`chi_F=0`) / equal-diffusion (`Da=Di`) negative controls; does not load or segment the vendored Gallus gallus limb specimen; and returns only 2 analytical scalar endpoints in `measure_morphoeval()`.

- **Model fidelity to source: 49** — Evaluates the analytical TGF-beta / inhibitor wavenumber `compute_hentschel_dispersion`, but replaces the coupled 4-field `(c_a, c_i, F, R)` reaction-diffusion-haptotaxis PDE and 2D proximodistal wing bud solver with a static trigonometric profile without integrating `dF/dt = k_F*c_a^+*R_2 - gamma_F*F` or `dR/dt = D_R*Lap(R) - chi_F*div(R*grad(F)) + r_R*R*(1-R)`.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L25–35
- **Morphological realism / specimen match: 28** — Although `datasets/tgfb_fibronectin_limb/chick_limb_chondrogenesis_specimen.jpg` is vendored on disk, the v1 baseline script never loads, segments, or plots the Gallus gallus forelimb specimen and performs no quantitative comparison of stylopod (1 humerus), zeugopod (2 radius+ulna), and autopod (3 digits) bone profiles against the haptotactic PDE.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L40–68
- **Parameter expressivity: 45** — Evaluates only 3 static mode numbers without simulating the full 4-stage AER-FGF zonation series (`m = 1 -> 2 -> 3 -> 4`), omits the 2D micromass nodular-spot vs exogenous TGF-beta2 stripe morphospace (Miura & Shiota 2000; Kiskowski et al. 2004), omits anti-fibronectin (`chi_F = 0`) and equal-diffusion (`Da = Di`) negative controls, and exposes only `--self-test` on the CLI.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L40–68
- **Morphometric validation: 42** — `measure_morphoeval()` returns only 2 analytical scalar endpoints without measuring 4-field haptotactic cell density peak counts, F-R colocalization correlation, micromass spot/stripe area fractions, negative-control quenching, or segmented Gallus gallus wing bone counts and spacing ratios.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L70–92
- **Source and data traceability: 60** — Header cites Hentschel et al. (2004) DOI `10.1098/rspb.2004.2772`, and `datasets/tgfb_fibronectin_limb/` is present on disk, but the v1 script does not load `tgfb_fibronectin_limb_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L1–10
- **Code-to-model traceability: 58** — Basic docstrings are present, but functions lack explicit equation mappings to Hentschel et al. (2004) Eqs. (2.1)-(3.14), Kiskowski et al. (2004) haptotaxis kinetics, and the 3 Lean 4 theorems in `Loop_014_tgfb_fibronectin_limb.lean`.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L25–38
- **Reproducibility package: 68** — Includes `--self-test`, but `self_test()` checks only 2 scalar equalities without verifying dataset SHA-256 locks, 4-field PDE cell condensation counts across all 4 FGF zones (`[1, 2, 3, 4]`), 2D wing bud `[1, 2, 3]` compartments, 2D micromass spot/stripe area fractions, negative-control quenching, specimen bone segmentation, or bitwise two-run determinism.  
  evidence: `2004-hentschel-tgfb-fibronectin-limb.py` L94–100

Remediation:
- Replace the static cosine profile with a genuine 4-field coupled reaction-diffusion-haptotaxis PDE solver integrating TGF-beta (c_a), FGF-dependent inhibitor (c_i), ECM fibronectin (F), and haptotactic mesenchymal cell density (R) via DCT-II IMEX + conservative haptotactic flux discretization in 1D and 2D.
- Simulate all 4 proximodistal AER-FGF zonation regimes from Hentschel et al. (2004) (HH20 stylopod fa=3 -> 1 humerus; HH24 zeugopod fa=6 -> 2 radius+ulna; HH27 wing autopod fa=11 -> 3 digits II-IV; HH28 leg/high-TGFb fa=18 -> 4 pedal digits), the 2D micromass nodular-spot vs exogenous TGF-beta2 stripe morphospace (Kiskowski et al. 2004; Miura & Shiota 2000), and dual negative controls (equal diffusion Da=Di=0.5 and anti-fibronectin no-haptotaxis chi_F=0).
- Load and SHA-256-verify all 4 files in `datasets/tgfb_fibronectin_limb/` at runtime; segment the humerus (1 bone), radius+ulna (2 bones), and digits II-IV (3 elements) from `chick_limb_chondrogenesis_specimen.jpg`; and compare specimen bone counts and spacing ratios against the 4-field PDE.
- Expose full CLI parameters (--fa, --L-AP-um, --Da, --Di, --chi-F, --k-F, --steps, --seed), expand `measure_morphoeval()` to 8 reference-backed morphometric measurements, and render the 8-panel comparison PNG and 4-frame growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hentschel, Glimm, Glazier & Newman (2004) Proc. R. Soc. B / Kiskowski et al. (2004) Dev. Biol. multiscale TGF-beta / FGF-inhibitor reaction-diffusion and fibronectin haptotaxis PDE in developing Gallus gallus limb bud
- **spec_fingerprint** (semantic model + inputs): `a2738dce4265d3cbf1985dad5976f0edb36bd4a8ed6349bde78d12e6cc7cd510`
- **input_dataset_sha256**: `datasets/tgfb_fibronectin_limb/chick_limb_chondrogenesis_specimen.jpg` (`771e26f9fe8c813d…`), `datasets/tgfb_fibronectin_limb/scout_manifest.json` (`c810bd77e66557c8…`), `datasets/tgfb_fibronectin_limb/tgfb_fibronectin_limb_reference.json` (`33d554e44b712e2b…`)
- `docs/media/tgfb_fibronectin_limb_comparison.png` — artifact_sha256 `eea82e7f7885e55a…`, 719,240 B, 2230×1349 px

| measurement | value | reference | method |
|---|---:|---|---|
| HH20 stylopod haptotactic cell condensation count m(fa=3.0) | 1.0 elements | 1.0 | coupled 4-field (c_a, c_i, F, R) Hentschel et al. (2004) PDE mesenchymal cell density R(x) peak count at fa=3.0 (c_FGF=0.25) *(wavelength lambda_AP = 1150.0 um, F-R colocalization r = 0.9617, specimen stylopod bones = 1)* |
| HH24 zeugopod haptotactic cell condensation count m(fa=6.0) | 2.0 elements | 2.0 | coupled 4-field (c_a, c_i, F, R) Hentschel et al. (2004) PDE mesenchymal cell density R(x) peak count at fa=6.0 (c_FGF=0.55) *(wavelength lambda_AP = 575.0 um, PDE spacing = 577.3 um, specimen zeugopod bones = 2)* |
| HH27 chick wing autopod haptotactic cell condensation count m(fa=11.0) | 3.0 elements | 3.0 | coupled 4-field (c_a, c_i, F, R) Hentschel et al. (2004) PDE mesenchymal cell density R(x) peak count at fa=11.0 (c_FGF=0.85) *(wavelength lambda_AP = 383.3 um, PDE spacing = 383.3 um, specimen autopod digits = 3)* |
| HH28 hindlimb / high-TGF-beta autopod condensation count m(fa=18.0) | 4.0 elements | 4.0 | coupled 4-field (c_a, c_i, F, R) Hentschel et al. (2004) PDE mesenchymal cell density R(x) peak count at fa=18.0 (c_FGF=1.00) *(wavelength lambda_AP = 312.5 um, PDE spacing = 313.7 um)* |
| 2D composite chick wing bud total haptotactic skeletal elements (1 + 2 + 3) | 6.0 elements | 6.0 | 2D proximodistal multiscale (c_a, c_i, F, R) PDE cumulative cell condensation peak count across stylopod, zeugopod, and autopod zones *(compartment counts = [1, 2, 3] vs specimen [1, 2, 3])* |
| fibronectin F(x) vs mesenchymal cell density R(x) haptotactic colocalization Pearson r | 0.9997 Pearson r | 0.98 | spatial correlation between ECM fibronectin F(x) and haptotactic mesenchymal cell density R(x) in HH27 wing autopod *(micromass WT nodular area fraction = 0.198 vs +TGF-beta2 stripe area fraction = 0.448)* |
| anti-fibronectin no-haptotaxis (chi_F=0) and equal-diffusion (Da=Di) negative control cell amplitude | 0.0 std(R) control | 0.0 | mesenchymal cell density std(R) when haptotaxis is blocked (chi_F=0) or morphogen diffusivities are equal (Da=Di=0.5) vs WT autopod *(WT std(R) = 0.7574 (3 digits) vs chi_F=0 std(R) = 8.65e-11 (0 elements) and Da=Di std(R) = 8.65e-11 (0 elements))* |
| segmented Gallus gallus wing specimen zeugopod-to-autopod spacing ratio | 1.4074 dimensionless ratio | 1.4074 | ratio of segmented radius-ulna spacing (38.0 px) to mean autopod digit II-IV spacing (27.0 px) on datasets/tgfb_fibronectin_limb/chick_limb_chondrogenesis_specimen.jpg *(segmented specimen compartment bone counts = [1, 2, 3] (1 humerus -> 2 radius+ulna -> 3 digits II-IV, total = 6))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2004-hentschel-tgfb-fibronectin-limb.py` prior to Step 12 remediation in `2026-10-19-tgfb-fibronectin-limb-loop`.
