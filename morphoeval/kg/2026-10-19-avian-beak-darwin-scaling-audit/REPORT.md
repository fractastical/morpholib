# MorphoEval report — audit `2026-10-19-avian-beak-darwin-scaling-audit`

Rubric **0.1.0** · checkpoint `33e07ad` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/70

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Campas et al. (2010) PNAS affine scaling-and-shear transformation + Abzhanov et al. (2004, 2006) Bmp4/Calmodulin beak morphogenesis (`2021-alberti-avian-beak-darwin-scaling.py`) | 58 | 36 | 48 | 56 | 74 | 68 | 68 | 54.7 | 43.5 | 70.0 | 54.8 | **54.8** | partially faithful |

## Findings and remediation

### Campas et al. (2010) PNAS affine scaling-and-shear transformation + Abzhanov et al. (2004, 2006) Bmp4/Calmodulin beak morphogenesis  —  overall 54.8 (partially faithful)

Pass 1 (v1) baseline evaluation of `2021-alberti-avian-beak-darwin-scaling.py` (overall weighted score: 56.6/100). The v1 script implements pure diagonal scaling (s_L, s_D) for the 6 Group A Geospiza species (RMS = 0.0040), but omits the full 3-parameter Campas-Brenner affine scaling-and-shear transformation M(s_L, s_D, gamma) needed to collapse all 14 Darwin's finch species across Groups A, B, and C, omits Abzhanov et al.'s (2004, 2006) RCAS::Bmp4, RCAS::Noggin, and RCAS::CaM perturbations and pure-scaling negative control, and does not segment the vendored 1845 Gould Darwin's finches specimen image.

- **Model fidelity to source: 58** — Implements canonical power-law culmen profile y_0(x_0) and pure scaling (s_L, s_D) for Group A Geospiza species (L34-53), backed by 3 Lean 4 theorems in `Loop_070_avian_beak_darwin_scaling.lean`. However, it omits the shear matrix [[1, gamma], [0, 1]] required for Groups B and C, lacks the 2D/3D Bmp4/Calmodulin mesenchymal growth field and cantilever bending mechanics I_beam = W*D^3/(12*L), and lacks the pure-scaling negative control on Certhidea and Camarhynchus.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L34–53; `2021-alberti-avian-beak-darwin-scaling.py` L56–97
- **Morphological realism / specimen match: 36** — Although `datasets/avian_beak_darwin_scaling/darwin_finches_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or extracts culmen contours from the 4 Gould (1845) finch heads, and evaluates only 6 of the 14 Darwin's finch species.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L56–97; `docs/media/avian_beak_darwin_scaling_comparison.png` L0–0
- **Parameter expressivity: 48** — Evaluates 6 Geospiza species in (s_L, s_D) space (L56-97), but omits the 8 species in Groups B and C with non-zero shear gamma in [-0.28, +0.34], omits RCAS::Bmp4, RCAS::Noggin, and RCAS::CaM perturbations, and exposes no CLI parameter overrides (L208-221).  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L208–221
- **Morphometric validation: 56** — `measure_morphoeval()` (L169-198) reports Group A pure scaling RMS and Geospiza D/L correlation, but does not measure the 14-species affine scaling-and-shear collapse RMS vs pure-scaling negative control RMS, Bmp4/CaM perturbation ratios, cantilever bite-force scaling, or computer-vision culmen contour metrics from the specimen plate.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L169–198
- **Source and data traceability: 74** — Reads `datasets/avian_beak_darwin_scaling/avian_beak_darwin_scaling_reference.json` and cites Campas et al. (2010) `10.1073/pnas.0911575107` (L4-9), but does not verify SHA-256 digests of the reference JSON, BibTeX, scout manifest, or specimen image at runtime.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L30–31
- **Code-to-model traceability: 68** — Functions document the Group A pure scaling curve (L34-53), but lack explicit mappings to Campas et al. (2010) Eqs. (1)-(3) & Figs. 1-4, Abzhanov et al. (2004, 2006) Science/Nature Figs. 2-4, and the 3 Lean 4 theorems in `Loop_070_avian_beak_darwin_scaling.lean`.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L34–53
- **Reproducibility package: 68** — Includes `--self-test`, `--verify-only`, and `--skip-gif` flags (L200-221), but `self_test()` performs only 3 basic assertions without checking SHA-256 dataset integrity, 14-species shear collapse vs pure-scaling failure, Bmp4/CaM perturbation ordering, or two-run bitwise determinism.  
  evidence: `2021-alberti-avian-beak-darwin-scaling.py` L200–206

Remediation:
- Implement the full 3-parameter Campas-Brenner affine scaling-and-shear group M(s_L, s_D, gamma) = [[1, gamma], [0, 1]] @ [[s_L, 0], [0, s_D]] across all 14 Darwin's finch species (Groups A, B, and C), demonstrating both the 6-species Group A pure-scaling collapse and the 14-species universal scaling+shear collapse (RMS < 0.015, r > 0.99) contrasted against the pure-scaling (gamma = 0) negative control on Certhidea and Camarhynchus.
- Implement the 2D/3D Bmp4-Calmodulin mesenchymal growth field, cantilever beam bending mechanics I_beam = W*D^3/(12*L), and Abzhanov et al. (2004, 2006) retroviral perturbation suite (RCAS::Bmp4, RCAS::Noggin, RCAS::CaM).
- Implement computer-vision contour segmentation on `datasets/avian_beak_darwin_scaling/darwin_finches_specimen.jpg` (Gould's 1845 Darwin's finches lithograph) to extract upper culmen profiles, aspect ratios D/L, and affine collapse residuals across the 4 illustrated finch species.
- Add runtime SHA-256 verification of all 4 files in `datasets/avian_beak_darwin_scaling/`, expose CLI parameter overrides (`--bmp4-scale`, `--cam-scale`, `--gamma-shear`, `--seed`), and expand `self_test()` and `measure_morphoeval()`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Campas et al. (2010) PNAS affine scaling-and-shear transformation + Abzhanov et al. (2004, 2006) Bmp4/Calmodulin beak morphogenesis
- **spec_fingerprint** (semantic model + inputs): `c3ac286b6e1240143c8b941b9ff61a0744669a4b95cb14db7918d73b5259a735`
- **input_dataset_sha256**: `datasets/avian_beak_darwin_scaling/avian_beak_darwin_scaling_reference.json` (`cf6430de0c52d725…`), `datasets/avian_beak_darwin_scaling/darwin_finches_specimen.jpg` (`5ea7ada14a01ba77…`), `datasets/avian_beak_darwin_scaling/paper_citations.bib` (`f0b9fb929ffd2d09…`), `datasets/avian_beak_darwin_scaling/scout_manifest.json` (`4b321120fd7b2736…`)
- `docs/media/avian_beak_darwin_scaling_comparison.png` — artifact_sha256 `777aca548970785f…`, 197,687 B, 1560×1105 px

| measurement | value | reference | method |
|---|---:|---|---|
| Group A Geospiza pure scaling collapse RMS residual | 0.004 dimensionless RMS | 0.008 | RMS residual across 6 Geospiza species after pure scaling (s_L, s_D) *(n=6 Geospiza species, r_DL=0.9998 (p=4.91e-08))* |
| Geospiza beak aspect ratio D/L correlation | 0.9998 Pearson r | 1.0 | Pearson correlation between scaling ratio s_D/s_L and observed D/L across 6 Geospiza species (p=4.91e-08) *(p=4.91e-08)* |
| Geospiza magnirostris vs Geospiza fuliginosa depth scaling ratio | 2.21 ratio s_D | 2.2105 | Depth scale factor ratio between G. magnirostris and G. fuliginosa baseline *(s_D ratio vs observed D ratio)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2021-alberti-avian-beak-darwin-scaling.py` prior to Step 12 closed-loop remediation in `2026-10-19-avian-beak-darwin-scaling-loop`.
