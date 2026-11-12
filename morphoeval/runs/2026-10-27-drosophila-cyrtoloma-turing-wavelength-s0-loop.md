# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-cyrtoloma-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-cyrtoloma-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila cyrtoloma Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.00 s** (0.00 min) — Tier A: `0.00 s` · Tier B: `0.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `skipped — no ~/.nanopub/profile.yml` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **0.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

Wall-clock cells above are the initializer's zeros. `track_loop.py start` rejected `--role`, so those calls did not record durations. The scientific results below are from the script, the two audits, and `verify_image_fitness.py`.

## Pass results

*Drosophila cyrtoloma*, detail-tier Turing carrier, **4.3 µm (4.315e-6 m, log10 = −5.365)**, **Holocene 0 Ma**, chronoseries 9/41. Predecessor *D. buzzatii*, successor *D. deflecta*. Kinetics are Schnakenberg (1979); the wavenumber is Turing's \(k_c\).

Pass 1 diagnostic integrated only to γt = 0.55. Pearson r on that sweep was **0.906**. Audit overall **79.2**. Remediations: develop the pattern, count stripes against domain length, run the equal-diffusion control, segment the plate, and simulate the two neighbours.

Pass 2 does that. Domain-length Pearson r = **0.9917**, RMSE = **0.549** stripes. Activator std 0.580 versus equal-diffusion std 0.0021. Simulated carriers: buzzatii 8.07 µm, cyrtoloma 4.315 µm, deflecta 6.83 µm (hypothesis 6.05 µm; order holds). Linear \(k_c\) on the calibrated grid is 3.70 µm, not forced equal to 4.315 µm. Audit overall **88.2**.

Image fitness (pass 2), using the verifier's weights on the 12-frame 640×480 GIF: CV similarity **92.8**, specimen-match 86, morphometric-validation 88, so image_fitness = **90.0**. Plate Otsu circularity 0.772 against a disk reference of 1.0.

Gates after pass 2: image_fitness ≥ 70, CV ≥ 58, frames ≥ 6, resolution ≥ 340 (rendered at 480), a non-circular specimen metric, Pearson r > 0.98, audit overall ≥ 85. All hold. Pass 3 was not run.

Signing: `~/.nanopub/profile.yml` is absent. Both knowledge graphs were built and passed `--conform`. They were not signed and not published.

The specimen file is a byte copy of the gallery plate. Wikimedia Commons had no bitmap for this species on 2026-10-27. DOI 10.11646/zootaxa.2400.1.1, used by slot 3061, is Zhu, Liu & Lis on *Macroscytus* and is not cited here.
