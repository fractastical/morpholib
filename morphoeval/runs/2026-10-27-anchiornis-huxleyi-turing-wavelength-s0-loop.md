# MorphoEval Full-Loop Telemetry — `2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-anchiornis-huxleyi-turing-wavelength-s0`
- **Primary paper / dataset:** Anchiornis huxleyi Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **2.77 s** (0.05 min) — Tier A: `0.47 s` · Tier B: `2.30 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **18,609** (input: `0`, output: `18,609`, subagent: `0`) — Tier A: `9,304` · Tier B: `9,305`
- **Total artifact tokens produced:** `18,609`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 1,164 | 0 | **1,164** | 1,164 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `anchiornis_huxleyi_holotype.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 8,076 | 0 | **8,076** | 8,076 | `2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.07 | 0.00 | 0 | 3,562 | 0 | **3,562** | 3,562 | `2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 1.88 | 0.00 | 0 | 2,447 | 0 | **2,447** | 2,447 | `2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 3,296 | 0 | **3,296** | 3,296 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **2.77** | **0.00** | **0** | **18,609** | **0** | **18,609** | **18,609** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/anchiornis_huxleyi_turing_wavelength_s0/scout_manifest.json` (4,206 B, `1,164` tok, SHA-256 `38d5b8705d3e6756…`)

### Step 3 — `source` (Formal source)
- **Artifact:** `datasets/anchiornis_huxleyi_turing_wavelength_s0/anchiornis_huxleyi_holotype.jpg` (524,760 B, `64` tok, SHA-256 `4415de64f740a3c5…`)

### Step 5 — `script` (Script)
- **Artifact:** `models/2026-delphi10000-anchiornis-huxleyi-turing-wavelength-s0.py` (27,891 B, `8,076` tok, SHA-256 `1eabccb9d193d39f…`)

### Step 7 — `audit` (Dual-agent audit)
- **Artifact:** `morphoeval/audits/2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit.yaml` (13,969 B, `3,562` tok, SHA-256 `f43edf1bff63af80…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit.measurements.yaml` (8,770 B, `2,447` tok, SHA-256 `e81afb51b064a1d3…`)

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit-v2/REPORT.md` (12,009 B, `3,296` tok, SHA-256 `4fb9fbd252ee7641…`)

## Result (slot 3111)

*Anchiornis huxleyi*, Late Jurassic Tiaojishan Formation, **160.0 Ma**, chronoseries 1/1. Detail scale **2.0 nm** (2.0e-9 m). Schnakenberg parameters a=0.062, b=1.28, Du=0.72, Dv=31.5. The 29 nm envelope (slot 1229) is the coarser mode on the same horizon; there is no successor chronospecies. This run's assessor is grok-4.7. The header model names above are the tracker defaults, not a second assessor.

Pass 1 overall **83.9** (below the 85 gate). Pass 2 closed the three remediation items (second holotype ROI, equal-diffusion field in the figure, nonlinear wavelength archived) and scores **89.0**. Pass 3 was not run.

Gates after pass 2, all met:

- image_fitness **91.8** (threshold 70)
- cv_sim_similarity **96.0** (threshold 58), from `verify_image_fitness.compute_sim_vs_specimen_cv_similarity` on the GIF last frame versus the CC BY-SA holotype
- GIF **12** frames at **640×480**
- non-circular specimen metrics, including contour circularity 0.128 versus a circle at 1, and band-weighted wavelength 2.517 nm versus the analytical target 2.0 nm
- dispersion Pearson r **0.99999999**, RMSE 6.1e-5
- audit overall **89.0**

Signing was not done. `sign_morphoeval.py` exits on `ModuleNotFoundError: requests`, and `~/.nanopub/profile.yml` is absent, so no trusty URI was produced and `check_signed.mjs` was not run. Nanopubs in `morphoeval/kg/2026-10-27-anchiornis-huxleyi-turing-wavelength-s0-audit-v2/` are unsigned. Conformance (`--conform`) passed for pass 1 and pass 2.
