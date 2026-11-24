# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-suzukii-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-suzukii-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila suzukii Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.91 s** (0.02 min) — Tier A: `0.46 s` · Tier B: `0.45 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **0.91** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** slot 3207; slot 3207 recorded

### Step 2 — `paper` (Paper)
- **Notes:** slot 3207; slot 3207 recorded

### Step 3 — `source` (Formal source)
- **Notes:** slot 3207; slot 3207 recorded

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3207; slot 3207 recorded

### Step 5 — `script` (Script)
- **Notes:** slot 3207; slot 3207 recorded

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3207; slot 3207 recorded

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3207; slot 3207 recorded

### Step 8 — `measure` (Measure)
- **Notes:** slot 3207; slot 3207 recorded

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3207; slot 3207 recorded

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3207; slot 3207 recorded

### Step 11 — `graph` (Public record)
- **Notes:** slot 3207; slot 3207 recorded

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3207; slot 3207 recorded

## Pass 2 gate results

The tracker table above is the bookkeeping close-out. The simulation itself is `models/2026-delphi10000-drosophila-suzukii-turing-wavelength-s0.py` (seed 3207). Pass 2 audit overall is **87.2**.

| Gate | Result |
|---|---|
| image_fitness >= 70 | 90.6 |
| cv_sim_similarity >= 58 | 96.0 |
| GIF frames >= 6 (rendered 12) | 12 |
| resolution >= 340 px | 960×640 |
| non-circular specimen metric | yes (Otsu foreground 0.203 vs prior 0.40, and 11 other value≠reference rows) |
| Pearson r > 0.98 | 0.9999999995 (RMSE 4.16e-4) |
| audit overall >= 85 | 87.2 |

No Pass 3. The noise-seeded labyrinth spacing KS stays 0.63; the stripe seed locks to 2.046 µm (CV 0.056). That split is reported, not adjusted.

Signing: `~/.nanopub/profile.yml` is absent, so the nanopubs were not signed and `check_signed.mjs` was not run. Both knowledge graphs passed `build_morphoeval_kg.py --conform`.

Scale is 2.046e-6 m (2.0 µm) and the age is 0 Ma. Predecessor *Drosophila subobscura* has f_u < 0 (no Turing band). Successor *Drosophila takahashii* has onset wavelength 2.4552 µm.
