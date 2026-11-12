# MorphoEval Full-Loop Telemetry — `2026-10-27-bacillus-subtilis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-bacillus-subtilis-turing-wavelength-s0`
- **Primary paper / dataset:** Bacillus subtilis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.08 s** (0.02 min) — Tier A: `0.54 s` · Tier B: `0.54 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.08** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Outcome

Pass 2 meets the gates. Pass 3 was not run.

- Audit v1 overall 84.4. Audit v2 overall 90.1 (`supersedes` the v1 audit).
- image_fitness 89.2, cv_sim_similarity 89.4, GIF 12 frames at 520 px.
- Pearson r 0.99998 on the Turing band. Bulk radial wavelength 16.00 µm.
- Specimen ridge count 43 versus model circumference mode 72 (not a self-copy).
- Scale 16 µm, age 0 Ma, Bacillus chronoseries 1/1. Wavenumber versus the 10 µm tier center is 0.624.
- Signing: `~/.nanopub/profile.yml` is absent, so nanopubs were not signed and pass 2 has no `published.json` trusty URIs to supersede. Both KG builds passed the offline conformance check.
- The step table above records the telemetry logger, not the wall clock of the model runs. The assessor of record in the audits is Grok 4.7.

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 2 — `paper` (Paper)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 3 — `source` (Formal source)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 4 — `rules` (Pin the rules)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 5 — `script` (Script)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 6 — `output` (Output & compare)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 8 — `measure` (Measure)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 9 — `build` (Write as claims)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 10 — `publish` (Sign & verify)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 11 — `graph` (Public record)
- **Notes:** loop 3130; loop 3130 recorded after the step

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** loop 3130; loop 3130 recorded after the step
