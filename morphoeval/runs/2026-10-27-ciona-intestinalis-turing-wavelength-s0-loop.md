# MorphoEval Full-Loop Telemetry — `2026-10-27-ciona-intestinalis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-ciona-intestinalis-turing-wavelength-s0`
- **Primary paper / dataset:** Ciona intestinalis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **248.00 s** (4.13 min) — Tier A: `170.00 s` · Tier B: `78.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 10.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 8.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **248.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Commons scout. Slot 3148 is outside campaign_100_slots.yaml. No collision with ciona-intestinalis-turing-wavelength-s0.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, README. Scale 5.056e-05 m (51 um label), age 0 Ma, chronoseries 1/1.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored YPM IZ 082346 CC0 JPEG by Eric A. Lazo-Wasem. SHA-256 bacc05b7298b188c.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3148: trace, det, cross, discriminant, equal-diffusion ablation, scale sandwich. 0 sorry.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg spectral solver, --self-test, measure_morphoeval, alpha sweep, KS, ablations.

### Step 6 — `output` (Output & compare)
- **Notes:** self-test passed. GIF 12 frames 560x779. Comparison PNG. Pearson r 0.9994. Wavelength 50.05 um.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit overall 82.2. Remediation R1 fitness JSON, R2 dispersion zoom, R3 adult spacing control.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100. Ten non-circular measurements.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for pass 1 and pass 2. Rubric not re-emitted; supersedes bos-v7 rubric URI.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. nanopub module is not installed and ~/.nanopub/profile.yml is absent. No trusty URI minted.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for both audits. Pass 2 overall 89.0 faithful.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R3. image_fitness 92.4, CV 96, Pearson 0.9994, 12 frames, 560 px, 10 non-circular metrics. Gates pass. No pass 3.
