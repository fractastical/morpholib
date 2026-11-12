# MorphoEval Full-Loop Telemetry — `2026-10-27-aurelia-aurita-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-aurelia-aurita-turing-wavelength-s0`
- **Primary paper / dataset:** Aurelia aurita Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1342.60 s** (22.38 min) — Tier A: `1341.98 s` · Tier B: `0.61 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 1341.47 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.13 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1342.60** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex + Wikimedia queried separately; campaign slot 3129 is outside the 1-100 manifest; OpenAlex and Wikimedia queried. No collision with aurelia-aurita-turing-wavelength-s0. Sibling stub abrams-aurelia-aurita-turing-wavelength was not copied.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON states 201 um (0.0002006 m) and age 0 Ma. Schnakenberg PDE, Jacobian, and claim boundary recorded.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Martin Thoma CC0 File:Aurelia-aurita-3.jpg. SHA-256 f4f54dc98dc4d9b364c41744f74c8a6d97b9d4f1fdde43d396bb40a48cb6a31f.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3129: trace, det, Turing discriminant 22500>8000, equal diffusion fails, 201 um label, chronoseries 1/1. No sorry. lean binary not installed.

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script: Schnakenberg spectral integrator, self-test, GIF and PNG.

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 1 self-test r=0.9999. GIF 12 frames 720x540. cv_sim=96. Control panel autoscaled.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.2. Single assessor grok-4.7. Remediation R1-R5.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100. 11 measurements. Pearson r 0.9999.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for audit and audit-v2. Loci not verified: origin is not fractastical/morpholib.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml is absent. No signature was fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 85.0 faithful on the v2 audit. Supersession chain recorded.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2: shared colour scale, 8 px bell body (circularity 0.462), gonad reference 4 with measured count 1. Gates: image_fitness 88.4, cv 96, frames 12, 540 px, pearson 0.9999, overall 85.0. Pass 3 not required.

## Gates after pass 2

All of these passed. Pass 3 was not run.

- image_fitness 88.4 (threshold 70)
- cv_sim_similarity 96.0 (threshold 58)
- GIF frames 12 (threshold 6), shorter side 540 px (threshold 340)
- non-circular specimen metrics include bell circularity 0.462 against 1, and gonad lobes 1 against 4
- dispersion Pearson r 0.9999 (threshold 0.98)
- audit overall 85.0 (threshold 85)

201 µm is the assigned detail scale, not a length measured on the Karlsruhe photograph. Nanopub signing was not available (`~/.nanopub/profile.yml` absent). The lobe counter did not recover four gonads.
