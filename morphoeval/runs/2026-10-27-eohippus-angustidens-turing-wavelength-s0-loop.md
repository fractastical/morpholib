# MorphoEval Full-Loop Telemetry — `2026-10-27-eohippus-angustidens-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-eohippus-angustidens-turing-wavelength-s0`
- **Primary paper / dataset:** Eohippus angustidens Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **972.34 s** (16.21 min) — Tier A: `971.81 s` · Tier B: `0.54 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **997** (input: `0`, output: `997`, subagent: `0`) — Tier A: `997` · Tier B: `0`
- **Total artifact tokens produced:** `997`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 971.36 | 0.00 | 0 | 997 | 0 | **997** | 997 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **972.34** | **0.00** | **0** | **997** | **0** | **997** | **997** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and collision check for slot 3219; OpenAlex DOIs resolved; Commons empty; no slug collision with the 3.3 mm sibling
- **Artifact:** `datasets/eohippus_angustidens_turing_wavelength_s0/scout_manifest.json` (3,551 B, `997` tok, SHA-256 `543e8435e637b43f…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg/Turing/Murray equations, 78 µm and 52.0 Ma

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery specimen and pinned SHA-256

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3219 theorems and math-kg yaml

### Step 5 — `script` (Script)
- **Notes:** Pass 1 model, then pass 2 aligned lattice, weight table, second seed

### Step 6 — `output` (Output & compare)
- **Notes:** self-test r=0.99712; GIF 640x480 x12; CV 88.8

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.1; pass 2 overall 86.9

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100/100, 13 non-circular measurements

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK for audit and audit-v2

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml absent; nanopubs not signed

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written for both audit directories

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** R1-R4 closed; gates pass; pass 3 not required

## Pass 2 gates

- image_fitness = 86.8 (0.55×88.8 + 0.30×81 + 0.15×91)
- cv_sim_similarity = 88.8
- GIF = 12 frames, 640×480
- Pearson r (dispersion) = 0.99712
- Audit v2 overall = 86.9
- Non-circular specimen metrics: 13 (value ≠ reference)
- Pass 3: not required
- Signing: unavailable (`~/.nanopub/profile.yml` absent). Nanopubs were built and conformance-checked, not signed.
- Second assessor: not run. Scores are single-assessor (grok-4.7).
