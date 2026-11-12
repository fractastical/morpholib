# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-heteroneura-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-heteroneura-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila heteroneura Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **809.43 s** (13.49 min) — Tier A: `808.99 s` · Tier B: `0.44 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 404.02 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 404.68 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **809.43** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3188 is outside campaign_100; OpenAlex/Crossref/Wikimedia query for heteroneura Turing wavelength; Campaign slot 3188 is absent; OpenAlex query used. No s0 filename collision. Commons returned no bitmap; gallery specimen is the fallback.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README state 5.2 µm and 0 Ma.; Equations, claim boundary, and SHA-256 pins written.

### Step 3 — `source` (Formal source)
- **Notes:** Gallery specimen vendored after Commons returned no bitmap.; drosophila_heteroneura_specimen.jpg SHA-256 23a06203ecdd717e393f6f664479e7d90ba52cd79865ddd32447db4e55f33402.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3188 and math KG yaml.; Four theorems, zero sorry. lean binary not on PATH; integer orderings checked by the Python self-test.

### Step 5 — `script` (Script)
- **Notes:** Pass 1 model with Schnakenberg dispersion, radial envelope, sweeps, ablation; Self-test passed. Schnakenberg lambda_c=5.222 µm, Pearson r=0.98482, CV=87.8, 12 frames at 800x600.

### Step 6 — `output` (Output & compare)
- **Notes:** Rendered comparison PNG and growth GIF.; docs/media/drosophila_heteroneura_turing_wavelength_s0_comparison.png and _growth.gif.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 seven-dimension audit.; Pass 1 overall 83.3 before remediation. Pass 2 overall 87.8.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py on v1 and v2.; 10 measurements, quick_gate 100/100. Values differ from references.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform.; ProbKnow conformance OK for both audits. Shared rubric not re-emitted (--rubric-from bos-taurus v7).

### Step 10 — `publish` (Sign & verify)
- **Notes:** sign_morphoeval.py if a nanopub profile exists.; Signing not performed. ~/.nanopub/profile.yml is absent. Nanopubs are unsigned. This is not a signed publication.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md for both audit KG directories.; v2 REPORT overall 87.8, verdict faithful.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 carrier and Lean mode-count alignment.; R1 and R2 resolved. R3 second assessor and R4 head width remain open. Gates passed; pass 3 not run.

## Pass 2 gates

All gates passed. Pass 3 was not run.

| Gate | Result |
|---|---|
| image_fitness >= 70 | 87.9 |
| cv_sim_similarity >= 58 | 87.8 |
| GIF frames >= 6 | 12 |
| resolution >= 340 px | 800×600 |
| non-circular specimen metric | stripe spacing 68.4685 µm vs reference 5.222 µm, plus Pearson, KS, coherency, and carrier period |
| Pearson r > 0.98 | 0.98482 |
| audit overall >= 85 | 87.8 (pass 2) |

Signing was not performed. `~/.nanopub/profile.yml` is absent, so the nanopubs are unsigned. Conformance was checked with `build_morphoeval_kg.py --conform`.
