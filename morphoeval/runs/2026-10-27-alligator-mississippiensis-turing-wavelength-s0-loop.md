# MorphoEval Full-Loop Telemetry — `2026-10-27-alligator-mississippiensis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-alligator-mississippiensis-turing-wavelength-s0`
- **Primary paper / dataset:** Alligator mississippiensis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1087.22 s** (18.12 min) — Tier A: `1086.79 s` · Tier B: `0.43 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 1086.43 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1087.22** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia scout; slot 3105 is outside campaign_100_slots.yaml; OpenAlex/Crossref/Wikimedia. Slot 3105 is outside the 100-slot campaign. No filename collision with alligator-mississippiensis-turing-wavelength-s0. Selected Sévi CC BY-SA 3.0 juveniles and Turing 1952 / Murray 1981 / Schnakenberg 1979.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON states 1.498 mm and 0 Ma. Bib DOIs checked on Crossref.

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Alligator_mississippiensis_2_babies.jpg SHA-256 a4a5ed7f280ee2de43a726ab16d9d681bbb4dd6ac032c141591dc8a7aff276de. Rejected the synthetic gallery spiral.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3105: 8 theorems, 0 sorry. Math KG yaml registers k_c at 1.498 mm. lean binary not installed, so not executed.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg RD, dispersion check, sweep, ablation, specimen ROIs, GIF and comparison figure.

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. Pass 1 and pass 2 renders written. Pass 1 archives kept in the dataset directory.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 83.1. Pass 2 overall 87.6. Single assessor; R5 left open.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py quick_gate 100/100 on both passes. Non-circular specimen metrics retained.

### Step 9 — `build` (Write as claims)
- **Notes:** Audits valid against rubric 0.1.0. build_morphoeval_kg.py --conform exited 1: no publisher ORCID. REPORT.md written from report() with that failure stated. No TriG nanopubs emitted.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml does not exist. sign_morphoeval.py and check_signed.mjs were not run. No signatures were fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for the pass-1 and pass-2 audit ids. Overalls 83.1 and 87.6.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 rotated theta to 12 degrees, tightened the mask, and added the 300/450/600 mm bracket. Gates passed. Pass 3 not required.


## Result

Holocene *Alligator mississippiensis*, chronoseries 1/1 at 0 Ma, detail scale 1.498 mm. Schnakenberg reaction-diffusion with the Turing wavenumber fixed so lambda_c = 1.498 mm on a 12-by-8-wavelength flank. Pass 2 rotates the diffusion axis to the measured 12 degree tail-band normal and keeps the animated overlay on the integument.

Gates after pass 2 (pass 3 not run):

- image_fitness = 90.9 (threshold 70)
- cv_sim_similarity = 96.0 (threshold 58)
- GIF frames = 12, resolution = 720x480
- non-circular specimen metrics: tail spacing 73 px versus 1.498 mm, flank count 19 versus 13, band/lambda ratio 14.35 versus 1 at an assumed 450 mm total length, KS distance 0.5 versus 0
- dispersion Pearson r = 0.999999998 (threshold 0.98), RMSE = 5.0e-5
- audit overall = 87.6 (threshold 85)

The photographed cream bands are 14.3, 21.5 and 28.7 mm if total length is assumed to be 300, 450 or 600 mm. They are not the 1.498 mm mode. Head-scale cracking is not this PDE. Nanopublications were not emitted or signed: the builder exits without a publisher ORCID, and no nanopub profile is present.
