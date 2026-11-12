# MorphoEval Full-Loop Telemetry — `2026-10-27-baculites-compressus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-baculites-compressus-turing-wavelength-s0`
- **Primary paper / dataset:** Baculites compressus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **249.00 s** (4.15 min) — Tier A: `170.00 s` · Tier B: `79.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 12.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 10.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 2.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **249.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Crossref + Wikimedia; OpenAlex 429; no s0 collision; Crossref + Wikimedia; OpenAlex 429; no s0 collision

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README; 94.16 um and 73.5 Ma; Reference JSON, bib, README; 94.16 um and 73.5 Ma

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Kevmin CC BY-SA 3.0 Baculites compressus 01.jpg and pinned SHA-256; Vendored Kevmin CC BY-SA 3.0 Baculites compressus 01.jpg and pinned SHA-256

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3131 theorems and math-kg yaml; Lean Loop_3131 theorems and math-kg yaml

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg k_c model with self-test; Schnakenberg k_c model with self-test

### Step 6 — `output` (Output & compare)
- **Notes:** PNG, 12-frame 720x480 GIF, metrics JSON; PNG, 12-frame 720x480 GIF, metrics JSON

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 83.3; Pass 2 overall 86.1; Pass 1 overall 83.3; Pass 2 overall 86.1

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 90/100 both passes; measure_outputs quick_gate 90/100 both passes

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow graphs after publisher ORCID from the repo report; ProbKnow graphs after publisher ORCID from the repo report

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml; graphs were not signed; Signing unavailable: no ~/.nanopub/profile.yml; graphs were not signed

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written by build_morphoeval_kg.py; REPORT.md written by build_morphoeval_kg.py

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 timestep 0.06 and Bearpaw/Pierre caption; gates met, no Pass 3; Pass 2 timestep 0.06 and Bearpaw/Pierre caption; gates met, no Pass 3
