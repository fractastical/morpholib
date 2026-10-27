# MorphoEval Full-Loop Telemetry — `2026-10-27-aulacopleura-koninckii-meraspis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-aulacopleura-koninckii-meraspis-turing-wavelength-s0`
- **Primary paper / dataset:** Aulacopleura koninckii_meraspis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.17 s** (0.02 min) — Tier A: `0.56 s` · Tier B: `0.61 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.17** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** slot 3126; slot 3126 recorded

### Step 2 — `paper` (Paper)
- **Notes:** slot 3126; slot 3126 recorded

### Step 3 — `source` (Formal source)
- **Notes:** slot 3126; slot 3126 recorded

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3126; slot 3126 recorded

### Step 5 — `script` (Script)
- **Notes:** slot 3126; slot 3126 recorded

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3126; slot 3126 recorded

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3126; slot 3126 recorded

### Step 8 — `measure` (Measure)
- **Notes:** slot 3126; slot 3126 recorded

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3126; slot 3126 recorded

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3126; slot 3126 recorded

### Step 11 — `graph` (Public record)
- **Notes:** slot 3126; slot 3126 recorded

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3126; slot 3126 recorded

## Result

Pass 2 overall is **87.7**. Pass 1 overall is **82.9**. No Pass 3.

Gates after Pass 2: image_fitness 91.3, cv_sim_similarity 95.7, GIF 16 frames at 720×560, Pearson r 0.99937, six specimen or dispersion metrics with value ≠ reference, audit overall 87.7.

The law is a near-onset Schnakenberg wavelength fixed at 3.125 µm (428.5 Ma, meraspid degree 12). Relative to successor holaspis18, k_c stays put and the thorax gains 6 segments (copy ratio 1.5). Counted furrows on the Commons photograph are 11, not 12.

Signing: `~/.nanopub/profile.yml` is absent, so the nanopubs were conformance-checked and left unsigned. Nothing was signed with a stand-in key.

Build used `--conform --no-verify-loci --rubric-from` the existing Bos taurus v7 graph so the shared rubric nanopub was not re-emitted, and so paths that do not yet exist on the parent commit were not treated as a failed pin.
