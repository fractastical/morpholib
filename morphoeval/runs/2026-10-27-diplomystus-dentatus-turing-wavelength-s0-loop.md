# MorphoEval Full-Loop Telemetry — `2026-10-27-diplomystus-dentatus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-diplomystus-dentatus-turing-wavelength-s0`
- **Primary paper / dataset:** Diplomystus dentatus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1229.22 s** (20.49 min) — Tier A: `1228.78 s` · Tier B: `0.44 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **1,970** (input: `0`, output: `1,970`, subagent: `0`) — Tier A: `1,970` · Tier B: `0`
- **Total artifact tokens produced:** `1,970`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 1228.40 | 0.00 | 0 | 1,970 | 0 | **1,970** | 1,970 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1229.22** | **0.00** | **0** | **1,970** | **0** | **1,970** | **1,970** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3170 is outside campaign_100_slots.yaml; custom OpenAlex/Crossref/Commons scout; Custom OpenAlex/Crossref/Commons scout; slot 3170 is outside campaign_100_slots.yaml. No collision with the 7.56 mm sibling.
- **Artifact:** `datasets/diplomystus_dentatus_turing_wavelength_s0/scout_manifest.json` (7,248 B, `1,970` tok, SHA-256 `a89dba7e88074788…`)

### Step 2 — `paper` (Paper)
- **Notes:** loop 3170 paper

### Step 3 — `source` (Formal source)
- **Notes:** loop 3170 source

### Step 4 — `rules` (Pin the rules)
- **Notes:** loop 3170 rules

### Step 5 — `script` (Script)
- **Notes:** loop 3170 script

### Step 6 — `output` (Output & compare)
- **Notes:** loop 3170 output

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** loop 3170 audit

### Step 8 — `measure` (Measure)
- **Notes:** loop 3170 measure

### Step 9 — `build` (Write as claims)
- **Notes:** loop 3170 build

### Step 10 — `publish` (Sign & verify)
- **Notes:** loop 3170 publish

### Step 11 — `graph` (Public record)
- **Notes:** loop 3170 graph

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** loop 3170 remed

## Slot 3170 result

Pass 2 gates passed. No Pass 3.

- image_fitness 89.7 (threshold 70)
- cv_sim_similarity 96.0 (threshold 58)
- GIF 12 frames, 960×640
- dispersion Pearson r 1.000; wavelength-sweep Pearson r 0.997
- audit overall 86.4 (threshold 85)
- non-circular specimen metrics include flank spacing 6.33 mm versus the 0.1843 mm wave, aspect ratio 1.47 versus a circular null, and orientation KS 0.135

Signing was not performed. `~/.nanopub/profile.yml` is absent, so the knowledge-graph nanopubs are unsigned. The publisher URI on those graphs is the repository, not an ORCID, and no signature was fabricated.

The dorsoventral anisotropy trial (Dy/Dx = 1.8) selected 212 µm and was rejected so the onset law stays at 184.3 µm.
