# MorphoEval Full-Loop Telemetry — `2026-10-27-acetabularia-acetabulum-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-acetabularia-acetabulum-turing-wavelength-s0`
- **Primary paper / dataset:** Acetabularia acetabulum Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **560.84 s** (9.35 min) — Tier A: `560.34 s` · Tier B: `0.50 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 559.92 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **560.84** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia. Slot 3103 is outside campaign_100_slots.yaml, so scout used --query. No slug collision.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, README. Scale 25.28 um and age 0 Ma.

### Step 3 — `source` (Formal source)
- **Notes:** CC0 cap (Cardenosa) and CC BY colony (Bas), plus gallery fallback. SHA-256 pinned.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3103, six theorems, zero sorry. Math KG yaml.

### Step 5 — `script` (Script)
- **Notes:** Brusselator, k_c, anisotropy, --self-test.

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 640x480 x 12 frames and comparison PNG.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 85.4. Remediation: anisotropy sweep, grain, second assessor.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100. Seven non-circular metrics.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for pass 1 and pass 2. Origin remote is not github.com/fractastical/morpholib, so commit-pinned loci were not verified.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Nanopubs left unsigned. Not faked.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written for pass 1 (85.4) and pass 2 (89.9).

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 anisotropy sweep and seeded grain. CV 87.3, image_fitness 88.1, Pearson r 0.9996, audit 89.9. All gates passed. Pass 3 not run.

## Pass-2 gates

All gates passed after pass 2. Pass 3 was not run.

- image_fitness 88.1 (>= 70)
- cv_sim_similarity 87.3 (>= 58), 12 frames, 640×480
- Pearson r (dispersion) 0.9996 and wavelength sweep 0.99999 (both > 0.98)
- audit overall 89.9 (>= 85); pass 1 was 85.4
- non-circular metrics: saturated wavelength 25.08 µm vs tier 25.28 µm; specimen coherency 0.625 vs render 0.541; orientation KS 0.191 vs 0
- Signing: unavailable. `~/.nanopub/profile.yml` is absent, so the KG nanopubs are conformant and unsigned.
