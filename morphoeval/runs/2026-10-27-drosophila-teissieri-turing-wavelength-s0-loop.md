# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-teissieri-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-teissieri-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila teissieri Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1218.09 s** (20.30 min) — Tier A: `1217.50 s` · Tier B: `0.59 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **1,644** (input: `0`, output: `1,644`, subagent: `0`) — Tier A: `1,644` · Tier B: `0`
- **Total artifact tokens produced:** `1,644`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1217.01 | 0.00 | 0 | 1,644 | 0 | **1,644** | 1,644 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1218.09** | **0.00** | **0** | **1,644** | **0** | **1,644** | **1,644** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Commons. No CC bitmap. No slug collision with slot 2761.
- **Artifact:** `datasets/drosophila_teissieri_turing_wavelength_s0/scout_manifest.json` (6,020 B, `1,644` tok, SHA-256 `bd49aed74b61068c…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README. Scale 1.72e-6 m, age 0 Ma.

### Step 3 — `source` (Formal source)
- **Notes:** Gallery fallback specimen SHA-256 pinned. Commons had no bitmap.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3209, six theorems, zero sorry; math KG yaml.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg + Turing k_c, self-test pass, Pearson r 0.99349.

### Step 6 — `output` (Output & compare)
- **Notes:** comparison.png and 12-frame 800x600 growth.gif.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 then Pass 2 audit. Overall 87.9. R3 remains open.

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100, 20 measurements.

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for v1 and v2. Unsigned.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Not faked.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 87.9 faithful.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 added sigma-peak and rendered-frame KS. Gates pass; no pass 3.

## Result (Pass 2)

Schnakenberg kinetics on a periodic 20.64 µm patch, with diffusivities solved so Turing's formula gives λ_c = 1.72 µm at geological age 0 Ma. Dispersion Pearson r = 0.99349 (RMSE 0.030 1/s). The growth-rate peak is 1.985 µm; the saturated wavelength is 1.903 µm (4.2% from that peak, 10.6% from the formula wavelength). Those are reported separately.

| Gate | Result |
|---|---|
| image_fitness | 87.3 (≥ 70) |
| cv_sim_similarity | 88.4 (≥ 58) |
| GIF frames | 12 (≥ 6) |
| resolution | 800×600 (≥ 340) |
| non-circular specimen metrics | 5, values differ from references |
| Pearson r | 0.99349 (> 0.98) |
| audit overall | 87.9 (≥ 85), `2026-10-27-drosophila-teissieri-turing-wavelength-s0-audit-v2` |

Pass 3 was not run. Signing was not performed: `~/.nanopub/profile.yml` is absent. Nanopubs were checked with `build_morphoeval_kg.py --conform` and left unsigned. R3 stays open (single assessor; Commons had no CC bitmap of *D. teissieri*). The audit assessor is Grok 4.7; the tracker header above keeps its default model labels.
