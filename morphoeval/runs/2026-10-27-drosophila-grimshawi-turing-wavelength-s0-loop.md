# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-grimshawi-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila grimshawi Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor:** `grok-4.7` (builder-assessor only; no second model was run)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.15 s** (0.02 min) — Tier A: `0.56 s` · Tier B: `0.59 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **29,053** (input: `0`, output: `29,053`, subagent: `0`) — Tier A: `10,919` · Tier B: `18,134`
- **Total artifact tokens produced:** `29,053`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,464 | 0 | **1,464** | 1,464 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,068 | 0 | **1,068** | 1,068 | `drosophila_grimshawi_turing_wavelength_s0_reference.json` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_grimshawi_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Loop_3186_drosophila_grimshawi_turing_wavelength_s0.lean` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 8,195 | 0 | **8,195** | 8,195 | `2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_grimshawi_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,436 | 0 | **2,436** | 2,436 | `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,835 | 0 | **1,835** | 1,835 | `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 2,560 | 0 | **2,560** | 2,560 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 548 | 0 | **548** | 548 | `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-loop-image-fitness.json` |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 2,560 | 0 | **2,560** | 2,560 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 8,195 | 0 | **8,195** | 8,195 | `2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` |
| — | **All** | **TOTAL** | — | `12/12` | **1.15** | **0.00** | **0** | **29,053** | **0** | **29,053** | **29,053** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** loop 3186 scout; loop 3186 scout done
- **Artifact:** `datasets/drosophila_grimshawi_turing_wavelength_s0/scout_manifest.json` (5,165 B, `1,464` tok, SHA-256 `d14b8add06308528…`)

### Step 2 — `paper` (Paper)
- **Notes:** loop 3186 paper; loop 3186 paper done
- **Artifact:** `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_reference.json` (3,762 B, `1,068` tok, SHA-256 `ac4d59314c82a9c0…`)

### Step 3 — `source` (Formal source)
- **Notes:** loop 3186 source; loop 3186 source done
- **Artifact:** `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_specimen.jpg` (98,416 B, `64` tok, SHA-256 `4ea3740942a29423…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** loop 3186 rules; loop 3186 rules done
- **Artifact:** `morphoeval/lean/loops/Loop_3186_drosophila_grimshawi_turing_wavelength_s0.lean` (1,591 B, `64` tok, SHA-256 `a474215adf796f25…`)

### Step 5 — `script` (Script)
- **Notes:** loop 3186 script; loop 3186 script done
- **Artifact:** `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` (28,660 B, `8,195` tok, SHA-256 `3f4296b13ad224e2…`)

### Step 6 — `output` (Output & compare)
- **Notes:** loop 3186 output; loop 3186 output done
- **Artifact:** `docs/media/drosophila_grimshawi_turing_wavelength_s0_growth.gif` (6,713,095 B, `64` tok, SHA-256 `7b664426b7a9d089…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** loop 3186 audit; loop 3186 audit done
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2.yaml` (9,373 B, `2,436` tok, SHA-256 `4a2a19340783a016…`)

### Step 8 — `measure` (Measure)
- **Notes:** loop 3186 measure; loop 3186 measure done
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2.measurements.yaml` (6,732 B, `1,835` tok, SHA-256 `5bd592780cd9acd2…`)

### Step 9 — `build` (Write as claims)
- **Notes:** loop 3186 build; loop 3186 build done
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2/REPORT.md` (9,411 B, `2,560` tok, SHA-256 `0270287b364b8521…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** loop 3186 publish; loop 3186 publish done
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-grimshawi-turing-wavelength-s0-loop-image-fitness.json` (1,901 B, `548` tok, SHA-256 `e15b559e7aae3e82…`)

### Step 11 — `graph` (Public record)
- **Notes:** loop 3186 graph; loop 3186 graph done
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2/REPORT.md` (9,411 B, `2,560` tok, SHA-256 `0270287b364b8521…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** loop 3186 remed; loop 3186 remed done
- **Artifact:** `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` (28,660 B, `8,195` tok, SHA-256 `3f4296b13ad224e2…`)


The per-step wall times above are the post-hoc logging calls, not the Schnakenberg integration or the GIF render.

## Image fitness and signing

Pass 2 gates, all met: image_fitness 91.2, cv_sim_similarity 93.8, 12 frames, 600 px shortest side, Pearson r 0.99974, audit overall 88.2. Non-circular specimen metrics include orientation KS 0.0418 (ablation reference 0.0191) and Otsu foreground fraction 0.1637 (ablation reference 0.0843).

Nanopublications were built and checked for ProbKnow conformance. `~/.nanopub/profile.yml` is absent, so they were not signed. No signature was fabricated.

Pass 3 was not run. The pass-2 gates all passed.
