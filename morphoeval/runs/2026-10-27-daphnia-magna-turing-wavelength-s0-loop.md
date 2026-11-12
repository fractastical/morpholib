# MorphoEval Full-Loop Telemetry — `2026-10-27-daphnia-magna-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-daphnia-magna-turing-wavelength-s0`
- **Primary paper / dataset:** Daphnia magna Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `none-single-assessor`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **471.00 s** (7.85 min) — Tier A: `335.00 s` · Tier B: `136.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **25,719** (input: `0`, output: `25,719`, subagent: `0`) — Tier A: `14,772` · Tier B: `10,947`
- **Total artifact tokens produced:** `25,719`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 48.00 | 0.00 | 0 | 1,271 | 0 | **1,271** | 1,271 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 2,997 | 0 | **2,997** | 2,997 | `daphnia_magna_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `daphnia_magna_head_sem.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 1,211 | 0 | **1,211** | 1,211 | `Loop_3165_daphnia_magna_turing_wavelength_s0.lean`, `delphi10000-daphnia-magna-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 180.00 | 0.00 | 0 | 9,101 | 0 | **9,101** | 9,101 | `2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `daphnia_magna_turing_wavelength_s0_comparison.png`, `daphnia_magna_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 35.00 | 0.00 | 0 | 3,117 | 0 | **3,117** | 3,117 | `2026-10-27-daphnia-magna-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 1,686 | 0 | **1,686** | 1,686 | `2026-10-27-daphnia-magna-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 2,661 | 0 | **2,661** | 2,661 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 3.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 70.00 | 0.00 | 0 | 3,483 | 0 | **3,483** | 3,483 | `2026-10-27-daphnia-magna-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-daphnia-magna-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **471.00** | **0.00** | **0** | **25,719** | **0** | **25,719** | **25,719** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3165 is outside campaign_100. OpenAlex, Crossref, and Wikimedia. No collision on daphnia-magna-turing-wavelength-s0.
- **Artifact:** `datasets/daphnia_magna_turing_wavelength_s0/scout_manifest.json` (4,677 B, `1,271` tok, SHA-256 `ccdb3083e47de81d…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 2.736e-06 m, 0 Ma, chronoseries 1/1.
- **Artifact:** `datasets/daphnia_magna_turing_wavelength_s0/daphnia_magna_turing_wavelength_s0_reference.json` (5,434 B, `1,646` tok, SHA-256 `99ef36907be2c6a9…`)
- **Artifact:** `datasets/daphnia_magna_turing_wavelength_s0/paper_citations.bib` (2,183 B, `671` tok, SHA-256 `52534c86e8d423a7…`)
- **Artifact:** `datasets/daphnia_magna_turing_wavelength_s0/README.md` (2,663 B, `680` tok, SHA-256 `63155543804a5cec…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA 4.0 head SEM, Paulo Corgosinho, SHA-256 pinned.
- **Artifact:** `datasets/daphnia_magna_turing_wavelength_s0/daphnia_magna_head_sem.jpg` (66,265 B, `64` tok, SHA-256 `0cd44adfc5e82cd2…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Four Lean theorems, zero sorry.
- **Artifact:** `morphoeval/lean/loops/Loop_3165_daphnia_magna_turing_wavelength_s0.lean` (1,957 B, `64` tok, SHA-256 `a10581bec42b05b1…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-daphnia-magna-turing-wavelength-s0.yaml` (4,225 B, `1,147` tok, SHA-256 `65e60a575614d1f5…`)

### Step 5 — `script` (Script)
- **Notes:** Spectral Schnakenberg model with sweep, ablation, and specimen segmentation.
- **Artifact:** `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` (31,520 B, `9,101` tok, SHA-256 `590d8e694ff87b12…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. PNG plus 12-frame 720x540 GIF.
- **Artifact:** `docs/media/daphnia_magna_turing_wavelength_s0_comparison.png` (345,784 B, `64` tok, SHA-256 `9fac22d3c08e135d…`)
- **Artifact:** `docs/media/daphnia_magna_turing_wavelength_s0_growth.gif` (2,671,805 B, `64` tok, SHA-256 `50d8f2ffb0e2ca69…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 84.0.
- **Artifact:** `morphoeval/audits/2026-10-27-daphnia-magna-turing-wavelength-s0-audit.yaml` (12,021 B, `3,117` tok, SHA-256 `ad818ed82aa18b49…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 1 quick_gate 100/100.
- **Artifact:** `morphoeval/audits/2026-10-27-daphnia-magna-turing-wavelength-s0-audit.measurements.yaml` (5,762 B, `1,686` tok, SHA-256 `6cd72c414c75a29c…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 1 KG conformant.
- **Artifact:** `morphoeval/kg/2026-10-27-daphnia-magna-turing-wavelength-s0-audit/REPORT.md` (9,794 B, `2,661` tok, SHA-256 `75561b0ba2ab45ab…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Nothing was fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md written for pass 1 and pass 2.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 held-out r=0.99999978, coarsening 1.3601, overall 86.4, image_fitness 89.4, CV 96. Gates passed. No pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-daphnia-magna-turing-wavelength-s0-audit-v2.yaml` (10,320 B, `2,763` tok, SHA-256 `e928ea8b1b70ace1…`)
- **Artifact:** `morphoeval/runs/2026-10-27-daphnia-magna-turing-wavelength-s0-loop-image-fitness.json` (2,343 B, `720` tok, SHA-256 `f3a337c804a841f5…`)

## Outcome (Pass 2)

Gates after Pass 2 all hold. Pass 3 was not run.

- image_fitness 89.4 (0.55×96.0 + 0.30×76 + 0.15×92)
- cv_sim_similarity 96.0
- GIF 12 frames, 720×540
- Pearson r 0.99999995 on the scored band and 0.99999978 on the held-out odd modes
- audit overall 86.4 (Pass 1 was 84.0)
- non-circular specimen metrics include the cuticle peak 20.17 µm against the model wavelength 2.736 µm

Signing did not run. `~/.nanopub/profile.yml` is absent, and no signature was written. The knowledge-graph files are unsigned TriG.

The chronoseries is 1/1 at 0 Ma, so there is no other chronospecies. The length that changes is the tier: 2.736 µm sits between a melanosome-scale predecessor and an ommatidial successor. The ×130 cuticle mesh, measured at 20.17 µm on the Corgosinho SEM, is not the Turing field.
