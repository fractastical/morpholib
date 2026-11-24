# MorphoEval Full-Loop Telemetry — `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-confuciusornis-sanctus-turing-wavelength-s0`
- **Primary paper / dataset:** Confuciusornis sanctus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` (single assessor; no second assessor)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **249.68 s** (4.16 min) — Tier A: `243.87 s` · Tier B: `5.81 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **27,621** (input: `0`, output: `27,621`, subagent: `0`) — Tier A: `11,565` · Tier B: `16,056`
- **Total artifact tokens produced:** `27,621`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 239.15 | 0.00 | 0 | 64 | 0 | **64** | 64 | `confuciusornis_sanctus_specimen.jpg` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.94 | 0.00 | 0 | 3,149 | 0 | **3,149** | 3,149 | `confuciusornis_sanctus_turing_wavelength_s0_reference.json`, `paper_citations.bib` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.94 | 0.00 | 0 | 64 | 0 | **64** | 64 | `confuciusornis_sanctus_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.94 | 0.00 | 0 | 971 | 0 | **971** | 971 | `Loop_3152_confuciusornis_sanctus_turing_wavelength_s0.lean`, `delphi10000-confuciusornis-sanctus-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.94 | 0.00 | 0 | 7,189 | 0 | **7,189** | 7,189 | `2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.95 | 0.00 | 0 | 128 | 0 | **128** | 128 | `confuciusornis_sanctus_turing_wavelength_s0_comparison.png`, `confuciusornis_sanctus_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.96 | 0.00 | 0 | 7,154 | 0 | **7,154** | 7,154 | `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit.yaml`, `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.96 | 0.00 | 0 | 1,754 | 0 | **1,754** | 1,754 | `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.97 | 0.00 | 0 | 3,058 | 0 | **3,058** | 3,058 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.97 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.98 | 0.00 | 0 | 3,058 | 0 | **3,058** | 3,058 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.98 | 0.00 | 0 | 1,032 | 0 | **1,032** | 1,032 | `2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **249.68** | **0.00** | **0** | **27,621** | **0** | **27,621** | **27,621** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia; slot 3152 absent from campaign_100; No path collision with 5.4 mm sibling 1999-chiappe; slot 3152 not in campaign_100. OpenAlex/Crossref DOIs verified. Wikimedia CC BY-SA 4.0 Senckenberg slab vendored.
- **Artifact:** `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_specimen.jpg` (2,709,657 B, `64` tok, SHA-256 `9e83715a2090a049…`)

### Step 2 — `paper` (Paper)
- **Notes:** Eqs 1-5, Schnakenberg carrier, 143.4 um, 125.0 Ma, chronoseries 1/1
- **Artifact:** `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_turing_wavelength_s0_reference.json` (7,239 B, `2,083` tok, SHA-256 `95376a32ced97ca5…`)
- **Artifact:** `datasets/confuciusornis_sanctus_turing_wavelength_s0/paper_citations.bib` (3,492 B, `1,066` tok, SHA-256 `969b2a6a15f61ba8…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA 4.0 Senckenberg slab SHA-256 9e83715a2090a0499eb6dde56991bc7af2cd428c5a3e3347becd2d05c7809614
- **Artifact:** `datasets/confuciusornis_sanctus_turing_wavelength_s0/confuciusornis_sanctus_specimen.jpg` (2,709,657 B, `64` tok, SHA-256 `9e83715a2090a049…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Six Lean theorems, zero sorry; lean binary not on PATH
- **Artifact:** `morphoeval/lean/loops/Loop_3152_confuciusornis_sanctus_turing_wavelength_s0.lean` (1,822 B, `64` tok, SHA-256 `b6f8af426e058c63…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-confuciusornis-sanctus-turing-wavelength-s0.yaml` (3,318 B, `907` tok, SHA-256 `cb51fabdfed32a89…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 script with Du sweep, ablation, segmentation, Pearson r=0.9964
- **Artifact:** `models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py` (25,497 B, `7,189` tok, SHA-256 `46cbcf663f4f80cc…`)

### Step 6 — `output` (Output & compare)
- **Notes:** 640x480 12-frame GIF and comparison PNG
- **Artifact:** `docs/media/confuciusornis_sanctus_turing_wavelength_s0_comparison.png` (479,795 B, `64` tok, SHA-256 `25071a44c68f3944…`)
- **Artifact:** `docs/media/confuciusornis_sanctus_turing_wavelength_s0_growth.gif` (2,469,500 B, `64` tok, SHA-256 `ae6b79ed11ee7b3b…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall ~63.3; Pass 2 overall 85.7
- **Artifact:** `morphoeval/audits/2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit.yaml` (13,096 B, `3,306` tok, SHA-256 `36c47f56e1c374ec…`)
- **Artifact:** `morphoeval/audits/2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2.yaml` (15,068 B, `3,848` tok, SHA-256 `673f37ac8052500d…`)

### Step 8 — `measure` (Measure)
- **Notes:** 9 measurements; quick_gate 80 validated
- **Artifact:** `morphoeval/audits/2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2.measurements.yaml` (6,463 B, `1,754` tok, SHA-256 `072c607c3bfc5367…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK with --allow-reintroduction; loci unverified until commit
- **Artifact:** `morphoeval/kg/2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2/REPORT.md` (11,180 B, `3,058` tok, SHA-256 `e5a6513ff3dc5fcf…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. No signature fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 85.7 faithful
- **Artifact:** `morphoeval/kg/2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-audit-v2/REPORT.md` (11,180 B, `3,058` tok, SHA-256 `e5a6513ff3dc5fcf…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4. Gates passed. Pass 3 not required. image_fitness 89.7, CV 96.0, r=0.9964
- **Artifact:** `morphoeval/runs/2026-10-27-confuciusornis-sanctus-turing-wavelength-s0-loop-image-fitness.json` (3,666 B, `1,032` tok, SHA-256 `01d2a41ce899ddf2…`)
