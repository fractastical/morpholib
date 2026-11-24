# MorphoEval Full-Loop Telemetry — `2026-10-27-charnia-masoni-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-charnia-masoni-turing-wavelength-s0`
- **Primary paper / dataset:** Charnia masoni Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.09 s** (0.02 min) — Tier A: `0.56 s` · Tier B: `0.53 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **32,621** (input: `0`, output: `32,621`, subagent: `0`) — Tier A: `16,164` · Tier B: `16,457`
- **Total artifact tokens produced:** `32,621`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 1,089 | 0 | **1,089** | 1,089 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 3,949 | 0 | **3,949** | 3,949 | `charnia_masoni_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `charnia_masoni_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,359 | 0 | **1,359** | 1,359 | `Loop_3145_charnia_masoni_turing_wavelength_s0.lean`, `delphi10000-charnia-masoni-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,575 | 0 | **9,575** | 9,575 | `2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `charnia_masoni_turing_wavelength_s0_comparison.png`, `charnia_masoni_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 5,366 | 0 | **5,366** | 5,366 | `2026-10-27-charnia-masoni-turing-wavelength-s0-audit.yaml`, `2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 2,343 | 0 | **2,343** | 2,343 | `2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 3,000 | 0 | **3,000** | 3,000 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 5,138 | 0 | **5,138** | 5,138 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 610 | 0 | **610** | 610 | `2026-10-27-charnia-masoni-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.09** | **0.00** | **0** | **32,621** | **0** | **32,621** | **32,621** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3145 is outside campaign_100; OpenAlex/Crossref/Wikimedia used.; No collision with charnia-masoni-turing-wavelength-s0.
- **Artifact:** `datasets/charnia_masoni_turing_wavelength_s0/scout_manifest.json` (4,072 B, `1,089` tok, SHA-256 `ee8179dfab5e793f…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_turing_wavelength_s0_reference.json` (6,088 B, `1,655` tok, SHA-256 `262080091420a03a…`)
- **Artifact:** `datasets/charnia_masoni_turing_wavelength_s0/paper_citations.bib` (3,841 B, `1,214` tok, SHA-256 `f4cb4c63fbfbc0fd…`)
- **Artifact:** `datasets/charnia_masoni_turing_wavelength_s0/README.md` (4,094 B, `1,080` tok, SHA-256 `4488c865883388ab…`)

### Step 3 — `source` (Formal source)
- **Notes:** CC BY-SA 3.0 cast by Ghedoghedo, Tübingen, 2013. Synthetic gallery JPEG rejected.
- **Artifact:** `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_specimen.jpg` (2,588,631 B, `64` tok, SHA-256 `c7e8d470af175675…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3145_charnia_masoni_turing_wavelength_s0.lean` (1,827 B, `64` tok, SHA-256 `867ce2504f1bc690…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-charnia-masoni-turing-wavelength-s0.yaml` (4,617 B, `1,295` tok, SHA-256 `239d26bc573a01db…`)

### Step 5 — `script` (Script)
- **Notes:** Self-test r=0.99844, sweep r=1.0.
- **Artifact:** `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` (32,137 B, `9,575` tok, SHA-256 `4658b004cd5ffbc1…`)

### Step 6 — `output` (Output & compare)
- **Notes:** 12 frames, 640x1280, CV 89.5.
- **Artifact:** `docs/media/charnia_masoni_turing_wavelength_s0_comparison.png` (342,160 B, `64` tok, SHA-256 `261c01363f55e4ae…`)
- **Artifact:** `docs/media/charnia_masoni_turing_wavelength_s0_growth.gif` (2,526,756 B, `64` tok, SHA-256 `db62de37cf28701f…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 78.8; pass 2 overall 87.5. No second-company assessor.
- **Artifact:** `morphoeval/audits/2026-10-27-charnia-masoni-turing-wavelength-s0-audit.yaml` (7,522 B, `1,925` tok, SHA-256 `63e4e3dd12c79bba…`)
- **Artifact:** `morphoeval/audits/2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2.yaml` (13,496 B, `3,441` tok, SHA-256 `1a11846fb444f7ee…`)

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100/100. lambda_m 167.44 µm vs k_c reference 140.4 µm.
- **Artifact:** `morphoeval/audits/2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2.measurements.yaml` (8,129 B, `2,343` tok, SHA-256 `05a0ef9ed9ae5e6b…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Conformant with --rubric-from the existing Bos taurus rubric nanopub.
- **Artifact:** `morphoeval/kg/2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2/REPORT.md` (10,479 B, `3,000` tok, SHA-256 `01ab3a3738fe957b…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and no trusty URI was invented.

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-27-charnia-masoni-turing-wavelength-s0-audit/REPORT.md` (7,719 B, `2,138` tok, SHA-256 `ab54fa42d4a1db2d…`)
- **Artifact:** `morphoeval/kg/2026-10-27-charnia-masoni-turing-wavelength-s0-audit-v2/REPORT.md` (10,479 B, `3,000` tok, SHA-256 `01ab3a3738fe957b…`)

## Result

Pass 2 gates, all met on the committed artifacts. No pass 3.

- image_fitness = 87.3 (0.55×89.5 + 0.30×84 + 0.15×86)
- cv_sim_similarity = 89.5
- GIF frames = 12, resolution = 640×1280
- Pearson r = 0.9984 (dispersion) and 1.000 (wavelength sweep)
- audit overall = 87.5
- Fastest-mode wavelength 167.44 µm is not equal to the 140.4 µm determinant-critical reference
- Nanopubs were built and checked for template conformance. They were not signed: `~/.nanopub/profile.yml` is absent, and no trusty URI was invented.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 fixed the disk render, the mixed-k sweep, and the missing control panel. Gates pass. No pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-charnia-masoni-turing-wavelength-s0-loop-image-fitness.json` (2,061 B, `610` tok, SHA-256 `8ec6361b8125ec03…`)
