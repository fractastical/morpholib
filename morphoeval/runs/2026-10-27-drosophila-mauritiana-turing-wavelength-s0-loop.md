# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-mauritiana-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila mauritiana Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.22 s** (0.02 min) — Tier A: `0.56 s` · Tier B: `0.65 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **27,377** (input: `0`, output: `27,377`, subagent: `0`) — Tier A: `14,248` · Tier B: `13,129`
- **Total artifact tokens produced:** `27,377`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 1,351 | 0 | **1,351** | 1,351 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 2,888 | 0 | **2,888** | 2,888 | `drosophila_mauritiana_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Drosophila_mauritiana.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 1,236 | 0 | **1,236** | 1,236 | `Loop_3191_drosophila_mauritiana_turing_wavelength_s0.lean`, `delphi10000-drosophila-mauritiana-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 8,581 | 0 | **8,581** | 8,581 | `2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_mauritiana_turing_wavelength_s0_comparison.png`, `drosophila_mauritiana_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.11 | 0.00 | 0 | 5,268 | 0 | **5,268** | 5,268 | `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 4,012 | 0 | **4,012** | 4,012 | `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.12 | 0.00 | 0 | 2,773 | 0 | **2,773** | 2,773 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.12 | 0.00 | 0 | 1,076 | 0 | **1,076** | 1,076 | `2026-10-27-drosophila-mauritiana-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.22** | **0.00** | **0** | **27,377** | **0** | **27,377** | **27,377** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex plus Crossref; slot 3191 not in campaign_100; No slug collision with the 89 µm sibling
- **Artifact:** `datasets/drosophila_mauritiana_turing_wavelength_s0/scout_manifest.json` (5,239 B, `1,351` tok, SHA-256 `88be94822bc13906…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/drosophila_mauritiana_turing_wavelength_s0/drosophila_mauritiana_turing_wavelength_s0_reference.json` (6,116 B, `1,690` tok, SHA-256 `837cc650677e6049…`)
- **Artifact:** `datasets/drosophila_mauritiana_turing_wavelength_s0/paper_citations.bib` (1,801 B, `554` tok, SHA-256 `ac40fd9011933568…`)
- **Artifact:** `datasets/drosophila_mauritiana_turing_wavelength_s0/README.md` (2,554 B, `644` tok, SHA-256 `29501efb2e4ea0ab…`)

### Step 3 — `source` (Formal source)
- **Notes:** Gallery fallback; Commons had no species photo. SHA-256 pinned.
- **Artifact:** `datasets/drosophila_mauritiana_turing_wavelength_s0/Drosophila_mauritiana.jpg` (99,314 B, `64` tok, SHA-256 `0c64a140705a3c42…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3191_drosophila_mauritiana_turing_wavelength_s0.lean` (1,835 B, `64` tok, SHA-256 `1eb8362bf91d83d1…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-mauritiana-turing-wavelength-s0.yaml` (4,228 B, `1,172` tok, SHA-256 `dd84d9a4cfe22b08…`)

### Step 5 — `script` (Script)
- **Artifact:** `models/2026-delphi10000-drosophila-mauritiana-turing-wavelength-s0.py` (30,789 B, `8,581` tok, SHA-256 `29783632ad34e511…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test passed; 12 frames 640x480
- **Artifact:** `docs/media/drosophila_mauritiana_turing_wavelength_s0_comparison.png` (430,902 B, `64` tok, SHA-256 `cc30df7b3ad8dd5e…`)
- **Artifact:** `docs/media/drosophila_mauritiana_turing_wavelength_s0_growth.gif` (2,846,729 B, `64` tok, SHA-256 `b8adad8b9810f34a…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 78.9; pass 2 overall 89.4
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit.yaml` (9,453 B, `2,375` tok, SHA-256 `10b96fd66aaff30b…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2.yaml` (11,253 B, `2,893` tok, SHA-256 `5cb5421129b05c43…`)

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100; Pearson r 0.9997
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit.measurements.yaml` (6,998 B, `1,954` tok, SHA-256 `8c30637c7a1a4528…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2.measurements.yaml` (7,636 B, `2,058` tok, SHA-256 `5033738d629f3d18…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform OK for v1 and v2; rubric not re-emitted

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no requests module and no ~/.nanopub/profile.yml. Not signed.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT overall 89.4 faithful
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-mauritiana-turing-wavelength-s0-audit-v2/REPORT.md` (10,310 B, `2,773` tok, SHA-256 `562a3ad7879ca564…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 gates met: fitness 91.2, CV 93.9, r 0.9997, overall 89.4. Pass 3 not required.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-mauritiana-turing-wavelength-s0-loop-image-fitness.json` (3,666 B, `1,076` tok, SHA-256 `8f2b07dce0dba2ea…`)
