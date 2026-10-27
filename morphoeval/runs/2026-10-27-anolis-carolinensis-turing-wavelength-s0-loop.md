# MorphoEval Full-Loop Telemetry — `2026-10-27-anolis-carolinensis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-anolis-carolinensis-turing-wavelength-s0`
- **Primary paper / dataset:** Anolis carolinensis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **659.84 s** (11.00 min) — Tier A: `659.25 s` · Tier B: `0.59 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,756** (input: `0`, output: `31,756`, subagent: `0`) — Tier A: `14,017` · Tier B: `17,739`
- **Total artifact tokens produced:** `31,756`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 658.78 | 0.00 | 0 | 713 | 0 | **713** | 713 | `scout_manifest.json`, `anolis_carolinensis_specimen.jpg` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 3,088 | 0 | **3,088** | 3,088 | `anolis_carolinensis_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `anolis_carolinensis_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 788 | 0 | **788** | 788 | `Loop_3112_anolis_carolinensis_turing_wavelength_s0.lean`, `delphi10000-anolis-carolinensis-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 7,553 | 0 | **7,553** | 7,553 | `2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 1,811 | 0 | **1,811** | 1,811 | `anolis_carolinensis_turing_wavelength_s0_comparison.png`, `anolis_carolinensis_turing_wavelength_s0_growth.gif`, `metrics.json` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.10 | 0.00 | 0 | 2,675 | 0 | **2,675** | 2,675 | `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 3,335 | 0 | **3,335** | 3,335 | `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.10 | 0.00 | 0 | 5,059 | 0 | **5,059** | 5,059 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 2,686 | 0 | **2,686** | 2,686 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.10 | 0.00 | 0 | 3,984 | 0 | **3,984** | 3,984 | `2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-anolis-carolinensis-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **659.84** | **0.00** | **0** | **31,756** | **0** | **31,756** | **31,756** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** slot 3112 not in campaign_100; OpenAlex+Wikimedia query; OpenAlex 429; Wikimedia PD specimen Anolis_carolinensis-032.jpg; Crossref DOIs verified separately. No slug collision.
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/scout_manifest.json` (2,365 B, `649` tok, SHA-256 `20b2b934c012e202…`)
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_specimen.jpg` (179,632 B, `64` tok, SHA-256 `9a4dd3a9baffc3ce…`)

### Step 2 — `paper` (Paper)
- **Notes:** reference JSON, bib, README; scale 101.1 um, age 0 Ma
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_turing_wavelength_s0_reference.json` (5,164 B, `1,500` tok, SHA-256 `6d32cfacbb645944…`)
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/paper_citations.bib` (3,280 B, `955` tok, SHA-256 `1116086124046e8c…`)
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/README.md` (2,455 B, `633` tok, SHA-256 `a18705043f366b38…`)

### Step 3 — `source` (Formal source)
- **Notes:** Commons PD Anolis_carolinensis-032.jpg, Danny S. 2011, SHA-256 pinned
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/anolis_carolinensis_specimen.jpg` (179,632 B, `64` tok, SHA-256 `9a4dd3a9baffc3ce…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3112, 0 sorry, math kg yaml
- **Artifact:** `morphoeval/lean/loops/Loop_3112_anolis_carolinensis_turing_wavelength_s0.lean` (2,216 B, `64` tok, SHA-256 `852aa0260f457a09…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-anolis-carolinensis-turing-wavelength-s0.yaml` (2,452 B, `724` tok, SHA-256 `2fdfc44465d90ee4…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg self-test passed
- **Artifact:** `models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py` (25,453 B, `7,553` tok, SHA-256 `35e1d1544307100a…`)

### Step 6 — `output` (Output & compare)
- **Notes:** 12-frame 512px GIF, comparison PNG, metrics
- **Artifact:** `docs/media/anolis_carolinensis_turing_wavelength_s0_comparison.png` (466,662 B, `64` tok, SHA-256 `a5272637322b843e…`)
- **Artifact:** `docs/media/anolis_carolinensis_turing_wavelength_s0_growth.gif` (2,152,835 B, `64` tok, SHA-256 `3d7af53581c70860…`)
- **Artifact:** `datasets/anolis_carolinensis_turing_wavelength_s0/metrics.json` (5,426 B, `1,683` tok, SHA-256 `c537e53d6029029a…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** pass 1 overall 79.6
- **Artifact:** `morphoeval/audits/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit.yaml` (10,230 B, `2,675` tok, SHA-256 `b364a96c9b675cf4…`)

### Step 8 — `measure` (Measure)
- **Notes:** quick_gate 100/100
- **Artifact:** `morphoeval/audits/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit.measurements.yaml` (5,246 B, `1,394` tok, SHA-256 `874eca907f279960…`)
- **Artifact:** `morphoeval/audits/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2.measurements.yaml` (6,730 B, `1,941` tok, SHA-256 `aaf4ae27aabbe83c…`)

### Step 9 — `build` (Write as claims)
- **Notes:** both KGs --conform OK
- **Artifact:** `morphoeval/kg/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit/REPORT.md` (8,665 B, `2,373` tok, SHA-256 `1bdb75fd84d70360…`)
- **Artifact:** `morphoeval/kg/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2/REPORT.md` (9,679 B, `2,686` tok, SHA-256 `1950d351a71edaf1…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signing unavailable: no ~/.nanopub/profile.yml; graphs unsigned

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md pass1 79.6, pass2 85.4
- **Artifact:** `morphoeval/kg/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2/REPORT.md` (9,679 B, `2,686` tok, SHA-256 `1950d351a71edaf1…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** pass 2 gates passed; image_fitness 85.2; CV 88.9; no pass 3
- **Artifact:** `morphoeval/audits/2026-10-27-anolis-carolinensis-turing-wavelength-s0-audit-v2.yaml` (11,382 B, `2,989` tok, SHA-256 `ce9156e74d362baa…`)
- **Artifact:** `morphoeval/runs/2026-10-27-anolis-carolinensis-turing-wavelength-s0-loop-image-fitness.json` (3,323 B, `995` tok, SHA-256 `5f646631a26af97b…`)

## Pass-2 gates

All gates passed after pass 2. Pass 3 was not run.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 85.2 |
| cv_sim_similarity ≥ 58 | 88.9 |
| GIF frames ≥ 6 (rendered 12) | 12 |
| resolution ≥ 340 px (rendered 512) | 512 |
| ≥ 1 non-circular specimen metric | wavelength 99.82 µm vs 101.1 µm; specimen coherency 0.6275 vs field 0.5308 |
| Pearson r > 0.98 | 1.0, with RMSE 3.45 µm (proportional factor 0.971) |
| audit overall ≥ 85 | 85.4 (pass 1 was 79.6) |

Signing: `~/.nanopub/profile.yml` is absent. The trig files are unsigned. `build_morphoeval_kg.py --conform` passed. The creator IRI on the unsigned graphs is the one already used by this repository's MorphoEval nanopubs, because the builder requires a creator IRI. No signature was produced.
