# MorphoEval Full-Loop Telemetry — `2026-10-27-dickinsonia-costata-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-dickinsonia-costata-turing-wavelength-s0`
- **Primary paper / dataset:** Dickinsonia costata Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` only. A second assessor was not run.
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.09 s** (0.02 min) — Tier A: `0.52 s` · Tier B: `0.58 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **28,229** (input: `0`, output: `28,229`, subagent: `0`) — Tier A: `13,889` · Tier B: `14,340`
- **Total artifact tokens produced:** `28,229`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,854 | 0 | **1,854** | 1,854 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,651 | 0 | **2,651** | 2,651 | `dickinsonia_costata_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `dickinsonia_costata_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 918 | 0 | **918** | 918 | `Loop_3167_dickinsonia_costata_turing_wavelength_s0.lean`, `delphi10000-dickinsonia-costata-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 8,274 | 0 | **8,274** | 8,274 | `2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `dickinsonia_costata_turing_wavelength_s0_growth.gif`, `dickinsonia_costata_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 5,618 | 0 | **5,618** | 5,618 | `2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit.yaml`, `2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 2,069 | 0 | **2,069** | 2,069 | `2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 2,961 | 0 | **2,961** | 2,961 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 2,961 | 0 | **2,961** | 2,961 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 731 | 0 | **731** | 731 | `2026-10-27-dickinsonia-costata-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.09** | **0.00** | **0** | **28,229** | **0** | **28,229** | **28,229** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex + Wikimedia scout for Dickinsonia costata; Slot 3167 is not in campaign_100. Scouted by query. No collision with dickinsonia-costata-turing-wavelength-s0. Sibling 2.2 mm script left untouched.
- **Artifact:** `datasets/dickinsonia_costata_turing_wavelength_s0/scout_manifest.json` (6,373 B, `1,854` tok, SHA-256 `7d02cb51fc7fd66f…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON states 5.564e-5 m and 558 Ma. Six DOIs. Schnakenberg kinetics and Turing k_c.
- **Artifact:** `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_reference.json` (5,454 B, `1,520` tok, SHA-256 `654790d69ff50dc8…`)
- **Artifact:** `datasets/dickinsonia_costata_turing_wavelength_s0/paper_citations.bib` (2,159 B, `677` tok, SHA-256 `5eef4d2dd1042991…`)
- **Artifact:** `datasets/dickinsonia_costata_turing_wavelength_s0/README.md` (1,818 B, `454` tok, SHA-256 `2eab73577d2f595b…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored File:DickinsoniaCostata.jpg (Verisimilus, CC BY 2.5), SHA-256 9cde902e8577ea737c95a5807287f08ee1752c6ac649b56b9ed546ee354ba5f6.
- **Artifact:** `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_specimen.jpg` (1,072,917 B, `64` tok, SHA-256 `9cde902e8577ea73…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean namespace MathOfBiology.Loop_3167: six theorems, zero sorry. Septum holds 39 cellular pitches; diffusivity factor 1521.
- **Artifact:** `morphoeval/lean/loops/Loop_3167_dickinsonia_costata_turing_wavelength_s0.lean` (1,989 B, `64` tok, SHA-256 `178670014d80fc14…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-dickinsonia-costata-turing-wavelength-s0.yaml` (3,256 B, `854` tok, SHA-256 `d0a38a4eb0ca90ff…`)

### Step 5 — `script` (Script)
- **Notes:** Spectral Schnakenberg model with --self-test, sweep, ablation, and specimen ellipse.
- **Artifact:** `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` (30,053 B, `8,274` tok, SHA-256 `40ebecf673ab1316…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. r=0.999999999886, lambda=55.64 um, ablation ratio 4215, GIF 12 frames at 720x540.
- **Artifact:** `docs/media/dickinsonia_costata_turing_wavelength_s0_growth.gif` (1,204,227 B, `64` tok, SHA-256 `0262f370583c9cf4…`)
- **Artifact:** `docs/media/dickinsonia_costata_turing_wavelength_s0_comparison.png` (914,350 B, `64` tok, SHA-256 `35aa97b419408f00…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 69.2. Pass 2 overall 85.9 after specimen-matched GIF, sweep, ablation, and contour metrics.
- **Artifact:** `morphoeval/audits/2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit.yaml` (9,909 B, `2,528` tok, SHA-256 `586eb4b9c9369f21…`)
- **Artifact:** `morphoeval/audits/2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2.yaml` (11,911 B, `3,090` tok, SHA-256 `5734ca6719756c6d…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 quick_gate 100/100. Eleven measurements; specimen metrics differ from their references.
- **Artifact:** `morphoeval/audits/2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2.measurements.yaml` (7,477 B, `2,069` tok, SHA-256 `446f38d143a2963b…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2 KG conforms. Rubric nanopub not re-emitted (--rubric-from bos-taurus v7).
- **Artifact:** `morphoeval/kg/2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2/REPORT.md` (10,744 B, `2,961` tok, SHA-256 `644624526cf72ab5…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. Nanopubs were not signed and were not published.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for Pass 1 (69.2) and Pass 2 (85.9, supersedes Pass 1).
- **Artifact:** `morphoeval/kg/2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit-v2/REPORT.md` (10,744 B, `2,961` tok, SHA-256 `644624526cf72ab5…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 gates met: image_fitness 89.1, CV 96.0, 12 frames, 540 px, Pearson r 0.999999999886, audit 85.9. No Pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-dickinsonia-costata-turing-wavelength-s0-loop-image-fitness.json` (2,473 B, `731` tok, SHA-256 `e3891c4ace166ea9…`)

## Scientific result

Pass 2 meets every gate. No Pass 3 was required.

- Selected wavelength: 55.64 µm (5.564×10⁻⁵ m), White Sea, 558 Ma. Chronoseries 1/1.
- Pearson r of measured versus linear wavenumbers: 0.999999999886. Wavelength RMSE: 0.061 µm. KS: 0.125.
- Equal-diffusion control decays to roundoff. Reported ratio 4215 uses a 1×10⁻⁴ floor on the control standard deviation.
- Image fitness 89.1 = 0.55×96.0 + 0.30×78 + 0.15×86. GIF: 12 frames, 720×540. Audit overall 85.9.
- The spot field is not the fossil's modules. The GIF ends on the CC BY 2.5 photograph. Nanopubs are unsigned because no nanopub profile is present.
- Step wall-clock times above are the logging intervals, not the model runtime.
