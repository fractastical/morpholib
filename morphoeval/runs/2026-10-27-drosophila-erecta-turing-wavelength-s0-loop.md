# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-erecta-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-erecta-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila erecta Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.05 s** (0.02 min) — Tier A: `0.51 s` · Tier B: `0.54 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **27,801** (input: `0`, output: `27,801`, subagent: `0`) — Tier A: `12,346` · Tier B: `15,455`
- **Total artifact tokens produced:** `27,801`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,106 | 0 | **1,106** | 1,106 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,814 | 0 | **2,814** | 2,814 | `drosophila_erecta_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_erecta_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 851 | 0 | **851** | 851 | `Loop_3183_drosophila_erecta_turing_wavelength_s0.lean`, `delphi10000-drosophila-erecta-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 7,383 | 0 | **7,383** | 7,383 | `2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_erecta_turing_wavelength_s0_comparison.png`, `drosophila_erecta_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,998 | 0 | **1,998** | 1,998 | `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,539 | 0 | **1,539** | 1,539 | `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,099 | 0 | **2,099** | 2,099 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,099 | 0 | **2,099** | 2,099 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 7,720 | 0 | **7,720** | 7,720 | `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2.measurements.yaml`, `REPORT.md`, +1 more |
| — | **All** | **TOTAL** | — | `12/12` | **1.05** | **0.00** | **0** | **27,801** | **0** | **27,801** | **27,801** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia/GBIF; slot 3183 not in campaign_100; no collision with drosophila-erecta-turing-wavelength-s0
- **Artifact:** `datasets/drosophila_erecta_turing_wavelength_s0/scout_manifest.json` (4,212 B, `1,106` tok, SHA-256 `34ab8fb860de866b…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_turing_wavelength_s0_reference.json` (5,620 B, `1,481` tok, SHA-256 `db334ff62f6ae794…`)
- **Artifact:** `datasets/drosophila_erecta_turing_wavelength_s0/paper_citations.bib` (1,964 B, `587` tok, SHA-256 `2b2bacd184e5772d…`)
- **Artifact:** `datasets/drosophila_erecta_turing_wavelength_s0/README.md` (2,606 B, `746` tok, SHA-256 `1af9dc7489221e7e…`)

### Step 3 — `source` (Formal source)
- **Notes:** gallery fallback; Wikimedia and GBIF media empty; sha256 ef1ddb03
- **Artifact:** `datasets/drosophila_erecta_turing_wavelength_s0/drosophila_erecta_specimen.jpg` (94,419 B, `64` tok, SHA-256 `ef1ddb031fefd5f8…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_3183_drosophila_erecta_turing_wavelength_s0.lean` (1,956 B, `64` tok, SHA-256 `bd3e5d4688f81243…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-erecta-turing-wavelength-s0.yaml` (2,908 B, `787` tok, SHA-256 `f666382800d1fec4…`)

### Step 5 — `script` (Script)
- **Notes:** self-test passed
- **Artifact:** `models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py` (25,507 B, `7,383` tok, SHA-256 `899abb6779176fec…`)

### Step 6 — `output` (Output & compare)
- **Notes:** 16 frames 640x480
- **Artifact:** `docs/media/drosophila_erecta_turing_wavelength_s0_comparison.png` (637,721 B, `64` tok, SHA-256 `e24b0d85978700ec…`)
- **Artifact:** `docs/media/drosophila_erecta_turing_wavelength_s0_growth.gif` (2,568,655 B, `64` tok, SHA-256 `b30de403594b50fd…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** pass 1 overall 74.2
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit.yaml` (7,662 B, `1,998` tok, SHA-256 `f1671d1de75a11fc…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit.measurements.yaml` (5,402 B, `1,539` tok, SHA-256 `a696148ce1aa06c7…`)

### Step 9 — `build` (Write as claims)
- **Notes:** conformance OK
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit/REPORT.md` (7,773 B, `2,099` tok, SHA-256 `03f5273ccb6d4622…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signing unavailable: no ~/.nanopub/profile.yml; nanopubs not signed and not faked

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit/REPORT.md` (7,773 B, `2,099` tok, SHA-256 `03f5273ccb6d4622…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** pass 2 overall 86.9; image_fitness 82.6; cv 82.0; pearson 0.99803; gates pass; R4 and R5 remain open
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2.yaml` (9,472 B, `2,624` tok, SHA-256 `68ef864dd458c984…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2.measurements.yaml` (5,631 B, `1,610` tok, SHA-256 `93ca5927b390aa03…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-erecta-turing-wavelength-s0-audit-v2/REPORT.md` (9,002 B, `2,584` tok, SHA-256 `373ef9e5bcac16fc…`)
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-erecta-turing-wavelength-s0-loop-image-fitness.json` (3,078 B, `902` tok, SHA-256 `46a42012036f2c84…`)

## Pass 2 result

The scored assessor is Grok 4.7. A second provider did not score this loop (remediation R4, left open).

*Drosophila erecta*, Holocene, age 0 Ma, detail tier 2.262 µm. Predecessor *D. elegans* stays at its 101.1 µm primordium anchor and successor *D. eugracilis* at 100.6 µm, so the wavenumber at this slot is 44.69 times the elegans anchor. Age does not change.

Schnakenberg parameters are a=0.17, b=0.93, Dv/Du=16, Du=3.987e-14 m²/s. They are not the sibling power-law coefficients. Gallery annulus Pearson r=0.99803, RMSE=1.966, KS=0.064. Ring radius 97.30 µm versus the 101.1 µm elegans anchor. GIF CV similarity 82.0 (16 frames, 640×480).

image_fitness = 0.55×82.0 + 0.30×82 + 0.15×86 = 82.6. Pass 2 audit overall 86.9. All numeric gates passed, so there is no Pass 3.

Signing was not done: `~/.nanopub/profile.yml` is absent. Nanopubs were conformance-checked and left unsigned. Wikimedia Commons and GBIF returned no CC photograph, so the specimen is the project gallery fallback (R5, left open).
