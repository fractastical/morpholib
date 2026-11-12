# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-eugracilis-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila eugracilis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **711.00 s** (11.85 min) — Tier A: `570.00 s` · Tier B: `141.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,247** (input: `0`, output: `31,247`, subagent: `0`) — Tier A: `15,711` · Tier B: `15,536`
- **Total artifact tokens produced:** `31,247`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 180.00 | 0.00 | 0 | 1,272 | 0 | **1,272** | 1,272 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 90.00 | 0.00 | 0 | 3,336 | 0 | **3,336** | 3,336 | `drosophila_eugracilis_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_eugracilis_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 1,292 | 0 | **1,292** | 1,292 | `Loop_3184_drosophila_eugracilis_turing_wavelength_s0.lean`, `delphi10000-drosophila-eugracilis-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 200.00 | 0.00 | 0 | 9,619 | 0 | **9,619** | 9,619 | `2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_eugracilis_turing_wavelength_s0_comparison.png`, `drosophila_eugracilis_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 50.00 | 0.00 | 0 | 2,688 | 0 | **2,688** | 2,688 | `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 3,785 | 0 | **3,785** | 3,785 | `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 5,196 | 0 | **5,196** | 5,196 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 3,867 | 0 | **3,867** | 3,867 | `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **711.00** | **0.00** | **0** | **31,247** | **0** | **31,247** | **31,247** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** slot 3184 absent from campaign_100; OpenAlex/Crossref/Wikimedia query; slot 3184 not in campaign_100; Crossref-verified DOIs; Commons empty; no s0 collision
- **Artifact:** `datasets/drosophila_eugracilis_turing_wavelength_s0/scout_manifest.json` (4,666 B, `1,272` tok, SHA-256 `65cbd08f09bd7095…`)

### Step 2 — `paper` (Paper)
- **Notes:** 1.485e-6 m and 0 Ma in reference JSON and README
- **Artifact:** `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_reference.json` (5,839 B, `1,609` tok, SHA-256 `aeae6e2ee094bbd3…`)
- **Artifact:** `datasets/drosophila_eugracilis_turing_wavelength_s0/paper_citations.bib` (2,991 B, `948` tok, SHA-256 `963fd37391fe7bc7…`)
- **Artifact:** `datasets/drosophila_eugracilis_turing_wavelength_s0/README.md` (2,947 B, `779` tok, SHA-256 `6814a062121581e4…`)

### Step 3 — `source` (Formal source)
- **Notes:** gallery fallback SHA-256 f3705e55; Commons had no bitmap
- **Artifact:** `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_specimen.jpg` (95,993 B, `64` tok, SHA-256 `f3705e558598c7f5…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** six Lean theorems, zero sorry; math kg yaml
- **Artifact:** `morphoeval/lean/loops/Loop_3184_drosophila_eugracilis_turing_wavelength_s0.lean` (1,893 B, `64` tok, SHA-256 `6dd18c2b4a9520f3…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-eugracilis-turing-wavelength-s0.yaml` (4,400 B, `1,228` tok, SHA-256 `862ff721f681b271…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg k_c, self-test passed r=0.998
- **Artifact:** `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` (33,077 B, `9,619` tok, SHA-256 `06cb5918d75f4daa…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 12 frames 720x540; comparison PNG
- **Artifact:** `docs/media/drosophila_eugracilis_turing_wavelength_s0_comparison.png` (476,053 B, `64` tok, SHA-256 `6d16c18dee30d5d1…`)
- **Artifact:** `docs/media/drosophila_eugracilis_turing_wavelength_s0_growth.gif` (2,461,602 B, `64` tok, SHA-256 `b571e2432db179c4…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** pass 1 overall 76.7
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit.yaml` (10,297 B, `2,688` tok, SHA-256 `f2a6da52f7110591…`)

### Step 8 — `measure` (Measure)
- **Notes:** pass1 quick_gate 90; pass2 quick_gate 100
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit.measurements.yaml` (5,896 B, `1,778` tok, SHA-256 `65ff35ad2b7e8601…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2.measurements.yaml` (7,057 B, `2,007` tok, SHA-256 `15bc2e117427eea1…`)

### Step 9 — `build` (Write as claims)
- **Notes:** both KGs --conform OK
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit/REPORT.md` (9,038 B, `2,426` tok, SHA-256 `c480ae55e9afca11…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2/REPORT.md` (10,289 B, `2,770` tok, SHA-256 `7c9b70b616b10a53…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signing unavailable: no ~/.nanopub/profile.yml, no nanopub package, no @nanopub/sign

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for v1 76.7 and v2 86.1

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** pass 2 r=0.992 RMSE=0.701 ablation std 2.5e-4; gates pass; no pass 3
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit-v2.yaml` (11,332 B, `2,997` tok, SHA-256 `f80cd23d2f2ef03a…`)
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-eugracilis-turing-wavelength-s0-loop-image-fitness.json` (3,129 B, `870` tok, SHA-256 `19e5f9712173ef7a…`)

## Pass 2 gates

Pass 3 was not run. Every numeric gate passed after Pass 2.

| Gate | Result |
|---|---|
| image_fitness >= 70 | 84.7 (0.55×88.6 + 0.30×76 + 0.15×88) |
| cv_sim_similarity >= 58 | 88.6 |
| GIF frames >= 6 | 12 |
| resolution >= 340 px | 540 (720×540) |
| non-circular specimen metric | yes (FFT λ 1.809 µm vs analytic 1.485 µm; KS 0.343 vs 0; Pearson 0.992 vs 1) |
| Pearson r > 0.98 | 0.992 |
| audit overall >= 85 | 86.1 (Pass 1 was 76.7) |

The header lists a second assessor id because `track_loop.py init` records one. That assessor did not score this loop. Signing did not run: there is no `~/.nanopub/profile.yml`, and the nanopub signer dependencies are not installed. The TriG files are unsigned and passed `build_morphoeval_kg.py --conform`.
