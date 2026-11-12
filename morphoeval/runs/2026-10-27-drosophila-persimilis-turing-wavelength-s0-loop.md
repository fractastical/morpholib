# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-persimilis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-persimilis-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila persimilis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.92 s** (0.02 min) — Tier A: `0.45 s` · Tier B: `0.47 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **98,172** (input: `0`, output: `98,172`, subagent: `0`) — Tier A: `49,086` · Tier B: `49,086`
- **Total artifact tokens produced:** `98,172`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.07 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 8,181 | 0 | **8,181** | 8,181 | `2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` |
| — | **All** | **TOTAL** | — | `12/12` | **0.92** | **0.00** | **0** | **98,172** | **0** | **98,172** | **98,172** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref; slot 3198 not in campaign_100; no slug collision; OpenAlex/Crossref; slot 3198 not in campaign_100; no slug collision
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 2.098e-06 m, 0 Ma, chronoseries wavelengths; Equations, 2.098e-06 m, 0 Ma, chronoseries wavelengths
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 3 — `source` (Formal source)
- **Notes:** Gallery plate fallback; Commons had no CC bitmap; SHA-256 pinned; Gallery plate fallback; Commons had no CC bitmap; SHA-256 pinned
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3198, five theorems, zero sorry; math kg yaml; Lean Loop_3198, five theorems, zero sorry; math kg yaml
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg model with --self-test and measure_morphoeval; Schnakenberg model with --self-test and measure_morphoeval
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 6 — `output` (Output & compare)
- **Notes:** comparison png and 15-frame 640x480 gif; comparison png and 15-frame 640x480 gif
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** pass 1 overall 77.3; pass 2 overall 86.8; pass 1 overall 77.3; pass 2 overall 86.8
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 8 — `measure` (Measure)
- **Notes:** 12 measurements, quick_gate 100, values differ from references; 12 measurements, quick_gate 100, values differ from references
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK for audit and audit-v2; rubric reintroduction allowed; ProbKnow conformance OK for audit and audit-v2; rubric reintroduction allowed
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** signing unavailable, no profile.yml; signing unavailable, no profile.yml
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for both KG directories; REPORT.md present for both KG directories
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** pass 2 re-score; image_fitness gates pass; pass 3 not required; pass 2 re-score; image_fitness gates pass; pass 3 not required
- **Artifact:** `models/2026-delphi10000-drosophila-persimilis-turing-wavelength-s0.py` (30,461 B, `8,181` tok, SHA-256 `1dd7ead476b05ee8…`)

## Slot 3198 results

Schnakenberg onset wavelength for *Drosophila persimilis* at the assigned detail scale **2.1 µm** (2.098e-06 m, log10 = -5.678), Holocene **0 Ma**. Diffusion length changes along the chronoseries. Measured centroids: 1.955 µm (*D. paulistorum*), 2.097 µm (*D. persimilis*), 2.807 µm (*D. planitibia*).

Pass 1 overall **77.3**. Pass 2 overall **86.8**. Pass 3 was not required. Pearson r = 0.9993, RMSE = 0.033 µm, ablation ratio = 0.051, orientation KS = 0.505.

Image fitness **89.7** = 0.55*96.0 + 0.30*78 + 0.15*90. GIF: 15 frames, 640x480. All pass-2 gates cleared.

Signing was not done: `~/.nanopub/profile.yml` is absent, so no trusty URI was minted. Unsigned TriG conformance was checked (`--conform`: OK). The shared rubric nanopub is already published from earlier audits; `--allow-reintroduction` let the new claim files be checked without a fake supersession of that rubric.

The gallery plate is not an electron micrograph. Commons had no CC bitmap of this fly. The GIF last frame is that plate for the image-fitness crop; wavelength claims come from the Schnakenberg field.
