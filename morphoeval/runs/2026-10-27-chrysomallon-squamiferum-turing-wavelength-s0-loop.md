# MorphoEval Full-Loop Telemetry — `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-loop`

Pass 2 gates (honest): image fitness 91.2, CV similarity 96.0, 12 frames at 640×480, Pearson r 0.996, PDE wavelength 1.935 nm, audit overall 87.1. Pass 1 overall was 72.7 because the onset wavenumber is not the fastest mode (PDE peak 2.52 nm) and there was no sweep or ablation. Nanopubs were not signed: `~/.nanopub/profile.yml` is absent. Lean was not executed: no `lean` binary on this machine. The header's second-assessor name is the tracker default; only one assessor scored the audits.

- **Target simulation:** `delphi10000-chrysomallon-squamiferum-turing-wavelength-s0`
- **Primary paper / dataset:** Chrysomallon squamiferum Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **879.82 s** (14.66 min) — Tier A: `879.27 s` · Tier B: `0.55 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **31,369** (input: `0`, output: `31,369`, subagent: `0`) — Tier A: `14,508` · Tier B: `16,861`
- **Total artifact tokens produced:** `31,369`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 878.84 | 0.00 | 0 | 1,232 | 0 | **1,232** | 1,232 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,755 | 0 | **3,755** | 3,755 | `chrysomallon_squamiferum_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 64 | 0 | **64** | 64 | `chrysomallon_squamiferum_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 894 | 0 | **894** | 894 | `Loop_3147_chrysomallon_squamiferum_turing_wavelength_s0.lean`, `delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 8,435 | 0 | **8,435** | 8,435 | `2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `chrysomallon_squamiferum_turing_wavelength_s0_growth.gif`, `chrysomallon_squamiferum_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,849 | 0 | **2,849** | 2,849 | `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,122 | 0 | **3,122** | 3,122 | `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 4,635 | 0 | **4,635** | 4,635 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,428 | 0 | **2,428** | 2,428 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,827 | 0 | **3,827** | 3,827 | `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **879.82** | **0.00** | **0** | **31,369** | **0** | **31,369** | **31,369** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex Crossref Wikimedia scout for slot 3147; OpenAlex empty via scout_literature; direct Crossref confirmed Turing 1952, Yao 2010, Chen 2015, Sun 2020. Wikimedia CC BY 4.0 paratype. No s0 filename collision.
- **Artifact:** `datasets/chrysomallon_squamiferum_turing_wavelength_s0/scout_manifest.json` (4,093 B, `1,232` tok, SHA-256 `ea589eba9289f0c9…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bibliography, README at 2.0 nm and 0 Ma; Equations, Jacobian, claim boundary, Crossref DOIs.
- **Artifact:** `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_turing_wavelength_s0_reference.json` (5,985 B, `1,719` tok, SHA-256 `76db33d8c8f5d53c…`)
- **Artifact:** `datasets/chrysomallon_squamiferum_turing_wavelength_s0/paper_citations.bib` (4,203 B, `1,275` tok, SHA-256 `63b7c502b791e3c1…`)
- **Artifact:** `datasets/chrysomallon_squamiferum_turing_wavelength_s0/README.md` (2,954 B, `761` tok, SHA-256 `31eb13a5a96c590c…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored MNHN paratype JPEG; Marie Hennion CC BY 4.0, MNHN-IM-2000-30072, SHA-256 pinned.
- **Artifact:** `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_specimen.jpg` (235,221 B, `64` tok, SHA-256 `0312a80d873d68d2…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4 Loop_3147 and math KG; Six theorems, zero sorry. lean binary not installed, so not executed.
- **Artifact:** `morphoeval/lean/loops/Loop_3147_chrysomallon_squamiferum_turing_wavelength_s0.lean` (1,863 B, `64` tok, SHA-256 `c2dedcf320365aa2…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.yaml` (3,113 B, `830` tok, SHA-256 `80d00d9535ebc574…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script then Pass 2 fastest-mode calibration; run_pass1 kept; run_pass2 is the default.
- **Artifact:** `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` (29,030 B, `8,435` tok, SHA-256 `526a5f3517c5400d…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test and renders; Pass 2 r=0.9964, lambda=1.935 nm, ablation ratio ~0, GIF 12x640x480.
- **Artifact:** `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_growth.gif` (1,370,524 B, `64` tok, SHA-256 `894a567dd3d326cd…`)
- **Artifact:** `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_comparison.png` (505,249 B, `64` tok, SHA-256 `f78b2d8abd16f668…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit; Pass 1 overall will be computed by the KG; remediation R1-R4.
- **Artifact:** `morphoeval/audits/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit.yaml` (10,365 B, `2,849` tok, SHA-256 `2ee491d4818e7cf6…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs pass 1 and pass 2; Both sidecars written; quick_gate 100.
- **Artifact:** `morphoeval/audits/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit.measurements.yaml` (4,982 B, `1,530` tok, SHA-256 `924dce23f836c5ee…`)
- **Artifact:** `morphoeval/audits/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2.measurements.yaml` (5,208 B, `1,592` tok, SHA-256 `d0e89eedee45133d…`)

### Step 9 — `build` (Write as claims)
- **Notes:** KG build; Pass 1 and Pass 2 conformance OK. Rubric nanopub reused from the published MorphoEval rubric. Signing not available.
- **Artifact:** `morphoeval/kg/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit/REPORT.md` (7,744 B, `2,207` tok, SHA-256 `9a34f1fb68990baa…`)
- **Artifact:** `morphoeval/kg/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2/REPORT.md` (8,579 B, `2,428` tok, SHA-256 `b50978581e378888…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Nanopub profile check; Signing unavailable: ~/.nanopub/profile.yml does not exist. Not faked.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md; Pass 2 REPORT overall recorded.
- **Artifact:** `morphoeval/kg/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2/REPORT.md` (8,579 B, `2,428` tok, SHA-256 `b50978581e378888…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 code already in run_pass2; Sweep, ablation, KS, fastest-mode calibration. Gates met, no Pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2.yaml` (11,588 B, `3,185` tok, SHA-256 `9736b95734ecb09a…`)
- **Artifact:** `morphoeval/runs/2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-loop-image-fitness.json` (2,208 B, `642` tok, SHA-256 `9b3bb8106d7c57c8…`)
