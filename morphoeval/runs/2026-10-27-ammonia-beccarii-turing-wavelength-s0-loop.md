# MorphoEval Full-Loop Telemetry — `2026-10-27-ammonia-beccarii-turing-wavelength-s0-loop`

## Gates after pass 2

Pass 3 was not run. Every numeric gate cleared. Two process items stay open because this environment cannot close them.

| Gate | Result |
| --- | --- |
| image_fitness ≥ 70 | 88.5 |
| cv_sim_similarity ≥ 58 | 92.2 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 720×540 |
| non-circular specimen metric | bar-calibrated diameter 769.7 µm versus caption 600 µm |
| Pearson r > 0.98 | sweep 0.9949; dispersion 0.9999997 |
| audit overall ≥ 85 | 86.2 (pass 2) |

Signing was not done. `~/.nanopub/profile.yml` is absent, and `sign_morphoeval.py` cannot import `requests`. No signature was written.

R3 (second assessor) and R4 (Lean kernel check) remain open. Lean 4 is not installed, and no second model scored the audit.

- **Target simulation:** `delphi10000-ammonia-beccarii-turing-wavelength-s0`
- **Primary paper / dataset:** Ammonia beccarii Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `none`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.04 s** (0.02 min) — Tier A: `0.51 s` · Tier B: `0.54 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **41,359** (input: `0`, output: `41,359`, subagent: `0`) — Tier A: `17,958` · Tier B: `23,401`
- **Total artifact tokens produced:** `41,359`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,670 | 0 | **2,670** | 2,670 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 3,799 | 0 | **3,799** | 3,799 | `ammonia_beccarii_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Ammonia_beccarii_Hillewaert_2012.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,213 | 0 | **1,213** | 1,213 | `Loop_3108_ammonia_beccarii_turing_wavelength_s0.lean`, `delphi10000-ammonia-beccarii-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 10,084 | 0 | **10,084** | 10,084 | `2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `ammonia_beccarii_turing_wavelength_s0_growth.gif`, `ammonia_beccarii_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 3,609 | 0 | **3,609** | 3,609 | `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 5,391 | 0 | **5,391** | 5,391 | `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 3,385 | 0 | **3,385** | 3,385 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 6,856 | 0 | **6,856** | 6,856 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 4,160 | 0 | **4,160** | 4,160 | `2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-ammonia-beccarii-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.04** | **0.00** | **0** | **41,359** | **0** | **41,359** | **41,359** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Slot 3108 is outside campaign_100_slots.yaml. OpenAlex + Wikimedia fallback.; No collision on ammonia-beccarii-turing-wavelength-s0. Chose Hillewaert CC BY-SA plate.
- **Artifact:** `datasets/ammonia_beccarii_turing_wavelength_s0/scout_manifest.json` (9,698 B, `2,670` tok, SHA-256 `12dc86d6408aac7a…`)

### Step 2 — `paper` (Paper)
- **Notes:** Scale 31.37 µm = 3.137e-05 m, age 0 Ma, chronoseries 1/1.
- **Artifact:** `datasets/ammonia_beccarii_turing_wavelength_s0/ammonia_beccarii_turing_wavelength_s0_reference.json` (7,302 B, `2,053` tok, SHA-256 `6a6a4fb506092514…`)
- **Artifact:** `datasets/ammonia_beccarii_turing_wavelength_s0/paper_citations.bib` (3,277 B, `977` tok, SHA-256 `5ac93aba84d88754…`)
- **Artifact:** `datasets/ammonia_beccarii_turing_wavelength_s0/README.md` (2,921 B, `769` tok, SHA-256 `2258a4dfdcb921b6…`)

### Step 3 — `source` (Formal source)
- **Notes:** SHA-256 c15a90be96e2b088e80f55ac46f810f56c02bf2c7aaccce4bf483e1b7b3d2ade. Gallery Caryophyllia file rejected.
- **Artifact:** `datasets/ammonia_beccarii_turing_wavelength_s0/Ammonia_beccarii_Hillewaert_2012.jpg` (2,836,629 B, `64` tok, SHA-256 `c15a90be96e2b088…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Four Lean theorems, 0 sorry. Lean 4 binary not installed, so not kernel-checked.
- **Artifact:** `morphoeval/lean/loops/Loop_3108_ammonia_beccarii_turing_wavelength_s0.lean` (1,827 B, `64` tok, SHA-256 `8bf4fee855395105…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-ammonia-beccarii-turing-wavelength-s0.yaml` (4,038 B, `1,149` tok, SHA-256 `be69a05e1ff94638…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg λ_c = 31.37 µm, Dv/Du = 10.4, seed 3108.
- **Artifact:** `models/2026-delphi10000-ammonia-beccarii-turing-wavelength-s0.py` (35,190 B, `10,084` tok, SHA-256 `a30e98f98c590900…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test PASS. GIF 12 frames 720x540.
- **Artifact:** `docs/media/ammonia_beccarii_turing_wavelength_s0_growth.gif` (1,666,604 B, `64` tok, SHA-256 `2fea87438ec20b71…`)
- **Artifact:** `docs/media/ammonia_beccarii_turing_wavelength_s0_comparison.png` (813,963 B, `64` tok, SHA-256 `2f66f42e479d98dc…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 85.5. Remediations R1 diameter, R2 grid, R3 single assessor, R4 no Lean kernel.
- **Artifact:** `morphoeval/audits/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit.yaml` (13,997 B, `3,609` tok, SHA-256 `1c7c646c3227f938…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2 quick_gate 100/100, 14 measurements.
- **Artifact:** `morphoeval/audits/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit.measurements.yaml` (9,212 B, `2,709` tok, SHA-256 `3f354bf031fbf919…`)
- **Artifact:** `morphoeval/audits/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2.measurements.yaml` (9,411 B, `2,682` tok, SHA-256 `dfa0f3be1c69ff21…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2 conformance OK. Overall 86.2.
- **Artifact:** `morphoeval/kg/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2/REPORT.md` (11,738 B, `3,385` tok, SHA-256 `28dd4ec08cd08ffc…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml and sign_morphoeval.py cannot import requests. Nothing was signed.

### Step 11 — `graph` (Public record)
- **Notes:** Both REPORT.md files present.
- **Artifact:** `morphoeval/kg/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit/REPORT.md` (12,321 B, `3,471` tok, SHA-256 `cd7a4133d36efd45…`)
- **Artifact:** `morphoeval/kg/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2/REPORT.md` (11,738 B, `3,385` tok, SHA-256 `28dd4ec08cd08ffc…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1 and R2. image_fitness 88.5, CV 92.2, Pearson sweep r 0.9949, audit 86.2. R3 and R4 remain open. No pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-ammonia-beccarii-turing-wavelength-s0-audit-v2.yaml` (12,888 B, `3,423` tok, SHA-256 `2b6dc9e7e339f15e…`)
- **Artifact:** `morphoeval/runs/2026-10-27-ammonia-beccarii-turing-wavelength-s0-loop-image-fitness.json` (2,518 B, `737` tok, SHA-256 `3e201d0b8106ad51…`)
