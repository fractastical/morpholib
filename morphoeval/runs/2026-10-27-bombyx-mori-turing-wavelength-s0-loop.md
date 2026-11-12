# MorphoEval Full-Loop Telemetry — `2026-10-27-bombyx-mori-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-bombyx-mori-turing-wavelength-s0`
- **Primary paper / dataset:** Bombyx mori Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1047.28 s** (17.45 min) — Tier A: `1042.15 s` · Tier B: `5.13 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **24,325** (input: `0`, output: `24,325`, subagent: `0`) — Tier A: `11,778` · Tier B: `12,547`
- **Total artifact tokens produced:** `24,325`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1037.92 | 0.00 | 0 | 1,100 | 0 | **1,100** | 1,100 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.84 | 0.00 | 0 | 1,718 | 0 | **1,718** | 1,718 | `bombyx_mori_turing_wavelength_s0_reference.json`, `paper_citations.bib` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.84 | 0.00 | 0 | 128 | 0 | **128** | 128 | `bombyx_mori_ocellus_specimen.jpg`, `bombyx_mori_adult_cc0.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.84 | 0.00 | 0 | 1,094 | 0 | **1,094** | 1,094 | `Loop_3134_bombyx_mori_turing_wavelength_s0.lean`, `delphi10000-bombyx-mori-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.85 | 0.00 | 0 | 7,610 | 0 | **7,610** | 7,610 | `2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.85 | 0.00 | 0 | 128 | 0 | **128** | 128 | `bombyx_mori_turing_wavelength_s0_comparison.png`, `bombyx_mori_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.85 | 0.00 | 0 | 4,840 | 0 | **4,840** | 4,840 | `2026-10-27-bombyx-mori-turing-wavelength-s0-audit.yaml`, `2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.86 | 0.00 | 0 | 1,840 | 0 | **1,840** | 1,840 | `2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.86 | 0.00 | 0 | 2,589 | 0 | **2,589** | 2,589 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.85 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.85 | 0.00 | 0 | 2,589 | 0 | **2,589** | 2,589 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.85 | 0.00 | 0 | 689 | 0 | **689** | 689 | `2026-10-27-bombyx-mori-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1047.28** | **0.00** | **0** | **24,325** | **0** | **24,325** | **24,325** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref: Yamaguchi 2013, Yoda 2014, Turing 1952, Gierer 1972. No s0 collision.
- **Artifact:** `datasets/bombyx_mori_turing_wavelength_s0/scout_manifest.json` (3,988 B, `1,100` tok, SHA-256 `9e4a322221a85be7…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON states 41 µm and 0 Ma.
- **Artifact:** `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_turing_wavelength_s0_reference.json` (4,869 B, `1,315` tok, SHA-256 `0a11d56294fb484c…`)
- **Artifact:** `datasets/bombyx_mori_turing_wavelength_s0/paper_citations.bib` (1,382 B, `403` tok, SHA-256 `08ab9cb9580075e6…`)

### Step 3 — `source` (Formal source)
- **Notes:** Ocellus card SHA 2adf5122…; CC0 adult SHA 76144516….
- **Artifact:** `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_ocellus_specimen.jpg` (96,142 B, `64` tok, SHA-256 `2adf51228d2da684…`)
- **Artifact:** `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_adult_cc0.jpg` (2,117,901 B, `64` tok, SHA-256 `76144516fb810472…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3134: 5 theorems, 0 sorry. λ·k identity and σ sign split.
- **Artifact:** `morphoeval/lean/loops/Loop_3134_bombyx_mori_turing_wavelength_s0.lean` (1,726 B, `64` tok, SHA-256 `1780905d5bd50e2d…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-bombyx-mori-turing-wavelength-s0.yaml` (3,670 B, `1,030` tok, SHA-256 `43c930e5f3c96191…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 script. self-test passed.
- **Artifact:** `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` (26,603 B, `7,610` tok, SHA-256 `30f858c5dd379d98…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 12 frames 640x480. CV similarity 95.2.
- **Artifact:** `docs/media/bombyx_mori_turing_wavelength_s0_comparison.png` (677,579 B, `64` tok, SHA-256 `150ea32fdbaf2b9a…`)
- **Artifact:** `docs/media/bombyx_mori_turing_wavelength_s0_growth.gif` (3,132,427 B, `64` tok, SHA-256 `1f2f5b66669db45e…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 82.5. Pass 2 overall 91.6 after scale-bar fix.
- **Artifact:** `morphoeval/audits/2026-10-27-bombyx-mori-turing-wavelength-s0-audit.yaml` (8,721 B, `2,307` tok, SHA-256 `0a494dd0f0994a6a…`)
- **Artifact:** `morphoeval/audits/2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2.yaml` (9,240 B, `2,533` tok, SHA-256 `c3fd0bbb91ba91cb…`)

### Step 8 — `measure` (Measure)
- **Notes:** Dispersion Pearson r 0.9923. Radial r 0.9998. Pitch 66.67 µm vs reference 41.
- **Artifact:** `morphoeval/audits/2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2.measurements.yaml` (6,419 B, `1,840` tok, SHA-256 `c059bad936ad8cfc…`)

### Step 9 — `build` (Write as claims)
- **Notes:** KG conformance OK for v1 and v2. --no-verify-loci because new files are not on the base commit.
- **Artifact:** `morphoeval/kg/2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2/REPORT.md` (8,987 B, `2,589` tok, SHA-256 `b47910c5e8878b4d…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and no trusty URI was invented.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 91.6.
- **Artifact:** `morphoeval/kg/2026-10-27-bombyx-mori-turing-wavelength-s0-audit-v2/REPORT.md` (8,987 B, `2,589` tok, SHA-256 `b47910c5e8878b4d…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** R1-R4 closed. image_fitness 93.6, cv 95.2, gates passed, no pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-bombyx-mori-turing-wavelength-s0-loop-image-fitness.json` (2,337 B, `689` tok, SHA-256 `63640617a230aef6…`)

## Pass 2 gates

Assessor for both audits is Grok 4.7, single agent. The actor names in the table above are the telemetry defaults.

Holocene *Bombyx mori*, age 0 Ma, chronoseries 1/1. Selected wavelength 41.016 µm. The ocellus card's own dark-ring pitch is 66.67 µm (reference 41 µm; the two are not set equal). Predecessor contrast: σ at 96 µm is −0.490.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 93.6 |
| cv_sim_similarity ≥ 58 | 95.2 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 480 (640×480) |
| non-circular specimen metric | pitch 66.67 µm ≠ 41 µm |
| Pearson r > 0.98 | dispersion 0.9923; radial 0.9998 |
| audit overall ≥ 85 | Pass 1: 82.5; Pass 2: 91.6 |

Pass 3 was not run. `~/.nanopub/profile.yml` is absent, so the nanopublications were conformance-checked and not signed.
