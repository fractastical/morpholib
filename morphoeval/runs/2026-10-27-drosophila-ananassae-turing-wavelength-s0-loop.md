# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-ananassae-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-ananassae-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila ananassae Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **102.01 s** (1.70 min) — Tier A: `102.01 s` · Tier B: `0.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **24,205** (input: `0`, output: `24,205`, subagent: `0`) — Tier A: `11,220` · Tier B: `12,985`
- **Total artifact tokens produced:** `24,205`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 102.01 | 0.00 | 0 | 1,384 | 0 | **1,384** | 1,384 | `scout_manifest.json`, `drosophila_ananassae_specimen.png` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.00 | 0.00 | 0 | 2,615 | 0 | **2,615** | 2,615 | `drosophila_ananassae_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `drosophila_ananassae_specimen.png` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.00 | 0.00 | 0 | 1,221 | 0 | **1,221** | 1,221 | `Loop_3176_drosophila_ananassae_turing_wavelength_s0.lean`, `delphi10000-drosophila-ananassae-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.00 | 0.00 | 0 | 5,808 | 0 | **5,808** | 5,808 | `2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_ananassae_turing_wavelength_s0_comparison.png`, `drosophila_ananassae_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.00 | 0.00 | 0 | 5,587 | 0 | **5,587** | 5,587 | `2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.00 | 0.00 | 0 | 1,712 | 0 | **1,712** | 1,712 | `2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.00 | 0.00 | 0 | 2,456 | 0 | **2,456** | 2,456 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.00 | 0.00 | 0 | 2,456 | 0 | **2,456** | 2,456 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.00 | 0.00 | 0 | 774 | 0 | **774** | 774 | `2026-10-27-drosophila-ananassae-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **102.01** | **0.00** | **0** | **24,205** | **0** | **24,205** | **24,205** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Crossref seed DOIs; PD Wikimedia female plate; no -s0 collision
- **Artifact:** `datasets/drosophila_ananassae_turing_wavelength_s0/scout_manifest.json` (4,997 B, `1,320` tok, SHA-256 `0a9bf3b0d2fae76c…`)
- **Artifact:** `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_specimen.png` (22,391 B, `64` tok, SHA-256 `e3f07821c201b2b5…`)

### Step 2 — `paper` (Paper)
- **Notes:** Schnakenberg kinetics, Turing dispersion, chronospecies wavelengths extracted into reference JSON
- **Artifact:** `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_turing_wavelength_s0_reference.json` (4,173 B, `1,145` tok, SHA-256 `2768c7dcb983c315…`)
- **Artifact:** `datasets/drosophila_ananassae_turing_wavelength_s0/paper_citations.bib` (2,572 B, `809` tok, SHA-256 `d4d761d6d8a25cae…`)
- **Artifact:** `datasets/drosophila_ananassae_turing_wavelength_s0/README.md` (2,698 B, `661` tok, SHA-256 `48b4b240e9246fde…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored PD Wikimedia File:Drosophila ananassae female.png SHA-256 pinned
- **Artifact:** `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_specimen.png` (22,391 B, `64` tok, SHA-256 `e3f07821c201b2b5…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Four Lean theorems, 0 sorry; math KG yaml
- **Artifact:** `morphoeval/lean/loops/Loop_3176_drosophila_ananassae_turing_wavelength_s0.lean` (1,980 B, `64` tok, SHA-256 `29388fbb6c76fcd5…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-ananassae-turing-wavelength-s0.yaml` (4,072 B, `1,157` tok, SHA-256 `73bd85a5f5f96d42…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 script with Schnakenberg solver and self-test
- **Artifact:** `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` (19,848 B, `5,808` tok, SHA-256 `a9775a4a6b135c92…`)

### Step 6 — `output` (Output & compare)
- **Notes:** PNG 1200x800 and 12-frame 960x640 GIF; CV similarity 94.3
- **Artifact:** `docs/media/drosophila_ananassae_turing_wavelength_s0_comparison.png` (555,209 B, `64` tok, SHA-256 `4a7d8f8ac9bfbd85…`)
- **Artifact:** `docs/media/drosophila_ananassae_turing_wavelength_s0_growth.gif` (3,444,176 B, `64` tok, SHA-256 `87c0dd9a019df28b…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 74.8; pass 2 overall 87.4 supersedes it
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit.yaml` (10,397 B, `2,691` tok, SHA-256 `8c1f93030d8d0f35…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2.yaml` (10,904 B, `2,896` tok, SHA-256 `4f68d604a9a34754…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2: 11 measurements, pearson_r 0.99483, quick_gate 90
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2.measurements.yaml` (6,300 B, `1,712` tok, SHA-256 `00357210509414e3…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Both audits conform. Rubric nanopub reintroduced with --allow-reintroduction; not signed.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2/REPORT.md` (8,919 B, `2,456` tok, SHA-256 `e05c35a7669eb7bc…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml missing, and sign_morphoeval.py raised ModuleNotFoundError: requests before the profile check. No signatures were produced.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 87.4 for the v2 audit
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2/REPORT.md` (8,919 B, `2,456` tok, SHA-256 `e05c35a7669eb7bc…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 fixed the four pass-1 gaps. image_fitness 91.2, cv 94.3, r 0.99483. Pass 3 not required.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-ananassae-turing-wavelength-s0-loop-image-fitness.json` (2,652 B, `774` tok, SHA-256 `e3364a8b8098af6b…`)

## Result (slot 3176)

*Drosophila ananassae*, Holocene, 0 Ma, detail-scale Turing wavelength 2.1 µm (2.106e-06 m). Predecessor *D. americana* is 139 µm (65.9× coarser). Successor *D. biarmipes* is 96 µm (45.6× coarser). The script does not copy the 85 µm sibling parameters.

Schnakenberg kinetics with a = 0.11, b = 1.15, Dv/Du = 19.7. The fastest linear mode is placed at 2.106 µm. The saturated FFT wavelength is 2.030 µm. Across six diffusion scales, Pearson r = 0.99483 and RMSE = 8.05e-08 m. Equal diffusion and linear kinetics decay; the nominal system grows.

The specimen is the public-domain Wikimedia plate `File:Drosophila ananassae female.png` (225×112), not the synthetic gallery spiral. Last-frame CV similarity against that plate is 94.3. The GIF is 12 frames at 960×640.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 91.2 |
| cv_sim_similarity ≥ 58 | 94.3 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 640 |
| non-circular specimen metric | yes (circularity 0.136 vs 1, Pearson 0.99483 vs 1, coherency gap 0.0099 vs 0) |
| Pearson r > 0.98 | 0.99483 |
| audit overall ≥ 85 | 87.4 (pass 1 was 74.8) |

Pass 3 was not run. Scores are from one assessor. `~/.nanopub/profile.yml` is absent, so the nanopubs are unsigned. The header lists a second assessor model because that name was passed at `track_loop.py init`; that model did not score this audit. The Lean file has four theorems and no `sorry`; no Lean binary was on PATH, so the kernel was not run.
