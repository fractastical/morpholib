# MorphoEval Full-Loop Telemetry — `2026-10-27-ambulocetus-natans-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-ambulocetus-natans-turing-wavelength-s0`
- **Primary paper / dataset:** Ambulocetus natans Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.92 s** (0.02 min) — Tier A: `0.45 s` · Tier B: `0.48 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **27,884** (input: `0`, output: `27,884`, subagent: `0`) — Tier A: `13,200` · Tier B: `14,684`
- **Total artifact tokens produced:** `27,884`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,103 | 0 | **2,103** | 2,103 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 2,640 | 0 | **2,640** | 2,640 | `ambulocetus_natans_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 64 | 0 | **64** | 64 | `ambulocetus_natans_naturalis_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 914 | 0 | **914** | 914 | `Loop_3106_ambulocetus_natans_turing_wavelength_s0.lean`, `delphi10000-ambulocetus-natans-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 7,351 | 0 | **7,351** | 7,351 | `2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 128 | 0 | **128** | 128 | `ambulocetus_natans_turing_wavelength_s0_comparison.png`, `ambulocetus_natans_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 2,735 | 0 | **2,735** | 2,735 | `2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 1,519 | 0 | **1,519** | 1,519 | `2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 2,326 | 0 | **2,326** | 2,326 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 4,772 | 0 | **4,772** | 4,772 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 3,332 | 0 | **3,332** | 3,332 | `2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-ambulocetus-natans-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **0.92** | **0.00** | **0** | **27,884** | **0** | **27,884** | **27,884** | — |

## Result

Pass 2 gates all hold. Pass 3 was not run.

- image_fitness = 89.2 (0.55 * 91.6 + 0.30 * 84 + 0.15 * 91)
- cv_sim_similarity = 91.6 against the Naturalis mount
- GIF = 12 frames, 720×540
- Pearson r = 0.9997, RMSE = 0.0034
- Audit overall = 87.3 (pass 1), 88.9 (pass 2)
- Non-circular specimen metrics: foreground fraction 0.161 vs reference 0.114; coherency 0.359 vs 0.114; KS = 0.066 vs 0; emergent wavelength 177.4 µm vs 224 µm
- Nanopubs were not signed: `~/.nanopub/profile.yml` is absent. No trusty URI was invented.
- Residual, not a failed gate: the nonlinear peak stays near 177 µm rather than the analytic 224 µm target.

Step wall-clock values above are the telemetry bookkeeping intervals. The Schnakenberg self-test itself takes a few seconds.

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and Wikimedia scout; slot 3106 is outside the 100-slot campaign file; No s0 path collision. Sibling 1 cm script left untouched.
- **Artifact:** `datasets/ambulocetus_natans_turing_wavelength_s0/scout_manifest.json` (7,232 B, `2,103` tok, SHA-256 `f049426cc611a5e7…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations, parameters, 224 um, 47.5 Ma
- **Artifact:** `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_reference.json` (4,350 B, `1,207` tok, SHA-256 `0246234fefb8db6a…`)
- **Artifact:** `datasets/ambulocetus_natans_turing_wavelength_s0/paper_citations.bib` (2,259 B, `726` tok, SHA-256 `ec46df5e30a90905…`)
- **Artifact:** `datasets/ambulocetus_natans_turing_wavelength_s0/README.md` (2,562 B, `707` tok, SHA-256 `b7cec352efb7cab6…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Naturalis mount; CC BY-SA 3.0, Ghedoghedo, SHA-256 pinned
- **Artifact:** `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_naturalis_specimen.jpg` (372,743 B, `64` tok, SHA-256 `738e2188c8bb2936…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3106 and math KG yaml; 7 theorems, 0 sorry
- **Artifact:** `morphoeval/lean/loops/Loop_3106_ambulocetus_natans_turing_wavelength_s0.lean` (2,706 B, `64` tok, SHA-256 `a33d27670aaef737…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-ambulocetus-natans-turing-wavelength-s0.yaml` (3,160 B, `850` tok, SHA-256 `1a4703da36490d88…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg script with self-test
- **Artifact:** `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` (24,862 B, `7,351` tok, SHA-256 `0e0b1765f63151ae…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test and media; r=0.9997, 12 frames, 720x540
- **Artifact:** `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png` (248,114 B, `64` tok, SHA-256 `efd6b1735e46c2ca…`)
- **Artifact:** `docs/media/ambulocetus_natans_turing_wavelength_s0_growth.gif` (3,240,023 B, `64` tok, SHA-256 `d0ad8d52ca0864ae…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 rubric scores; Overall 87.3; remediations R1-R3
- **Artifact:** `morphoeval/audits/2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit.yaml` (10,542 B, `2,735` tok, SHA-256 `4d446da651a1ca41…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs pass 1
- **Artifact:** `morphoeval/audits/2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit.measurements.yaml` (5,475 B, `1,519` tok, SHA-256 `8ea27b2157df464c…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform; Conformance OK, 10 nanopubs
- **Artifact:** `morphoeval/kg/2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit/REPORT.md` (8,636 B, `2,326` tok, SHA-256 `9474ced5e8d02ddc…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing; No ~/.nanopub/profile.yml; nanopubs left unsigned

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present
- **Artifact:** `morphoeval/kg/2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit/REPORT.md` (8,636 B, `2,326` tok, SHA-256 `9474ced5e8d02ddc…`)
- **Artifact:** `morphoeval/kg/2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit-v2/REPORT.md` (8,797 B, `2,446` tok, SHA-256 `522e27fd38e1d28a…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 remediation and image fitness; Overall 88.9; image_fitness 89.2; cv 91.6; gates passed; pass 3 not required
- **Artifact:** `morphoeval/audits/2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit-v2.yaml` (10,278 B, `2,755` tok, SHA-256 `885322117fa51376…`)
- **Artifact:** `morphoeval/runs/2026-10-27-ambulocetus-natans-turing-wavelength-s0-loop-image-fitness.json` (1,953 B, `577` tok, SHA-256 `1a62b597a9eb8e9e…`)
