# MorphoEval Full-Loop Telemetry — `2026-10-27-gekko-gecko-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-gekko-gecko-turing-wavelength-s0`
- **Primary paper / dataset:** Gekko gecko Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.99 s** (0.02 min) — Tier A: `0.49 s` · Tier B: `0.49 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **22,422** (input: `0`, output: `22,422`, subagent: `0`) — Tier A: `13,470` · Tier B: `8,952`
- **Total artifact tokens produced:** `22,422`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 633 | 0 | **633** | 633 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 2,574 | 0 | **2,574** | 2,574 | `gekko_gecko_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `Gekko_gecko_full_cc-by-sa.jpg`, `Gekko_gecko_dorsum_cc-by-sa.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 843 | 0 | **843** | 843 | `Loop_3235_gekko_gecko_turing_wavelength_s0.lean`, `delphi10000-gekko-gecko-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 9,164 | 0 | **9,164** | 9,164 | `2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 128 | 0 | **128** | 128 | `gekko_gecko_turing_wavelength_s0_growth.gif`, `gekko_gecko_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 4,602 | 0 | **4,602** | 4,602 | `2026-10-27-gekko-gecko-turing-wavelength-s0-audit.yaml`, `2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 1,759 | 0 | **1,759** | 1,759 | `2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 2,294 | 0 | **2,294** | 2,294 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 297 | 0 | **297** | 297 | `2026-10-27-gekko-gecko-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **0.99** | **0.00** | **0** | **22,422** | **0** | **22,422** | **22,422** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Crossref/Wikimedia scout; slot 3235 is outside campaign_100 so query-driven.; No collision with gekko-gecko-turing-wavelength-s0. Verified DOIs via Crossref.
- **Artifact:** `datasets/gekko_gecko_turing_wavelength_s0/scout_manifest.json` (2,194 B, `633` tok, SHA-256 `58372215ad167234…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, README state 242 um and 0 Ma.
- **Artifact:** `datasets/gekko_gecko_turing_wavelength_s0/gekko_gecko_turing_wavelength_s0_reference.json` (4,002 B, `1,181` tok, SHA-256 `b00877ae162067f0…`)
- **Artifact:** `datasets/gekko_gecko_turing_wavelength_s0/paper_citations.bib` (2,459 B, `717` tok, SHA-256 `e10f373a7d4af3fb…`)
- **Artifact:** `datasets/gekko_gecko_turing_wavelength_s0/README.md` (2,541 B, `676` tok, SHA-256 `f5c1f12f18d79073…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Dupont CC BY-SA 2.0 Tokay photo and dorsum crop; SHA-256 pinned.
- **Artifact:** `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_full_cc-by-sa.jpg` (858,000 B, `64` tok, SHA-256 `09aac21e60203080…`)
- **Artifact:** `datasets/gekko_gecko_turing_wavelength_s0/Gekko_gecko_dorsum_cc-by-sa.jpg` (63,194 B, `64` tok, SHA-256 `92d9092794d8c8df…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3235: six theorems, zero sorry. Math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3235_gekko_gecko_turing_wavelength_s0.lean` (1,893 B, `64` tok, SHA-256 `033b62ac881ccc8e…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-gekko-gecko-turing-wavelength-s0.yaml` (2,716 B, `779` tok, SHA-256 `edb300bccc0ce5bf…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg k_c calibrated to 242 um. Self-test 10/10.
- **Artifact:** `models/2026-delphi10000-gekko-gecko-turing-wavelength-s0.py` (32,103 B, `9,164` tok, SHA-256 `4cd9b2b01369da75…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 720x540 12 frames. Comparison PNG. Measured wavelength 232.994 um.
- **Artifact:** `docs/media/gekko_gecko_turing_wavelength_s0_growth.gif` (3,094,317 B, `64` tok, SHA-256 `724f685b336c401a…`)
- **Artifact:** `docs/media/gekko_gecko_turing_wavelength_s0_comparison.png` (813,471 B, `64` tok, SHA-256 `2da99ed0b8d8beee…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 79.9 with open R1-R6. Pass 2 overall 86.8.
- **Artifact:** `morphoeval/audits/2026-10-27-gekko-gecko-turing-wavelength-s0-audit.yaml` (8,801 B, `2,189` tok, SHA-256 `9fc351fb73652962…`)
- **Artifact:** `morphoeval/audits/2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2.yaml` (9,504 B, `2,413` tok, SHA-256 `69a72a9b38956b5d…`)

### Step 8 — `measure` (Measure)
- **Notes:** Sidecar: Pearson r>0.98, RMSE 10.079 um, specimen spacing 0.046 vs 0.0939.
- **Artifact:** `morphoeval/audits/2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2.measurements.yaml` (6,446 B, `1,759` tok, SHA-256 `9c74d2238c50330c…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 1 and Pass 2 KG --conform OK. Unsigned.
- **Artifact:** `morphoeval/kg/2026-10-27-gekko-gecko-turing-wavelength-s0-audit-v2/REPORT.md` (8,379 B, `2,294` tok, SHA-256 `b2d0925340f9ade1…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable. ~/.nanopub/profile.yml missing; sign_morphoeval.py also lacks the requests package. No signature was produced.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for audit and audit-v2. Supersession chain recorded.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4 and R6. R5 second assessor remains open. Gates passed; no Pass 3. image_fitness 85.9, CV 88.5.
- **Artifact:** `morphoeval/runs/2026-10-27-gekko-gecko-turing-wavelength-s0-loop-image-fitness.json` (912 B, `297` tok, SHA-256 `1111d0704bcf012e…`)

## Pass 2 gates

All gates passed on the Pass 2 render. No Pass 3.

- image_fitness **85.9** (gate >= 70), from 0.55×88.5 + 0.30×82 + 0.15×84
- cv_sim_similarity **88.5** (gate >= 58) against the Dupont dorsum crop
- GIF **12** frames at **720×540**
- specimen nearest-neighbour spacing fraction 0.046 versus render 0.0939 (not equal)
- specimen coherency 0.4594 versus render 0.734
- wavelength sweep Pearson r = 0.9999999999999989 (gate > 0.98), RMSE 10.079 µm; ring centroid 232.994 µm versus linear target 242 µm
- audit v2 overall **86.8** (gate >= 85)
- signing was not done: `~/.nanopub/profile.yml` is absent, and `sign_morphoeval.py` could not import `requests`. No trusty signature was written.
