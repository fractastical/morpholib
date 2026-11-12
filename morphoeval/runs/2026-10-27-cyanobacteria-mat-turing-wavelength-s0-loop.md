# MorphoEval Full-Loop Telemetry — `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-cyanobacteria-mat-turing-wavelength-s0`
- **Primary paper / dataset:** Cyanobacteria mat Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.93 s** (0.02 min) — Tier A: `0.46 s` · Tier B: `0.47 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **36,653** (input: `0`, output: `36,653`, subagent: `0`) — Tier A: `17,441` · Tier B: `19,212`
- **Total artifact tokens produced:** `36,653`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 3,070 | 0 | **3,070** | 3,070 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,357 | 0 | **2,357** | 2,357 | `cyanobacteria_mat_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `cyanobacteria_mat_holocene_culture_plate.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,140 | 0 | **1,140** | 1,140 | `Loop_3158_cyanobacteria_mat_turing_wavelength_s0.lean`, `delphi10000-cyanobacteria-mat-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 10,682 | 0 | **10,682** | 10,682 | `2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 128 | 0 | **128** | 128 | `cyanobacteria_mat_turing_wavelength_s0_growth.gif`, `cyanobacteria_mat_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 3,623 | 0 | **3,623** | 3,623 | `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 1,708 | 0 | **1,708** | 1,708 | `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 3,207 | 0 | **3,207** | 3,207 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 3,207 | 0 | **3,207** | 3,207 | `REPORT.md` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.08 | 0.00 | 0 | 3,210 | 0 | **3,210** | 3,210 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 4,257 | 0 | **4,257** | 4,257 | `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-cyanobacteria-mat-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **0.93** | **0.00** | **0** | **36,653** | **0** | **36,653** | **36,653** | — |

## Result

Pass 2 gates, measured on the culture-plate photograph and the 12-frame GIF:

- image_fitness **87.4** (0.55×90.7 + 0.30×82 + 0.15×86)
- cv_sim_similarity **90.7**
- GIF **12** frames, **640×520**
- dispersion Pearson r **0.999986**, RMSE **0.00558**
- audit v2 overall **87.1**
- specimen colony count 137 versus 65 spots; KS **0.453** (values differ from their references)

No gate failed, so there is no pass 3. Nanopub signing did not run: `~/.nanopub/profile.yml` is absent, and no signature was invented. The rubric audit is a single-assessor score. Step wall-clock values above are the bookkeeping intervals recorded when the steps were closed, not the solver runtime.

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex seed DOIs plus Wikimedia. No slug collision. Dresser Formation photos rejected (3.48 Ga).; OpenAlex seed DOIs plus Wikimedia. No slug collision. Dresser Formation photos rejected (3.48 Ga).
- **Artifact:** `datasets/cyanobacteria_mat_turing_wavelength_s0/scout_manifest.json` (10,589 B, `3,070` tok, SHA-256 `bd111d4deced0996…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg PDE, dispersion, 64.19 µm scale, 0 Ma chronoseries 1/1.; Extracted Schnakenberg PDE, dispersion, 64.19 µm scale, 0 Ma chronoseries 1/1.
- **Artifact:** `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_turing_wavelength_s0_reference.json` (3,769 B, `1,009` tok, SHA-256 `d166e23ba481aac8…`)
- **Artifact:** `datasets/cyanobacteria_mat_turing_wavelength_s0/paper_citations.bib` (2,555 B, `775` tok, SHA-256 `17b11feedc2a1e1d…`)
- **Artifact:** `datasets/cyanobacteria_mat_turing_wavelength_s0/README.md` (2,101 B, `573` tok, SHA-256 `2e53287bac639afd…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored gallery culture plate; SHA-256 pinned. Not the Paleoarchean Wikimedia fossils.; Vendored gallery culture plate; SHA-256 pinned. Not the Paleoarchean Wikimedia fossils.
- **Artifact:** `datasets/cyanobacteria_mat_turing_wavelength_s0/cyanobacteria_mat_holocene_culture_plate.jpg` (287,785 B, `64` tok, SHA-256 `858930a5ad520ee4…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3158: five theorems, zero sorry. Math KG yaml.; Lean Loop_3158: five theorems, zero sorry. Math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3158_cyanobacteria_mat_turing_wavelength_s0.lean` (2,033 B, `64` tok, SHA-256 `8b6613ffe9a441d0…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-cyanobacteria-mat-turing-wavelength-s0.yaml` (3,920 B, `1,076` tok, SHA-256 `b0eebc057ddee580…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg spectral solver, self-test, sweep, ablation, KS, eigenvector GoF.; Schnakenberg spectral solver, self-test, sweep, ablation, KS, eigenvector GoF.
- **Artifact:** `models/2026-delphi10000-cyanobacteria-mat-turing-wavelength-s0.py` (37,664 B, `10,682` tok, SHA-256 `e7a5a054568e7cd9…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 12 frames 640x520. cv_sim_similarity 90.7. Pearson r 0.999986.; Self-test passed. GIF 12 frames 640x520. cv_sim_similarity 90.7. Pearson r 0.999986.
- **Artifact:** `docs/media/cyanobacteria_mat_turing_wavelength_s0_growth.gif` (4,613,466 B, `64` tok, SHA-256 `2df1fd841d11f51f…`)
- **Artifact:** `docs/media/cyanobacteria_mat_turing_wavelength_s0_comparison.png` (619,956 B, `64` tok, SHA-256 `29d2a0dd6b33420d…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit overall 85.6. Four remediation items recorded.; Pass 1 audit overall 85.6. Four remediation items recorded.
- **Artifact:** `morphoeval/audits/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit.yaml` (14,624 B, `3,623` tok, SHA-256 `c6af47c65e9206b2…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100. Nine non-equal reference pairs.; measure_outputs quick_gate 100/100. Nine non-equal reference pairs.
- **Artifact:** `morphoeval/audits/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit.measurements.yaml` (6,034 B, `1,708` tok, SHA-256 `694675ee06fab616…`)

### Step 9 — `build` (Write as claims)
- **Notes:** KG build --conform OK after rubric supersede-from bos-taurus v7.; KG build --conform OK after rubric supersede-from bos-taurus v7.
- **Artifact:** `morphoeval/kg/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit/REPORT.md` (11,617 B, `3,207` tok, SHA-256 `88b7484959223fc9…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml. Signatures not fabricated.; Signing unavailable: no ~/.nanopub/profile.yml. Signatures not fabricated.
- **Artifact:** `morphoeval/kg/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit/REPORT.md` (11,617 B, `3,207` tok, SHA-256 `88b7484959223fc9…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 87.1 on the v2 graph. Conformance OK.; REPORT.md overall 87.1 on the v2 graph. Conformance OK.
- **Artifact:** `morphoeval/kg/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit-v2/REPORT.md` (11,460 B, `3,210` tok, SHA-256 `f986fe3ab3412e94…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R4. image_fitness 87.4, cv 90.7, pearson 0.999986. No pass 3.; Pass 2 closed R1-R4. image_fitness 87.4, cv 90.7, pearson 0.999986. No pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-audit-v2.yaml` (14,553 B, `3,652` tok, SHA-256 `f1d6ca6a5d8def2b…`)
- **Artifact:** `morphoeval/runs/2026-10-27-cyanobacteria-mat-turing-wavelength-s0-loop-image-fitness.json` (2,054 B, `605` tok, SHA-256 `1af6c3b6259c0a53…`)
