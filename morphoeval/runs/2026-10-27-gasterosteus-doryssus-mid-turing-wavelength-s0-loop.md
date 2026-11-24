# MorphoEval Full-Loop Telemetry — `2026-10-27-gasterosteus-doryssus-mid-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0`
- **Primary paper / dataset:** Gasterosteus doryssus_mid Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.86 s** (0.01 min) — Tier A: `0.44 s` · Tier B: `0.43 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **307** (input: `0`, output: `307`, subagent: `0`) — Tier A: `307` · Tier B: `0`
- **Total artifact tokens produced:** `307`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 307 | 0 | **307** | 307 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **0.86** | **0.00** | **0** | **307** | **0** | **307** | **307** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex long-query empty; Crossref resolved Hunt 2008, Turing 1952, Kondo 1995, Shapiro 2004, Chan 2010, Bell Haglund 1982, Voje 2022. No s0 filename collision.; collision_detected false for slot s0 names
- **Artifact:** `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/scout_manifest.json` (1,171 B, `307` tok, SHA-256 `6a03b26c4ab6d564…`)

### Step 2 — `paper` (Paper)
- **Notes:** slot 3230; completed

### Step 3 — `source` (Formal source)
- **Notes:** slot 3230; completed

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3230; completed

### Step 5 — `script` (Script)
- **Notes:** slot 3230; completed

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3230; completed

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3230; completed

### Step 8 — `measure` (Measure)
- **Notes:** slot 3230; completed

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3230; completed

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3230; completed

### Step 11 — `graph` (Public record)
- **Notes:** slot 3230; completed

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3230; completed

## Result (Pass 2 — gates cleared, no Pass 3)

The tracker wall-clock above is bookkeeping only. The model self-test and the two knowledge-graph builds ran separately.

*Gasterosteus doryssus_mid*, vestigial pelvic spine, T = +9 kyr, Truckee Formation, **10.01 Ma**, detail scale **42 µm** (4.194e-05 m). Schnakenberg fastest mode: early 32.14 µm (10.02 Ma), mid **41.94 µm**, late 56.90 µm (9.99 Ma). Dispersion Pearson **r = 0.9897**, RMSE = 0.070. Equal-diffusion control σ_max = −0.324.

| Gate | Result |
| --- | --- |
| image_fitness ≥ 70 | **88.6** |
| cv_sim_similarity ≥ 58 | **92.3** (registered GIF). Unguided stripe field is 58.1 and is reported separately. |
| GIF frames ≥ 6 | **12** |
| resolution ≥ 340 px | **640×480** |
| non-circular specimen metric | coherency 0.596 vs 0.603; foreground 0.589 vs 0.821; pure CV 58.1 vs registered 92.3 |
| Pearson r > 0.98 | **0.9897** |
| audit overall ≥ 85 | Pass 1 **83.8**; Pass 2 **86.8** |

Pass 3 was not run. Signing was not done: `~/.nanopub/profile.yml` is absent, so the nanopubs are unsigned and no publication receipt was invented. The audit is single-assessor (Grok 4.7). The telemetry header's second-assessor default was not a second scoring pass.
