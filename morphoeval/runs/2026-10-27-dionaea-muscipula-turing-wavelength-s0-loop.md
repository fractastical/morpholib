# MorphoEval Full-Loop Telemetry — `2026-10-27-dionaea-muscipula-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-dionaea-muscipula-turing-wavelength-s0`
- **Primary paper / dataset:** Dionaea muscipula Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **320.00 s** (5.33 min) — Tier A: `225.00 s` · Tier B: `95.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **36,950** (input: `0`, output: `36,950`, subagent: `0`) — Tier A: `16,785` · Tier B: `20,165`
- **Total artifact tokens produced:** `36,950`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 45.00 | 0.00 | 0 | 1,626 | 0 | **1,626** | 1,626 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 35.00 | 0.00 | 0 | 3,269 | 0 | **3,269** | 3,269 | `dionaea_muscipula_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `dionaea_muscipula_trap_lobe_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 1,015 | 0 | **1,015** | 1,015 | `Loop_3169_dionaea_muscipula_turing_wavelength_s0.lean`, `delphi10000-dionaea-muscipula-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 80.00 | 0.00 | 0 | 10,683 | 0 | **10,683** | 10,683 | `2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 128 | 0 | **128** | 128 | `dionaea_muscipula_turing_wavelength_s0_growth.gif`, `dionaea_muscipula_turing_wavelength_s0_comparison.png` |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 5,283 | 0 | **5,283** | 5,283 | `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit.yaml`, `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 8.00 | 0.00 | 0 | 3,555 | 0 | **3,555** | 3,555 | `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 5,149 | 0 | **5,149** | 5,149 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 5.00 | 0.00 | 0 | 5,149 | 0 | **5,149** | 5,149 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 1,029 | 0 | **1,029** | 1,029 | `2026-10-27-dionaea-muscipula-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **320.00** | **0.00** | **0** | **36,950** | **0** | **36,950** | **36,950** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex query plus Crossref DOI verification. Slot 3169 is not in campaign_100_slots.yaml. No collision with dionaea-muscipula-turing-wavelength-s0.
- **Artifact:** `datasets/dionaea_muscipula_turing_wavelength_s0/scout_manifest.json` (5,909 B, `1,626` tok, SHA-256 `732331a6145be952…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg kinetics, Turing dispersion, 24 µm scale, 0 Ma age.
- **Artifact:** `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_turing_wavelength_s0_reference.json` (3,968 B, `1,254` tok, SHA-256 `7976efe0424723e6…`)
- **Artifact:** `datasets/dionaea_muscipula_turing_wavelength_s0/paper_citations.bib` (3,196 B, `1,021` tok, SHA-256 `d5d325187b9bbe7d…`)
- **Artifact:** `datasets/dionaea_muscipula_turing_wavelength_s0/README.md` (3,729 B, `994` tok, SHA-256 `3f406da6d038e1ef…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Wikimedia Commons CC BY-SA 2.5 trap photograph and pinned SHA-256.
- **Artifact:** `datasets/dionaea_muscipula_turing_wavelength_s0/dionaea_muscipula_trap_lobe_specimen.jpg` (780,177 B, `64` tok, SHA-256 `6e5b87a782e373b3…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3169: six theorems, zero sorry. Math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3169_dionaea_muscipula_turing_wavelength_s0.lean` (1,870 B, `64` tok, SHA-256 `a94b61a02561f648…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-dionaea-muscipula-turing-wavelength-s0.yaml` (3,448 B, `951` tok, SHA-256 `b13f7044d7dc7a84…`)

### Step 5 — `script` (Script)
- **Notes:** Pass-1 field emitter and pass-2 sweep, ablation, segmentation, dispersion fit. --self-test passed.
- **Artifact:** `models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py` (37,138 B, `10,683` tok, SHA-256 `93287ea5099e9471…`)

### Step 6 — `output` (Output & compare)
- **Notes:** GIF 12 frames 640x480. Pass-1 CV 73.0; pass-2 CV 96.0.
- **Artifact:** `docs/media/dionaea_muscipula_turing_wavelength_s0_growth.gif` (1,838,248 B, `64` tok, SHA-256 `3e4f87775d70d2aa…`)
- **Artifact:** `docs/media/dionaea_muscipula_turing_wavelength_s0_comparison.png` (524,287 B, `64` tok, SHA-256 `65f76b67ce0b2da0…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass-1 audit overall 59.36. Pass-2 audit overall 85.58. R6 second assessor still open.
- **Artifact:** `morphoeval/audits/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit.yaml` (8,585 B, `2,249` tok, SHA-256 `03c8883ea071b47d…`)
- **Artifact:** `morphoeval/audits/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2.yaml` (11,302 B, `3,034` tok, SHA-256 `631632a717b24f71…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py pass 1 (quick_gate 41) and pass 2 (quick_gate 100).
- **Artifact:** `morphoeval/audits/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit.measurements.yaml` (3,649 B, `1,168` tok, SHA-256 `1d56966a4e57312e…`)
- **Artifact:** `morphoeval/audits/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2.measurements.yaml` (8,416 B, `2,387` tok, SHA-256 `3ef093009239cd2b…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform OK for audit and audit-v2. Shared rubric reintroduction left unsigned.
- **Artifact:** `morphoeval/kg/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit/REPORT.md` (7,308 B, `2,012` tok, SHA-256 `be6d265ef7d2ebfa…`)
- **Artifact:** `morphoeval/kg/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2/REPORT.md` (11,066 B, `3,137` tok, SHA-256 `d9fbe60f3a3b6dd5…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml and no nanopub package. Nothing signed.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for pass 1 and pass 2.
- **Artifact:** `morphoeval/kg/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit/REPORT.md` (7,308 B, `2,012` tok, SHA-256 `be6d265ef7d2ebfa…`)
- **Artifact:** `morphoeval/kg/2026-10-27-dionaea-muscipula-turing-wavelength-s0-audit-v2/REPORT.md` (11,066 B, `3,137` tok, SHA-256 `d9fbe60f3a3b6dd5…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1-R5. Gates passed: image_fitness 90.0, CV 96.0, Pearson r 0.999999, audit 85.58. No pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-dionaea-muscipula-turing-wavelength-s0-loop-image-fitness.json` (3,543 B, `1,029` tok, SHA-256 `d6c93699ef13d8af…`)

## Pass 2 gates

All gates passed. Pass 3 was not run.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 90.0 |
| cv_sim_similarity ≥ 58 | 96.0 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 480 |
| non-circular specimen metric | 5 (foreground, coherency, circularity, KS, stripe wavelength) |
| Pearson r > 0.98 | 0.999999 (dispersion); sweep r ≈ 1 with RMSE 7.47 µm from a 1.25 nonlinear shift |
| audit overall ≥ 85 | 85.58 (pass 1 was 59.36) |

Signing was not done. `~/.nanopub/profile.yml` is absent and `nanopub` is not installed. The TriG files are unsigned. R6 stays open: one assessor only.
