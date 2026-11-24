# MorphoEval Full-Loop Telemetry — `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-archaeopteris-hibernica-turing-wavelength-s0`
- **Primary paper / dataset:** Archaeopteris hibernica Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.97 s** (0.02 min) — Tier A: `0.48 s` · Tier B: `0.49 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **30,942** (input: `0`, output: `30,942`, subagent: `0`) — Tier A: `11,610` · Tier B: `19,332`
- **Total artifact tokens produced:** `30,942`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 940 | 0 | **940** | 940 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,423 | 0 | **1,423** | 1,423 | `archaeopteris_hibernica_turing_wavelength_s0_reference.json` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Archaeopteris_hibernica_NMNH.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `Loop_3119_archaeopteris_hibernica_turing_wavelength_s0.lean` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,055 | 0 | **9,055** | 9,055 | `2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `archaeopteris_hibernica_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,624 | 0 | **2,624** | 2,624 | `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,010 | 0 | **2,010** | 2,010 | `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,515 | 0 | **2,515** | 2,515 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 613 | 0 | **613** | 613 | `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-loop-image-fitness.json` |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,515 | 0 | **2,515** | 2,515 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 9,055 | 0 | **9,055** | 9,055 | `2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` |
| — | **All** | **TOTAL** | — | `12/12` | **0.97** | **0.00** | **0** | **30,942** | **0** | **30,942** | **30,942** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** slot 3119 scout; slot 3119 scout
- **Artifact:** `datasets/archaeopteris_hibernica_turing_wavelength_s0/scout_manifest.json` (3,378 B, `940` tok, SHA-256 `2250f09ace1abe2f…`)

### Step 2 — `paper` (Paper)
- **Notes:** slot 3119 paper; slot 3119 paper
- **Artifact:** `datasets/archaeopteris_hibernica_turing_wavelength_s0/archaeopteris_hibernica_turing_wavelength_s0_reference.json` (5,327 B, `1,423` tok, SHA-256 `655a409a6f13bf84…`)

### Step 3 — `source` (Formal source)
- **Notes:** slot 3119 source; slot 3119 source
- **Artifact:** `datasets/archaeopteris_hibernica_turing_wavelength_s0/Archaeopteris_hibernica_NMNH.jpg` (697,785 B, `64` tok, SHA-256 `20c480a93bc91bf3…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3119 rules; slot 3119 rules
- **Artifact:** `morphoeval/lean/loops/Loop_3119_archaeopteris_hibernica_turing_wavelength_s0.lean` (1,767 B, `64` tok, SHA-256 `f73651cbfaba4f2d…`)

### Step 5 — `script` (Script)
- **Notes:** slot 3119 script; slot 3119 script
- **Artifact:** `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` (32,094 B, `9,055` tok, SHA-256 `4ba0519ee0c20af1…`)

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3119 output; slot 3119 output
- **Artifact:** `docs/media/archaeopteris_hibernica_turing_wavelength_s0_growth.gif` (3,794,246 B, `64` tok, SHA-256 `d4f606fd4e0047da…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3119 audit; slot 3119 audit
- **Artifact:** `morphoeval/audits/2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2.yaml` (9,902 B, `2,624` tok, SHA-256 `d5d9c94f9cdf5dcf…`)

### Step 8 — `measure` (Measure)
- **Notes:** slot 3119 measure; slot 3119 measure
- **Artifact:** `morphoeval/audits/2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2.measurements.yaml` (7,430 B, `2,010` tok, SHA-256 `e56c2d58f207c938…`)

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3119 build; slot 3119 build
- **Artifact:** `morphoeval/kg/2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2/REPORT.md` (9,383 B, `2,515` tok, SHA-256 `c2d7687e3de78956…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3119 publish; slot 3119 publish
- **Artifact:** `morphoeval/runs/2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-loop-image-fitness.json` (2,050 B, `613` tok, SHA-256 `fe872fcb8664972b…`)

### Step 11 — `graph` (Public record)
- **Notes:** slot 3119 graph; slot 3119 graph
- **Artifact:** `morphoeval/kg/2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2/REPORT.md` (9,383 B, `2,515` tok, SHA-256 `c2d7687e3de78956…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3119 remed; slot 3119 remed
- **Artifact:** `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` (32,094 B, `9,055` tok, SHA-256 `4ba0519ee0c20af1…`)

## Scientific result (Pass 2)

Slot 3119, *Archaeopteris hibernica*, Famennian 370 Ma, cellular wavelength 31 µm. Schnakenberg kinetics with `k_c = argmax σ(k)`. The seeded steady state locks 31.0 µm. An unseeded run at the same γ selects 26.57 µm (finite-amplitude shift 4.43 µm). Dispersion Pearson r = 0.9865, RMSE = 0.0375. Equal-diffusion ablation ratio = 0.246. The gamma sweep is monotone (62 µm down to 15.5 µm analytical).

The specimen is the public-domain NMNH frond (`Archaeopteris_hibernica_NMNH.jpg`, SHA-256 `20c480a93bc91bf301127067e410d2ba9ecb359403975c64b7ced1918e65f3b1`). The growth GIF is 12 frames at 630×720. Computer-vision similarity of the last frame to that photograph is 96.0.

Pass 2 audit overall is 88.5. Image fitness is 92.1 = 0.55×96.0 + 0.30×86 + 0.15×90.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 92.1 |
| cv_sim_similarity ≥ 58 | 96.0 |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 630 |
| non-circular specimen metric | 5 (coherency, foreground, KS, radial RMSE, wavelength shift) |
| Pearson r > 0.98 | 0.9865 |
| audit overall ≥ 85 | 88.5 |

Pass 3 was not required. Nanopubs were built and passed the offline conformance check. They were not signed: `~/.nanopub/profile.yml` is not on this machine. The audit assessor is Grok 4.7. The assessor names in the telemetry header are the tracker defaults, not a second company.

Planation is the change at this age. *Tetraxylopteris schmidtii* (385 Ma, unplanated cortex) is reconstructed at 66 µm, this lamina at 31 µm, and *Elkinsia polymorpha* (365 Ma, thinner cupulate frond) at 21 µm. Those predecessor and successor wavelengths are the thickness scaling, not direct fossil measurements. The 31 µm figure is this slot's cellular scale, not a number quoted from Meyer-Berthaud, Scheckler and Wendt (1999).
