# MorphoEval Full-Loop Telemetry — `2026-10-27-dorudon-atrox-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-dorudon-atrox-turing-wavelength-s0`
- **Primary paper / dataset:** Dorudon atrox Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` only (no second assessor)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **0.88 s** (0.01 min) — Tier A: `0.44 s` · Tier B: `0.44 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **0.88** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** loop 3171 scout; loop 3171 scout done

### Step 2 — `paper` (Paper)
- **Notes:** loop 3171 paper; loop 3171 paper done

### Step 3 — `source` (Formal source)
- **Notes:** loop 3171 source; loop 3171 source done

### Step 4 — `rules` (Pin the rules)
- **Notes:** loop 3171 rules; loop 3171 rules done

### Step 5 — `script` (Script)
- **Notes:** loop 3171 script; loop 3171 script done

### Step 6 — `output` (Output & compare)
- **Notes:** loop 3171 output; loop 3171 output done

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** loop 3171 audit; loop 3171 audit done

### Step 8 — `measure` (Measure)
- **Notes:** loop 3171 measure; loop 3171 measure done

### Step 9 — `build` (Write as claims)
- **Notes:** loop 3171 build; loop 3171 build done

### Step 10 — `publish` (Sign & verify)
- **Notes:** loop 3171 publish; loop 3171 publish done

### Step 11 — `graph` (Public record)
- **Notes:** loop 3171 graph; loop 3171 graph done

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** loop 3171 remed; loop 3171 remed done

## Result

Pass 2 is the scored run. Pass 3 was not required.

*Dorudon atrox*, Priabonian 37.0 Ma, critical wavelength 253 µm. Onset λ_c is exactly 253 µm. With activator diffusivity held fixed, Dv/Du from 8 to 32 moves the fastest wavelength from 215.4 µm to 290.0 µm and σ_max from −0.135 to 0.254. Plate azimuthal fit: Pearson r = 0.9918 (polynomial-only baseline 0.8198), RMSE 0.0154, KS 0.09. The plate's secondary radial peak is 93.2 µm on its 100 µm bar; that is not the model wavelength.

Hindlimb domain at this fixed wavelength, as a scenario rather than a measured bud length: protocetid predecessor 8 λ, Dorudon 3 λ, crown-cetacean external hindlimb 0 λ.

| Gate | Result |
|---|---|
| image_fitness ≥ 70 | 88.8 |
| cv_sim_similarity ≥ 58 | 91.1 (plate); 79.6 (Wadi El-Hitan fossil, reported separately) |
| GIF frames ≥ 6 | 12 |
| resolution ≥ 340 px | 640×480 |
| non-circular specimen metric | Pearson 0.991801 vs 0.819761; secondary ring 93.24 µm vs 253 µm |
| Pearson r > 0.98 | 0.9918 |
| audit overall ≥ 85 | 87.6 (pass 2) |

Signing was not done. `~/.nanopub/profile.yml` is absent, so the nanopublications were checked for Project Delphi conformance and left unsigned. No trusty URI was invented.

The tracker wall-clock above is only the bookkeeping calls. The model self-test and the pass-2 render each took a few seconds.
