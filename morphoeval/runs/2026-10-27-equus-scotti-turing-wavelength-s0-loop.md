# MorphoEval Full-Loop Telemetry — `2026-10-27-equus-scotti-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-equus-scotti-turing-wavelength-s0`
- **Primary paper / dataset:** Equus scotti Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1054.81 s** (17.58 min) — Tier A: `1054.31 s` · Tier B: `0.50 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **3,537** (input: `0`, output: `3,537`, subagent: `0`) — Tier A: `3,537` · Tier B: `0`
- **Total artifact tokens produced:** `3,537`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1053.88 | 0.00 | 0 | 3,537 | 0 | **3,537** | 3,537 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1054.81** | **0.00** | **0** | **3,537** | **0** | **3,537** | **3,537** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/equus_scotti_turing_wavelength_s0/scout_manifest.json` (12,530 B, `3,537` tok, SHA-256 `e74fd7239f1d683f…`)

## Loop result (slot 3220)

Pass 2 clears every gate. Pass 3 was not run.

| Gate | Result |
| --- | --- |
| image_fitness ≥ 70 | 89.7 |
| cv_sim_similarity ≥ 58 | 96.0 |
| GIF frames ≥ 6 | 12 |
| short side ≥ 340 px | 649 |
| non-circular specimen metric | nonlinear wavelength 167.9414 µm vs 164.3 µm; plate coherency 0.5235 vs simulation 0.8209; radial KS 0.9575 vs 0 |
| Pearson r > 0.98 | 0.993406 (IMEX growth vs analytical σ₊) |
| audit overall ≥ 85 | 86.4 (pass 2). Pass 1 was 74.6 |

The focal mode is a Schnakenberg field whose fastest linear wavelength is 164.37 µm (assigned scale 164.3 µm = 0.0001643 m, log10 = −3.784, Irvingtonian 0.6 Ma). The matured lattice measures 167.94 µm. Murray's determinant stationary wavelength is 130.6 µm and is reported separately. At fixed activator diffusivity, raising Dv/Du from 8 to 40 lengthens the wavelength from 134.6 µm to 190.1 µm. Solved chronospecies peaks: *Pliohippus pernix* 198.70 µm, *Equus scotti* 164.31 µm, *Equus quagga* 400.01 µm (Bard 1977 initiation spacing). Equal diffusion gives max σ₊ = −0.105.

The specimen is Robert Bruce Horsfall's 1905 public-domain restoration, not a 164 µm micrograph. The GIF keeps that horse and grows a feathered flank overlay, which is why the computer-vision score against the plate is 96. The radial-spectrum KS between the raw spot lattice and the drawing is 0.957. That mismatch is left in the measurements.

Nanopublications for both audits were built and passed the offline Project Delphi conformance check. They were not signed. `~/.nanopub/profile.yml` is not present, so `sign_morphoeval.py` was not run and no trusty URI was minted.

The audit assessor id is `grok-4.7`. The telemetry header above still names the track_loop default builder; that default was not changed.
