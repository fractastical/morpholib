# MorphoEval Full-Loop Telemetry — `2026-10-27-danio-nigrofasciatus-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-danio-nigrofasciatus-turing-wavelength-s0`
- **Primary paper / dataset:** Danio nigrofasciatus Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **12.02 s** (0.20 min) — Tier A: `11.51 s` · Tier B: `0.51 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **2,182** (input: `0`, output: `2,182`, subagent: `0`) — Tier A: `2,182` · Tier B: `0`
- **Total artifact tokens produced:** `2,182`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 11.13 | 0.00 | 0 | 2,182 | 0 | **2,182** | 2,182 | `scout_manifest.json`, `danio_nigrofasciatus_specimen.jpg` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.07 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **12.02** | **0.00** | **0** | **2,182** | **0** | **2,182** | **2,182** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/danio_nigrofasciatus_turing_wavelength_s0/scout_manifest.json` (7,403 B, `2,118` tok, SHA-256 `c69ce39cb42633e4…`)
- **Artifact:** `datasets/danio_nigrofasciatus_turing_wavelength_s0/danio_nigrofasciatus_specimen.jpg` (36,185 B, `64` tok, SHA-256 `edea58cd2541052f…`)

## Pass 2 result

Slot 3163 models *Danio nigrofasciatus* at 27 µm (2.71e-05 m, log10 = −4.567, tier `4_cellular_10um`), Holocene, age 0 Ma. The Schnakenberg dispersion relation selects that wavelength. What changes along the Danio chronoseries at this scale is anisotropy, not age: *D. margaritatus* (α = 1.05, spots), *D. nigrofasciatus* (α dorsal 6 / ventral 1.15, broken ventral stripes), *D. rerio* (α = 8.5, continuous stripes).

Gates after Pass 2, all met, so Pass 3 was not run:

- image_fitness = 87.9 (formula 0.55·CV + 0.30·specimen-match + 0.15·morphometric-validation)
- cv_sim_similarity = 92.2
- GIF 12 frames, 1120×480
- Pearson r of eigenmode growth versus analytical σ(k) = 0.999997 (reference floor 0.98, not equal to the value)
- Pass 2 audit overall = 86.1
- Non-circular specimen metrics include trunk coherency 0.6583 (reference 0.5), wavelength 27.33 µm (reference 27.1), and trunk KS 0.336 (reference 0.05)

The dorsoventral darkness profile of the photograph is not reproduced (r = 0.023). That miss is stated in the audit and is not claimed as a fit.

Signing was not performed. `python3 morphoeval/sign_morphoeval.py` fails because the `nanopub` package is not installed, and `~/.nanopub/profile.yml` is absent. No signature was fabricated. The knowledge-graph build also warns that `00-rubric.trig` reintroduces a rubric resource already published by earlier audits, so that file must not be signed without a supersession link.
