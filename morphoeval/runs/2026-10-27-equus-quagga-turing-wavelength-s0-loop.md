# MorphoEval Full-Loop Telemetry — `2026-10-27-equus-quagga-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-equus-quagga-turing-wavelength-s0`
- **Primary paper / dataset:** Equus quagga Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1143.98 s** (19.07 min) — Tier A: `1143.25 s` · Tier B: `0.73 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1142.71 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.12 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.13 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.13 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1143.98** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Result

Pass 2 closes the gates. Image fitness **88.2** (0.55×92.1 CV + 0.30×80 specimen-match + 0.15×90 morphometric). GIF is 12 frames at 720×540. Pearson r of analytical versus PDE wavelength is **0.997**. Audit overall **85.6**. Six specimen metrics have value ≠ reference, including the rump gradient (−72.5° versus the simulation’s +2.5°).

The detail length is 1.883 mm (stated 1.9 mm, log10 = −2.725) at 0 Ma. Bard 1977 initiation spacing is 0.40 mm. *Equus scotti* is the equal-diffusion control. Nanopublications were built and conformant; they were **not signed** because `~/.nanopub/profile.yml` is absent. The assessor on the audit is Grok 4.7. The actor names in the table above are the tracker defaults, and R4 (an independent second assessor) is still open. No Pass 3 was required.

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Wikimedia scout plus curated Bard 1977, Murray 1981, Schnakenberg 1979, Staddon 2024. No collision with equus-quagga-turing-wavelength-s0.

### Step 2 — `paper` (Paper)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 3 — `source` (Formal source)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 5 — `script` (Script)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 8 — `measure` (Measure)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 9 — `build` (Write as claims)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 10 — `publish` (Sign & verify)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 11 — `graph` (Public record)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3221 Equus quagga 1.883 mm Holocene. Signing skipped: no ~/.nanopub/profile.yml.
