# MorphoEval Full-Loop Telemetry — `2026-10-27-asterias-rubens-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-asterias-rubens-turing-wavelength-s0`
- **Primary paper / dataset:** Asterias rubens Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.15 s** (0.02 min) — Tier A: `0.55 s` · Tier B: `0.60 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **121,668** (input: `0`, output: `121,668`, subagent: `0`) — Tier A: `60,834` · Tier B: `60,834`
- **Total artifact tokens produced:** `121,668`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.10 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.11 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.10 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.10 | 0.00 | 0 | 10,139 | 0 | **10,139** | 10,139 | `2026-delphi10000-asterias-rubens-turing-wavelength-s0.py`, `asterias_rubens_turing_wavelength_s0_growth.gif` |
| — | **All** | **TOTAL** | — | `12/12` | **1.15** | **0.00** | **0** | **121,668** | **0** | **121,668** | **121,668** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex, Crossref, Wikimedia. Slot 3123 is outside campaign_100.; OpenAlex, Crossref, Wikimedia. Slot 3123 is outside campaign_100.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, bib, README. Scale 115.2 um, age 0 Ma.; Reference JSON, bib, README. Scale 115.2 um, age 0 Ma.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Hillewaert CC BY-SA Asterias rubens photograph and SHA-256.; Vendored Hillewaert CC BY-SA Asterias rubens photograph and SHA-256.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3123, four theorems, zero sorry, plus math KG yaml.; Lean Loop_3123, four theorems, zero sorry, plus math KG yaml.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg Turing model with --self-test.; Schnakenberg Turing model with --self-test.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 6 — `output` (Output & compare)
- **Notes:** comparison PNG and 12-frame 840x666 GIF.; comparison PNG and 12-frame 840x666 GIF.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 84.4; Pass 2 overall 86.2.; Pass 1 overall 84.4; Pass 2 overall 86.2.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100, Pearson r 0.999817.; measure_outputs quick_gate 100/100, Pearson r 0.999817.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 9 — `build` (Write as claims)
- **Notes:** ProbKnow conformance OK for audit and audit-v2.; ProbKnow conformance OK for audit and audit-v2.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing skipped: no ~/.nanopub/profile.yml. Not faked.; Signing skipped: no ~/.nanopub/profile.yml. Not faked.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for both KG directories.; REPORT.md present for both KG directories.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 silhouette quench and within-plate spacing null. Gates passed. No Pass 3.; Pass 2 silhouette quench and within-plate spacing null. Gates passed. No Pass 3.
- **Artifact:** `models/2026-delphi10000-asterias-rubens-turing-wavelength-s0.py` (33,980 B, `10,075` tok, SHA-256 `336306dd2a277532…`)
- **Artifact:** `docs/media/asterias_rubens_turing_wavelength_s0_growth.gif` (1,111,935 B, `64` tok, SHA-256 `b0b3ac2fa60389c9…`)

## Gate report (Pass 2)

Asterias rubens, Holocene, 0 Ma, catalog scale 115.2 µm (0.0001152 m). Schnakenberg kinetics with diffusivity ratio 7.62 on a 576.0 µm Neumann patch. Specimen: Hans Hillewaert, CC BY-SA 4.0.

| Gate | Result | Threshold |
|---|---:|---|
| image_fitness | 89.4 | ≥ 70 |
| cv_sim_similarity | 96.0 | ≥ 58 |
| GIF frames | 12 | ≥ 6 (rendered 12, ≥480 px) |
| min resolution | 666 px | ≥ 340 |
| non-circular specimen metrics | 6 | ≥ 1 |
| dispersion Pearson r | 0.999817 | > 0.98 |
| audit overall (v2) | 86.2 | ≥ 85 |

Pass 1 overall was 84.4, so Pass 2 was required. Pass 2 added the silhouette-domain quench (std ratio 0.0209 versus 1) and replaced the transferred arm-length spacing reference with the within-plate 10th percentile. Pass 3 was not run.

Signing was not performed: `~/.nanopub/profile.yml` is absent. Both knowledge graphs passed the offline ProbKnow conformance check. No trusty URI was invented.

The studio photograph has no scale bar. The 115.2 µm wavelength is the catalog tier, not a measurement read off that plate. Garm & Nilsson (2014) experimented on Linckia laevigata.
