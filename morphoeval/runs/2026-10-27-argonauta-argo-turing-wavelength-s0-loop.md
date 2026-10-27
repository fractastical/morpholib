# MorphoEval Full-Loop Telemetry — `2026-10-27-argonauta-argo-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-argonauta-argo-turing-wavelength-s0`
- **Primary paper / dataset:** Argonauta argo Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **315.00 s** (5.25 min) — Tier A: `210.00 s` · Tier B: `105.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 60.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 5.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 10.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 25.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **315.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Crossref and Wikimedia. OpenAlex HTTP 429. No slug collision.

### Step 2 — `paper` (Paper)
- **Notes:** Equations, 95 um tier, age 0 Ma, Finn claim boundary.

### Step 3 — `source` (Formal source)
- **Notes:** Gallery plate and CC0 Di Turner shell, SHA-256 pinned.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Five Lean theorems, zero sorry; math KG yaml.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg critical mode, self-test r=0.99934.

### Step 6 — `output` (Output & compare)
- **Notes:** Smooth CV 80.2; grain 0.40 CV 86.2; 12-frame 800x600 GIF.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 81.3 on the smooth control.

### Step 8 — `measure` (Measure)
- **Notes:** 10 measurements, quick_gate 100.

### Step 9 — `build` (Write as claims)
- **Notes:** KG conformance OK with --rubric-from the shared rubric.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: no ~/.nanopub/profile.yml.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md pass 1 overall 81.3 and pass 2 overall 86.5.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 grain sweep and grained coherency. Gates met. Pass 3 not run.

## Result

Pass 2 meets the gates. Pass 3 was not run.

- Pearson r of the radial profile: 0.99934
- Inhibitory halo: 85.87 µm, compared with the 95 µm tier label (not set equal)
- Peak-to-peak period: 170.03 µm
- CV similarity: 80.2 on the smooth control, 86.2 on the growth GIF (grain amplitude 0.40)
- Image fitness (pass 2): 0.55×86.2 + 0.30×84 + 0.15×87 = 85.7
- Audit overall: 81.3 (pass 1), 86.5 (pass 2)
- GIF: 12 frames, 800×600
- Signing: not done. `nanopub` is not installed and `~/.nanopub/profile.yml` is absent. No signature was written.
- Scores are from grok-4.7 only. No second assessor scored this loop.
- Chronoseries: Argonauta 1/1, Holocene, 0 Ma. No predecessor node.

