# MorphoEval Full-Loop Telemetry — `2026-10-27-apis-mellifera-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-apis-mellifera-turing-wavelength-s0`
- **Primary paper / dataset:** Apis mellifera Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `none`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **235.00 s** (3.92 min) — Tier A: `167.00 s` · Tier B: `68.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 40.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 20.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 15.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 50.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | builder-agent (`grok-4.7`) | `completed` | 12.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | builder-agent (`grok-4.7`) | `completed` | 25.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | builder-agent (`grok-4.7`) | `completed` | 4.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool | `completed` | 6.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool | `completed` | 1.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 2.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`grok-4.7`) | `completed` | 30.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **235.00** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** slot 3115; logged after the step finished

### Step 2 — `paper` (Paper)
- **Notes:** slot 3115; logged after the step finished

### Step 3 — `source` (Formal source)
- **Notes:** slot 3115; logged after the step finished

### Step 4 — `rules` (Pin the rules)
- **Notes:** slot 3115; logged after the step finished

### Step 5 — `script` (Script)
- **Notes:** slot 3115; logged after the step finished

### Step 6 — `output` (Output & compare)
- **Notes:** slot 3115; logged after the step finished

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** slot 3115; logged after the step finished

### Step 8 — `measure` (Measure)
- **Notes:** slot 3115; logged after the step finished

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform; Conformance OK for audit and audit-v2. Rubric nanopub reintroduced locally with --allow-reintroduction; not signed. Origin remote is not github.com/fractastical/morpholib, so commit-pinned loci were not verified against that remote.

### Step 10 — `publish` (Sign & verify)
- **Notes:** sign; Signing unavailable. ~/.nanopub/profile.yml is absent and morphoeval/sign_morphoeval.py fails on import (no requests/nanopub). No signatures were created.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md; Pass 2 REPORT overall 86.3. Image fitness 89.4, CV 96.0, Pearson r 0.99999999. All Pass-2 gates passed. Pass 3 not run.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** slot 3115; logged after the step finished

## Result

Pass 2 gates all hold. Pass 3 was not run.

- Scale 11.2 µm (1.12×10⁻⁵ m, log10 = −4.951), Holocene 0 Ma, Apis chronoseries 1/1. No predecessor or successor chronospecies. The developmental predecessor is equal diffusion, σ_max = −0.129.
- λ_c = 11.20 µm, nonlinear FFT wavelength 11.00 µm, k_c = 5.612×10⁵ m⁻¹. Pearson r of the finite-difference growth against the discrete dispersion symbol is 0.99999999 (RMSE 7.5×10⁻⁵).
- Audit v2 overall **86.3** (model fidelity 90, specimen match 80, morphometric validation 84). Pass 1 overall was 74.5 because the raw lattice does not look like the worker.
- Image fitness **89.4** = 0.55×96.0 + 0.30×80 + 0.15×84. GIF is 12 frames at 640×480. Specimen circularity 0.163 versus a circle null of 1. The dark-peak counter returns 9 against a typical visible band count of 4.
- The Ribi, Warrant & Zeil (2011) rhabdom long axis is 8.7 µm. Relative gap (11.2 − 8.7) / 8.7 = 0.287. That gap is reported, not fitted away.
- Camazine (1991) comb scale is about 5.2 mm. Wavenumber ratio versus this slot is about 464. Those parameters were not reused.
- Signing did not happen. There is no `~/.nanopub/profile.yml`, and `sign_morphoeval.py` cannot import `requests`. The knowledge graphs are unsigned. ProbKnow conformance of both graphs was OK.
- Step wall times in the table above are bookkeeping estimates entered after the work, not a profiler trace. `--self-test` and the render each finished in a few seconds.
