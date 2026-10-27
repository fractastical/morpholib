# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-miranda-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-miranda-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila miranda Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `claude-sonnet-5-5-high`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.14 s** (0.02 min) — Tier A: `0.56 s` · Tier B: `0.57 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **0** (input: `0`, output: `0`, subagent: `0`) — Tier A: `0` · Tier B: `0`
- **Total artifact tokens produced:** `0`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.08 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1.14** | **0.00** | **0** | **0** | **0** | **0** | **0** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex DOI resolution; slot 3193 is outside campaign_100. No slug collision with s0.; OpenAlex DOI resolution; slot 3193 is outside campaign_100. No slug collision with s0.

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, README state 1.9 µm and 0 Ma.; Reference JSON, BibTeX, README state 1.9 µm and 0 Ma.

### Step 3 — `source` (Formal source)
- **Notes:** Gallery plate vendored and SHA-256 pinned. Wikimedia hits were the Miranda protein and were rejected.; Gallery plate vendored and SHA-256 pinned. Wikimedia hits were the Miranda protein and were rejected.

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3193: six theorems, zero sorry. Math KG yaml written.; Lean Loop_3193: six theorems, zero sorry. Math KG yaml written.

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg model with --self-test. Self-test passed.; Schnakenberg model with --self-test. Self-test passed.

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 1 then pass 2 renders. GIF 12 frames 640x480. λ=1.864 µm. Pearson r=0.999999996.; Pass 1 then pass 2 renders. GIF 12 frames 640x480. λ=1.864 µm. Pearson r=0.999999996.

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 80.3 with five remediation items. Pass 2 overall 89.4.; Pass 1 overall 80.3 with five remediation items. Pass 2 overall 89.4.

### Step 8 — `measure` (Measure)
- **Notes:** Pass 1 sidecar: 2 metrics, quick_gate 47. Pass 2 sidecar: 12 metrics, quick_gate 90.; Pass 1 sidecar: 2 metrics, quick_gate 47. Pass 2 sidecar: 12 metrics, quick_gate 90.

### Step 9 — `build` (Write as claims)
- **Notes:** KG v1 and v2 built. Conformance OK with --allow-reintroduction because 00-rubric.trig is already published.; KG v1 and v2 built. Conformance OK with --allow-reintroduction because 00-rubric.trig is already published.

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. No signatures were fabricated.; Signing unavailable: ~/.nanopub/profile.yml is absent. No signatures were fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for audit and audit-v2. Pass 2 overall 89.4.; REPORT.md present for audit and audit-v2. Pass 2 overall 89.4.

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed sweep, ablation, plate segmentation, dispersion GOF, and CV depth sweep. No pass 3: all gates passed.; Pass 2 closed sweep, ablation, plate segmentation, dispersion GOF, and CV depth sweep. No pass 3: all gates passed.

## Gates (pass 2)

All pass-2 gates cleared, so pass 3 was not run.

| Gate | Result |
|---|---|
| image_fitness | 86.985 (threshold 70) |
| cv_sim_similarity | 86.7 (threshold 58) |
| GIF | 12 frames, 640×480 |
| non-circular specimen metrics | foreground 0.3703, coherency 0.806, circularity 0.0543, orientation KS 0.3333 |
| Pearson r | 0.999999996 |
| audit overall | 89.4 (pass 1 was 80.3) |

Signing was not performed: `~/.nanopub/profile.yml` is absent. The knowledge graphs were conformance-checked and left unsigned. No second assessor scored this loop; the second-assessor name on the telemetry header is the init default, not a completed review.
