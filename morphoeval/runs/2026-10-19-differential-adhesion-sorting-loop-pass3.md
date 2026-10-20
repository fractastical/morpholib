# MorphoEval Full-Loop Telemetry — `2026-10-19-differential-adhesion-sorting-loop-pass3`

- **Target simulation:** `steinberg-differential-adhesion-sorting`
- **Primary paper / dataset:** Steinberg (1963) Science — Reconstruction of tissues by dissociated cells
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `7e4f207`
- **Completed steps:** `7 / 12`
- **Total wall-clock time:** **1893.88 s** (31.56 min) — Tier A: `941.48 s` · Tier B: `952.40 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **389,316** (input: `4,783`, output: `384,533`, subagent: `0`) — Tier A: `296,645` · Tier B: `92,671`
- **Total artifact tokens produced:** `84,912`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`claude-fable-5-1`) | `completed` | 806.73 | 0.00 | 3,585 | 270,739 | 0 | **274,324** | 31,516 | `1964-steinberg-differential-adhesion-sorting.py`, `Loop_041_differential_adhesion_sorting.lean`, `steinberg-differential-adhesion-sorting.yaml`, +3 more |
| 6 | A | **output** (Output & compare) | tool (`claude-fable-5-1`) | `completed` | 134.75 | 0.00 | 48 | 22,273 | 0 | **22,321** | 128 | `differential_adhesion_sorting_comparison.png`, `differential_adhesion_sorting_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | assessor (`claude-fable-5-1`) | `completed` | 801.81 | 0.00 | 704 | 53,349 | 0 | **54,053** | 8,084 | `2026-10-19-differential-adhesion-sorting-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | tool (`claude-fable-5-1`) | `completed` | 70.73 | 0.00 | 188 | 17,815 | 0 | **18,003** | 2,282 | `2026-10-19-differential-adhesion-sorting-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (`claude-fable-5-1`) | `completed` | 33.51 | 0.00 | 70 | 7,736 | 0 | **7,806** | 35,069 | `REPORT.md`, `morphoeval_2026-10-19-differential-adhesion-sorting-audit-v3.ttl` |
| 10 | B | **publish** (Sign & verify) | tool (`claude-fable-5-1`) | `completed` | 22.49 | 0.00 | 188 | 10,098 | 0 | **10,286** | 1,281 | `published.json` |
| 11 | B | **graph** (Public record) | tool (`claude-fable-5-1`) | `completed` | 23.86 | 0.00 | 0 | 2,523 | 0 | **2,523** | 6,552 | `REPORT.md`, `published.json` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `7/12` | **1893.88** | **0.00** | **4,783** | **384,533** | **0** | **389,316** | **84,912** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 remediation: 3D Cellular Potts lattice vs 2D cross-section sorting exponents; Pass 3 remediation: simulate_potts_regime_3d (64^3 sublattice-parallel CPM), fit_sorting_kinetics (log vs power law), compare_2d_vs_3d_sorting_kinetics (3D full / z-section / swap ablation / 3D control); 3 new Lean theorems (7 total, 0 sorry); 15 measurements; 13 self-test checks
- **Artifact:** `1964-steinberg-differential-adhesion-sorting.py` (82,412 B, `23,499` tok, SHA-256 `e75609a8dc2702c0…`)
- **Artifact:** `morphoeval/lean/loops/Loop_041_differential_adhesion_sorting.lean` (8,111 B, `64` tok, SHA-256 `a4b4a8f329f8bde3…`)
- **Artifact:** `morphoeval/math_kg_loops/steinberg-differential-adhesion-sorting.yaml` (8,833 B, `2,480` tok, SHA-256 `c2f18309da3f4794…`)
- **Artifact:** `datasets/differential_adhesion_sorting/differential_adhesion_sorting_reference.json` (12,052 B, `3,308` tok, SHA-256 `a881e19c58283599…`)
- **Artifact:** `datasets/differential_adhesion_sorting/paper_citations.bib` (3,189 B, `969` tok, SHA-256 `324000edc88409c3…`)
- **Artifact:** `datasets/differential_adhesion_sorting/README.md` (3,997 B, `1,196` tok, SHA-256 `202c32be008cc2ca…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test + full render (PNG/GIF) of Pass 3 script; 13/13 self-test checks pass (31 s); full render 29 s; PNG byte-identical across two runs (sha256 6046cb66...); GIF 18 frames x 4 panels incl. 3D z-section
- **Artifact:** `docs/media/differential_adhesion_sorting_comparison.png` (740,753 B, `64` tok, SHA-256 `6046cb66ae62af9e…`)
- **Artifact:** `docs/media/differential_adhesion_sorting_growth.gif` (381,536 B, `64` tok, SHA-256 `e8843b96a7cb9ea7…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 3 v3 audit re-scoring 7 rubric dimensions; v3 audit: 7 dims re-scored (93/87/90/91/91/93/92), 7 Pass-4 remediation items, --check valid; v3 audit written and --check valid: MF 90, SM 87, PE 91, MV 90, ST 91, CT 91, RP 93; v2 item closed; 7 Pass 4 remediation items
- **Artifact:** `morphoeval/audits/2026-10-19-differential-adhesion-sorting-audit-v3.yaml` (31,188 B, `8,084` tok, SHA-256 `852f8f3712200427…`)

### Step 8 — `measure` (Measure)
- **Notes:** 15 measurements captured incl. 7 Pass 3 (3D exponents, ablation, volume fraction); PNG+GIF fingerprinted; missing: []; measure_outputs: PNG+GIF fingerprinted, 15 measurements captured, missing: []
- **Artifact:** `morphoeval/audits/2026-10-19-differential-adhesion-sorting-audit-v3.measurements.yaml` (8,819 B, `2,282` tok, SHA-256 `ddf3126d56c8e3c0…`)

### Step 9 — `build` (Write as claims)
- **Notes:** 10 nanopubs, 787 triples, ProbKnow conformance OK, supersedes v2 KG; build_morphoeval_kg --supersede-from v2 --conform: 10 nanopubs, 781 triples, ProbKnow conformance OK
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v3/REPORT.md` (21,096 B, `5,863` tok, SHA-256 `cd6cf31a647f6efb…`)
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v3/morphoeval_2026-10-19-differential-adhesion-sorting-audit-v3.ttl` (103,461 B, `29,206` tok, SHA-256 `fe8eac647340b4bb…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** 10 nanopubs signed, 10/10 verified by check_signed.mjs (sign only, not published); sign_morphoeval: 10 nanopubs signed (not published); check_signed.mjs: 10 ok, 0 failed
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v3/published.json` (3,846 B, `1,281` tok, SHA-256 `ca9c845580d7bfb0…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md: overall 90.6 (faithful), supersedes audit-v2 recorded; REPORT.md verified: overall 89.9 (faithful), supersession chain v1 -> v2 -> v3 present
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v3/REPORT.md` (21,096 B, `5,863` tok, SHA-256 `cd6cf31a647f6efb…`)
- **Artifact:** `morphoeval/kg/2026-10-19-differential-adhesion-sorting-audit-v3/published.json` (2,006 B, `689` tok, SHA-256 `0ab0dfff18285717…`)
