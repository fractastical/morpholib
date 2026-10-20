# MorphoEval Full-Loop Telemetry — `2026-10-19-tgfb-fibronectin-limb-loop-pass3`

- **Target simulation:** `hentschel-tgfb-fibronectin-limb`
- **Primary paper / dataset:** Hentschel et al. (2004) J. Theor. Biol. — Dynamical mechanisms for skeletal pattern formation in the vertebrate limb
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `469bf3c`
- **Completed steps:** `7 / 12`
- **Total wall-clock time:** **2055.89 s** (34.26 min) — Tier A: `1787.03 s` · Tier B: `268.87 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **210,937** (input: `472`, output: `210,465`, subagent: `0`) — Tier A: `179,102` · Tier B: `31,835`
- **Total artifact tokens produced:** `54,018`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | claude-fable-5-1 (`gemini-4-argon`) | `completed` | 1681.97 | 0.00 | 472 | 176,288 | 0 | **176,760** | 35,229 | `2004-hentschel-tgfb-fibronectin-limb.py`, `Loop_014_tgfb_fibronectin_limb.lean`, `hentschel-tgfb-fibronectin-limb.yaml`, +3 more |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`gemini-4-argon`) | `completed` | 105.06 | 0.00 | 0 | 2,342 | 0 | **2,342** | 192 | `tgfb_fibronectin_limb_comparison.png`, `tgfb_fibronectin_limb_growth.gif`, `tgfb_fibronectin_limb_cpm_sorting.gif` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 | `completed` | 257.85 | 0.00 | 0 | 24,560 | 0 | **24,560** | 8,055 | `2026-10-19-tgfb-fibronectin-limb-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 | `completed` | 8.13 | 0.00 | 0 | 571 | 0 | **571** | 3,943 | `2026-10-19-tgfb-fibronectin-limb-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 | `completed` | 1.96 | 0.00 | 0 | 6,599 | 0 | **6,599** | 6,599 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 | `completed` | 0.84 | 0.00 | 0 | 105 | 0 | **105** | 0 | — |
| 11 | B | **graph** (Public record) | claude-fable-5-1 | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `7/12` | **2055.89** | **0.00** | **472** | **210,465** | **0** | **210,937** | **54,018** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 remediation: Kiskowski et al. (2004) cellular Potts (GGH) lattice coupled to the continuum 4-field PDE; Pass 3: GGH cellular Potts lattice coupled to continuum (Hill-switch cell state, two-way fibronectin), 4 regimes + wing bud; volume-filling micromass fix; clip_fraction_R collapse diagnostic; Lean Laws 4-5; 12-check self-test; 14 measurements
- **Artifact:** `2004-hentschel-tgfb-fibronectin-limb.py` (95,945 B, `27,369` tok, SHA-256 `7dacbc8c85b4478d…`)
- **Artifact:** `morphoeval/lean/loops/Loop_014_tgfb_fibronectin_limb.lean` (10,549 B, `64` tok, SHA-256 `83a98e97b144b625…`)
- **Artifact:** `morphoeval/math_kg_loops/hentschel-tgfb-fibronectin-limb.yaml` (11,391 B, `3,128` tok, SHA-256 `ea490c0f884e2930…`)
- **Artifact:** `datasets/tgfb_fibronectin_limb/kiskowski_ggh_cpm_reference.json` (6,273 B, `1,771` tok, SHA-256 `a7d13368e41d762a…`)
- **Artifact:** `datasets/tgfb_fibronectin_limb/paper_citations.bib` (4,621 B, `1,340` tok, SHA-256 `c2be675a1125adf7…`)
- **Artifact:** `datasets/tgfb_fibronectin_limb/README.md` (5,287 B, `1,557` tok, SHA-256 `43f4abb7d9d0b9fe…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 3 self-test + render; 12/12 self-test checks pass; 12-panel PNG + 2 GIFs rendered (~12 s)
- **Artifact:** `docs/media/tgfb_fibronectin_limb_comparison.png` (853,011 B, `64` tok, SHA-256 `f0533eb3a1514405…`)
- **Artifact:** `docs/media/tgfb_fibronectin_limb_growth.gif` (133,489 B, `64` tok, SHA-256 `cf2b1e6137130183…`)
- **Artifact:** `docs/media/tgfb_fibronectin_limb_cpm_sorting.gif` (224,663 B, `64` tok, SHA-256 `a8ec71c77f252e52…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 3 audit v3 authored (line-pinned, 7 dims re-scored); overall 82.6 (v2 92.6): GGH coupling closed; chemotactic collapse of 1D/2D/micromass continuum exposed; 7 Pass 4 items; build --check valid
- **Artifact:** `morphoeval/audits/2026-10-19-tgfb-fibronectin-limb-audit-v3.yaml` (31,259 B, `8,055` tok, SHA-256 `0cee2b1aefbe8088…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs on v3 audit; 14 measurements, 2 outputs fingerprinted, quick gate 100, missing []
- **Artifact:** `morphoeval/audits/2026-10-19-tgfb-fibronectin-limb-audit-v3.measurements.yaml` (14,405 B, `3,943` tok, SHA-256 `8ac1934606c1daaf…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build KG v3 superseding v2; KG built with supersession from v2; REPORT.md emitted
- **Artifact:** `morphoeval/kg/2026-10-19-tgfb-fibronectin-limb-audit-v3/REPORT.md` (23,962 B, `6,599` tok, SHA-256 `5bf0d9f9b9cedb2c…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** sign only (not published); 10 nanopubs signed and verified with check_signed.mjs; not published

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md check: overall score + superseded id; REPORT.md shows overall 82.6 and supersession chain audit -> v2 -> v3
