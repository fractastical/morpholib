# MorphoEval Full-Loop Telemetry — `2026-10-19-cortical-sulcal-axon-tension-loop`

- **Target simulation:** `2020-karbowski-cortical-sulcal-axon-tension.py`
- **Primary paper / dataset:** Van Essen (1997) Nature 385:313-318; Hilgetag & Barbas (2006) PLoS Comput. Biol. 2:e22; Mota & Herculano-Houzel (2015) Science 349:74-77
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `708cfeb`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **402.21 s** (6.70 min) — Tier A: `191.62 s` · Tier B: `210.59 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **227,449** (input: `424`, output: `227,025`, subagent: `0`) — Tier A: `92,813` · Tier B: `134,636`
- **Total artifact tokens produced:** `41,909`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1.69 | 0.00 | 0 | 682 | 0 | **682** | 682 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 164.71 | 0.00 | 188 | 72,534 | 0 | **72,722** | 499 | `paper_citations.bib` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.11 | 0.00 | 0 | 4,273 | 0 | **4,273** | 4,273 | `cortical_sulcal_axon_tension_reference.json`, `coronal_cortex_axon_tracts_specimen.jpg`, `README.md` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 24.01 | 0.00 | 0 | 12,069 | 0 | **12,069** | 1,492 | `Loop_090_cortical_sulcal_axon_tension.lean`, `karbowski-cortical-sulcal-axon-tension.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.39 | 0.00 | 0 | 2,939 | 0 | **2,939** | 2,939 | `2020-karbowski-cortical-sulcal-axon-tension.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.72 | 0.00 | 0 | 128 | 0 | **128** | 128 | `cortical_sulcal_axon_tension_comparison.png`, `cortical_sulcal_axon_tension_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 2,816 | 0 | **2,816** | 2,816 | `2026-10-19-cortical-sulcal-axon-tension-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.62 | 0.00 | 0 | 882 | 0 | **882** | 882 | `2026-10-19-cortical-sulcal-axon-tension-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.53 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.96 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.11 | 0.00 | 0 | 2,342 | 0 | **2,342** | 2,342 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 208.28 | 0.00 | 236 | 128,360 | 0 | **128,596** | 25,856 | `2020-karbowski-cortical-sulcal-axon-tension.py`, `cortical_sulcal_axon_tension_comparison.png`, `cortical_sulcal_axon_tension_growth.gif`, +3 more |
| — | **All** | **TOTAL** | — | `12/12` | **402.21** | **0.00** | **424** | **227,025** | **0** | **227,449** | **41,909** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/cortical_sulcal_axon_tension/scout_manifest.json` (2,483 B, `682` tok, SHA-256 `3ef66e67c6994708…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/cortical_sulcal_axon_tension/paper_citations.bib` (1,622 B, `499` tok, SHA-256 `bcd7b0529c1c497d…`)

### Step 3 — `source` (Formal source)
- **Artifact:** `datasets/cortical_sulcal_axon_tension/cortical_sulcal_axon_tension_reference.json` (10,745 B, `3,132` tok, SHA-256 `51d4ed1c896638d9…`)
- **Artifact:** `datasets/cortical_sulcal_axon_tension/coronal_cortex_axon_tracts_specimen.jpg` (41,762 B, `64` tok, SHA-256 `ff9a25e993fd5aad…`)
- **Artifact:** `datasets/cortical_sulcal_axon_tension/README.md` (3,448 B, `1,077` tok, SHA-256 `c7d4144ac95d051e…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_090_cortical_sulcal_axon_tension.lean` (5,653 B, `64` tok, SHA-256 `5b2c4d26b4a01353…`)
- **Artifact:** `morphoeval/math_kg_loops/karbowski-cortical-sulcal-axon-tension.yaml` (5,483 B, `1,428` tok, SHA-256 `72b119eb1afa912e…`)

### Step 5 — `script` (Script)
- **Artifact:** `2020-karbowski-cortical-sulcal-axon-tension.py` (10,568 B, `2,939` tok, SHA-256 `b2e1eb4af6e6c09c…`)

### Step 6 — `output` (Output & compare)
- **Artifact:** `docs/media/cortical_sulcal_axon_tension_comparison.png` (223,333 B, `64` tok, SHA-256 `c8f0ef043aac3299…`)
- **Artifact:** `docs/media/cortical_sulcal_axon_tension_growth.gif` (81,670 B, `64` tok, SHA-256 `e8969dc72361dbdc…`)

### Step 7 — `audit` (Dual-agent audit)
- **Artifact:** `morphoeval/audits/2026-10-19-cortical-sulcal-axon-tension-audit.yaml` (11,474 B, `2,816` tok, SHA-256 `3b30f7dbd9718dfd…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-19-cortical-sulcal-axon-tension-audit.measurements.yaml` (2,785 B, `882` tok, SHA-256 `a70340cfb1626d5d…`)

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-19-cortical-sulcal-axon-tension-audit/REPORT.md` (8,977 B, `2,342` tok, SHA-256 `681e23b5987e853b…`)

### Step 12 — `remed` (Remediate & re-score)
- **Artifact:** `2020-karbowski-cortical-sulcal-axon-tension.py` (63,789 B, `16,678` tok, SHA-256 `825e5b302c746a26…`)
- **Artifact:** `docs/media/cortical_sulcal_axon_tension_comparison.png` (878,251 B, `64` tok, SHA-256 `9f110bc763cffc5c…`)
- **Artifact:** `docs/media/cortical_sulcal_axon_tension_growth.gif` (169,426 B, `64` tok, SHA-256 `f4bf5aaffbe90ed0…`)
- **Artifact:** `morphoeval/audits/2026-10-19-cortical-sulcal-axon-tension-audit-v2.yaml` (14,266 B, `3,769` tok, SHA-256 `31869e9bff486416…`)
- **Artifact:** `morphoeval/audits/2026-10-19-cortical-sulcal-axon-tension-audit-v2.measurements.yaml` (6,296 B, `1,875` tok, SHA-256 `31c4a6113548b90e…`)
- **Artifact:** `morphoeval/kg/2026-10-19-cortical-sulcal-axon-tension-audit-v2/REPORT.md` (11,936 B, `3,406` tok, SHA-256 `15f1d2782d33616e…`)
