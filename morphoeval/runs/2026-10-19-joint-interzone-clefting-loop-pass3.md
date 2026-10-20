# MorphoEval Full-Loop Telemetry — `2026-10-19-joint-interzone-clefting-loop-pass3`

- **Target simulation:** `onuff-joint-interzone-clefting`
- **Primary paper / dataset:** Hardingham & Gdf5 Interzone Segmentation Model (2016) Dev. Biol.
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `d58a084`
- **Completed steps:** `8 / 12`
- **Total wall-clock time:** **1576.56 s** (26.28 min) — Tier A: `1157.69 s` · Tier B: `418.87 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **258,131** (input: `6,712`, output: `251,419`, subagent: `0`) — Tier A: `203,874` · Tier B: `54,257`
- **Total artifact tokens produced:** `98,336`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 1017.96 | 0.00 | 6,148 | 177,988 | 0 | **184,136** | 36,368 | `2016-onuff-joint-interzone-clefting.py`, `Loop_018_joint_interzone_clefting.lean`, `synovial_cavitation_reference.json`, +3 more |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 139.74 | 0.00 | 564 | 19,174 | 0 | **19,738** | 192 | `joint_interzone_clefting_comparison.png`, `joint_interzone_clefting_cavitation.png`, `joint_interzone_clefting_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 275.83 | 0.00 | 0 | 37,151 | 0 | **37,151** | 10,614 | `2026-10-19-joint-interzone-clefting-audit-v3.yaml`, `onuff-joint-interzone-clefting.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 64.24 | 0.00 | 0 | 8,985 | 0 | **8,985** | 4,077 | `2026-10-19-joint-interzone-clefting-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 12.36 | 0.00 | 0 | 2,833 | 0 | **2,833** | 38,679 | `REPORT.md`, `morphoeval_2026-10-19-joint-interzone-clefting-audit-v3.ttl` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 9.83 | 0.00 | 0 | 326 | 0 | **326** | 658 | `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 17.75 | 0.00 | 0 | 2,406 | 0 | **2,406** | 5,941 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 38.86 | 0.00 | 0 | 2,556 | 0 | **2,556** | 1,807 | `slot-018.json` |
| — | **All** | **TOTAL** | — | `8/12` | **1576.56** | **0.00** | **6,712** | **251,419** | **0** | **258,131** | **98,336** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 remediation: poroelastic synovial cavitation solver coupled to 2D GDF5-Noggin RD PDE; Pass 3 remediation: simulate_poroelastic_cavitation + sweeps + specimen articular arc; 2 new Lean theorems; 12-check self-test
- **Artifact:** `2016-onuff-joint-interzone-clefting.py` (94,533 B, `27,161` tok, SHA-256 `c208c3904c6e5ad8…`)
- **Artifact:** `morphoeval/lean/loops/Loop_018_joint_interzone_clefting.lean` (10,294 B, `64` tok, SHA-256 `84835402631c4c75…`)
- **Artifact:** `datasets/joint_interzone_clefting/synovial_cavitation_reference.json` (10,595 B, `2,724` tok, SHA-256 `9ca273ead3e3606e…`)
- **Artifact:** `datasets/joint_interzone_clefting/README.md` (3,591 B, `890` tok, SHA-256 `437c005e750b55c3…`)
- **Artifact:** `datasets/joint_interzone_clefting/paper_citations.bib` (5,173 B, `1,569` tok, SHA-256 `4dba5970a1ffae76…`)
- **Artifact:** `datasets/joint_interzone_clefting/joint_interzone_clefting_reference.json` (12,213 B, `3,960` tok, SHA-256 `8ce3cc464602a827…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test 12/12 PASS; default render 7.8 s; specimen comparison visible in Panels D/F; cavitation figure Panels A-F
- **Artifact:** `docs/media/joint_interzone_clefting_comparison.png` (1,021,079 B, `64` tok, SHA-256 `470eccc646ae9036…`)
- **Artifact:** `docs/media/joint_interzone_clefting_cavitation.png` (658,959 B, `64` tok, SHA-256 `c612dc9311254321…`)
- **Artifact:** `docs/media/joint_interzone_clefting_growth.gif` (1,503,309 B, `64` tok, SHA-256 `1aad6f5f5b4845ae…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** v3 audit 89.4 (down from 96.6; deeper defects exposed), 8 Pass 4 remediation items, --check valid
- **Artifact:** `morphoeval/audits/2026-10-19-joint-interzone-clefting-audit-v3.yaml` (30,783 B, `7,709` tok, SHA-256 `a5475b3970c831a6…`)
- **Artifact:** `morphoeval/math_kg_loops/onuff-joint-interzone-clefting.yaml` (10,642 B, `2,905` tok, SHA-256 `3605be5474173b9e…`)

### Step 8 — `measure` (Measure)
- **Notes:** 16 measurements, 3 outputs fingerprinted, 5 dataset inputs hashed, quick_gate 100
- **Artifact:** `morphoeval/audits/2026-10-19-joint-interzone-clefting-audit-v3.measurements.yaml` (14,142 B, `4,077` tok, SHA-256 `faf54fd0a97054a2…`)

### Step 9 — `build` (Write as claims)
- **Notes:** 10 nanopubs, 887 triples, ProbKnow conformance OK, supersede-from v2
- **Artifact:** `morphoeval/kg/2026-10-19-joint-interzone-clefting-audit-v3/REPORT.md` (21,559 B, `5,941` tok, SHA-256 `68c84dd66c7d8231…`)
- **Artifact:** `morphoeval/kg/2026-10-19-joint-interzone-clefting-audit-v3/morphoeval_2026-10-19-joint-interzone-clefting-audit-v3.ttl` (113,355 B, `32,738` tok, SHA-256 `1c508013b42fba2e…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** 10 nanopubs signed, 10/10 verified with check_signed.mjs; NOT published
- **Artifact:** `morphoeval/kg/2026-10-19-joint-interzone-clefting-audit-v3/published.json` (1,925 B, `658` tok, SHA-256 `8456eb4f5b128785…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md: overall 89.4 (faithful), supersession chain v1 -> v2 -> v3 confirmed
- **Artifact:** `morphoeval/kg/2026-10-19-joint-interzone-clefting-audit-v3/REPORT.md` (21,559 B, `5,941` tok, SHA-256 `68c84dd66c7d8231…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 3 dispatch record + file-isolation check; Pass 3 dispatch fields appended; 8 Pass 4 items opened
- **Artifact:** `morphoeval/cloud/dispatches/slot-018.json` (6,297 B, `1,807` tok, SHA-256 `d3d7ca3fa9a7bef5…`)
