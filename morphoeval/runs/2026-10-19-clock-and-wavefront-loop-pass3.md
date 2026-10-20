# MorphoEval Full-Loop Telemetry — `2026-10-19-clock-and-wavefront-loop-pass3`

- **Target simulation:** `cooke-clock-and-wavefront`
- **Primary paper / dataset:** Cooke & Zeeman (1976) J. Theor. Biol. — A clock and wavefront model for control of the number of repeated structures
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `d58a084`
- **Completed steps:** `7 / 12`
- **Total wall-clock time:** **1628.86 s** (27.15 min) — Tier A: `1346.29 s` · Tier B: `282.57 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **274,860** (input: `6,664`, output: `268,196`, subagent: `0`) — Tier A: `202,718` · Tier B: `72,142`
- **Total artifact tokens produced:** `89,981`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 1333.82 | 0.00 | 6,664 | 195,862 | 0 | **202,526** | 33,996 | `1976-cooke-clock-and-wavefront.py`, `Loop_021_clock_and_wavefront.lean`, `cooke-clock-and-wavefront.yaml`, +3 more |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 12.47 | 0.00 | 0 | 192 | 0 | **192** | 192 | `clock_and_wavefront_comparison.png`, `clock_and_wavefront_vertex_sheet.png`, `clock_and_wavefront_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 234.09 | 0.00 | 0 | 67,282 | 0 | **67,282** | 7,706 | `2026-10-19-clock-and-wavefront-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 3.03 | 0.00 | 0 | 225 | 0 | **225** | 4,016 | `2026-10-19-clock-and-wavefront-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 17.79 | 0.00 | 0 | 1,151 | 0 | **1,151** | 37,469 | `REPORT.md`, `morphoeval_2026-10-19-clock-and-wavefront-audit-v3.ttl` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 9.97 | 0.00 | 0 | 622 | 0 | **622** | 668 | `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 17.68 | 0.00 | 0 | 2,862 | 0 | **2,862** | 5,934 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `7/12` | **1628.86** | **0.00** | **6,664** | **268,196** | **0** | **274,860** | **89,981** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 remediation: 2D vertex-model PSM sheet coupled to cusp wavefront (rosette cleft rotation) + Goldbeter 2007 FGF8/RA bistable gradient deriving v from tailbud elongation; Closed both Pass 2 items: Goldbeter 2007 FGF8/RA toggle + Dubrulle 2004 fgf8 mRNA gradient derive v_front = v_tb (0.0001% err); Farhadifar vertex sheet with A/P half-somite cleft tension (cleft 28.2->10.3 deg). 5 Lean theorems added (lean OK, 0 sorry). Self-test 6 groups, 13 measurements.
- **Artifact:** `1976-cooke-clock-and-wavefront.py` (92,507 B, `25,784` tok, SHA-256 `e2ebf4e179326ac5…`)
- **Artifact:** `morphoeval/lean/loops/Loop_021_clock_and_wavefront.lean` (9,455 B, `64` tok, SHA-256 `fde7da3ff865ef37…`)
- **Artifact:** `morphoeval/math_kg_loops/cooke-clock-and-wavefront.yaml` (12,640 B, `3,587` tok, SHA-256 `584eef08c1848845…`)
- **Artifact:** `datasets/clock_and_wavefront/fgf_ra_vertex_reference.json` (6,812 B, `2,037` tok, SHA-256 `1dedf71c930b5ed0…`)
- **Artifact:** `datasets/clock_and_wavefront/README.md` (5,212 B, `1,519` tok, SHA-256 `99e6c93537763605…`)
- **Artifact:** `datasets/clock_and_wavefront/paper_citations.bib` (3,278 B, `1,005` tok, SHA-256 `13145a25f0a40204…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 3 render: --self-test then default run (PNG + vertex-sheet PNG + GIF); self-test PASSED; default render ~8 s; 9-panel PNG + vertex-sheet PNG + 24-frame GIF (FGF/RA coupled model)
- **Artifact:** `docs/media/clock_and_wavefront_comparison.png` (1,415,556 B, `64` tok, SHA-256 `0a6c41da9df6f972…`)
- **Artifact:** `docs/media/clock_and_wavefront_vertex_sheet.png` (301,864 B, `64` tok, SHA-256 `6b712a94cc9ef0dc…`)
- **Artifact:** `docs/media/clock_and_wavefront_growth.gif` (1,259,028 B, `64` tok, SHA-256 `2621e3112126f1e4…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 3 re-score on 7 rubric dimensions with line-pinned evidence; v3 audit supersedes v2; v3 audit: MF 91, SM 87, PE 92, MV 91, ST 93, CT 91, RP 91 (overall ~90.3, flat vs 90.4); 7 Pass 4 remediation items; --check valid
- **Artifact:** `morphoeval/audits/2026-10-19-clock-and-wavefront-audit-v3.yaml` (29,583 B, `7,706` tok, SHA-256 `5ff10c8478e82239…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs on v3 audit; measurements.yaml written: 3 outputs fingerprinted, 13 measurements
- **Artifact:** `morphoeval/audits/2026-10-19-clock-and-wavefront-audit-v3.measurements.yaml` (13,902 B, `4,016` tok, SHA-256 `d27c79e1523cd833…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build v3 KG superseding v2 (--no-resolve-heads --conform); KG built with supersession chain from v2 (fixed multi-range code_locus URI in math_kg yaml)
- **Artifact:** `morphoeval/kg/2026-10-19-clock-and-wavefront-audit-v3/REPORT.md` (21,100 B, `5,934` tok, SHA-256 `9456a7ad51019ba1…`)
- **Artifact:** `morphoeval/kg/2026-10-19-clock-and-wavefront-audit-v3/morphoeval_2026-10-19-clock-and-wavefront-audit-v3.ttl` (106,180 B, `31,535` tok, SHA-256 `b39e802a573a229b…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** sign only (no publish); 10 nanopubs signed, 10/10 verified by check_signed.mjs; NOT published
- **Artifact:** `morphoeval/kg/2026-10-19-clock-and-wavefront-audit-v3/published.json` (1,880 B, `668` tok, SHA-256 `4261a8b5d8836cc1…`)

### Step 11 — `graph` (Public record)
- **Notes:** read REPORT.md; confirm overall score and superseded audit id; REPORT.md: overall 90.3 (faithful), chain audit -> v2 -> v3 confirmed; 817 triples, ProbKnow conformance OK
- **Artifact:** `morphoeval/kg/2026-10-19-clock-and-wavefront-audit-v3/REPORT.md` (21,100 B, `5,934` tok, SHA-256 `9456a7ad51019ba1…`)
