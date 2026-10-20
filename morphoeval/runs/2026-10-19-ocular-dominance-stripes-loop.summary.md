# MorphoEval Full-Loop Telemetry — `2026-10-19-ocular-dominance-stripes-loop`

- **Target simulation:** `1989-miller-ocular-dominance-stripes.py`
- **Primary paper / dataset:** Miller, Keller & Stryker (1989) Science 245:605-615; Hubel, Wiesel & LeVay (1977) Phil. Trans. R. Soc. B 278:377-409
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `2a34864`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **530.97 s** (8.85 min) — Tier A: `203.83 s` · Tier B: `327.14 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **227,560** (input: `800`, output: `226,760`, subagent: `0`) — Tier A: `99,740` · Tier B: `127,820`
- **Total artifact tokens produced:** `31,097`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 2.10 | 0.00 | 0 | 745 | 0 | **745** | 745 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 99.24 | 0.00 | 376 | 49,777 | 0 | **50,153** | 2,497 | `paper_citations.bib`, `ocular_dominance_stripes_reference.json` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 530 | 0 | **530** | 530 | `odc_tangential_pattern_specimen.png`, `macaque_striate_cortex_specimen.jpg`, `README.md` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 41.84 | 0.00 | 0 | 21,453 | 0 | **21,453** | 2,281 | `Loop_087_ocular_dominance_stripes.lean`, `miller-ocular-dominance-stripes.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 59.82 | 0.00 | 0 | 26,731 | 0 | **26,731** | 1,993 | `1989-miller-ocular-dominance-stripes.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.75 | 0.00 | 0 | 128 | 0 | **128** | 128 | `ocular_dominance_stripes_comparison.png`, `ocular_dominance_stripes_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 2,131 | 0 | **2,131** | 2,131 | `2026-10-19-ocular-dominance-stripes-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.57 | 0.00 | 0 | 864 | 0 | **864** | 864 | `2026-10-19-ocular-dominance-stripes-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.46 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 31.44 | 0.00 | 0 | 23,469 | 0 | **23,469** | 652 | `published.json` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 1,999 | 0 | **1,999** | 1,999 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 294.49 | 0.00 | 424 | 98,933 | 0 | **99,357** | 17,277 | `1989-miller-ocular-dominance-stripes.py`, `ocular_dominance_stripes_comparison.png`, `ocular_dominance_stripes_growth.gif`, +3 more |
| — | **All** | **TOTAL** | — | `12/12` | **530.97** | **0.00** | **800** | **226,760** | **0** | **227,560** | **31,097** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/ocular_dominance_stripes/scout_manifest.json` (2,709 B, `745` tok, SHA-256 `2f54657c6b0c2050…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/ocular_dominance_stripes/paper_citations.bib` (1,458 B, `439` tok, SHA-256 `29c3ea74bae42954…`)
- **Artifact:** `datasets/ocular_dominance_stripes/ocular_dominance_stripes_reference.json` (6,910 B, `2,058` tok, SHA-256 `7041693bc0b54b31…`)

### Step 3 — `source` (Formal source)
- **Artifact:** `datasets/ocular_dominance_stripes/odc_tangential_pattern_specimen.png` (59,613 B, `64` tok, SHA-256 `fac59bb151b4eb1d…`)
- **Artifact:** `datasets/ocular_dominance_stripes/macaque_striate_cortex_specimen.jpg` (3,057,365 B, `64` tok, SHA-256 `fe74eb6f9db7dea1…`)
- **Artifact:** `datasets/ocular_dominance_stripes/README.md` (1,438 B, `402` tok, SHA-256 `c3682b916e66a5c0…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_087_ocular_dominance_stripes.lean` (8,329 B, `64` tok, SHA-256 `38bd56d2945b15f9…`)
- **Artifact:** `morphoeval/math_kg_loops/miller-ocular-dominance-stripes.yaml` (8,827 B, `2,217` tok, SHA-256 `8bab5189b805fefd…`)

### Step 5 — `script` (Script)
- **Artifact:** `1989-miller-ocular-dominance-stripes.py` (6,994 B, `1,993` tok, SHA-256 `cd991f7ef237b7df…`)

### Step 6 — `output` (Output & compare)
- **Artifact:** `docs/media/ocular_dominance_stripes_comparison.png` (1,083,402 B, `64` tok, SHA-256 `5c86c681629586dc…`)
- **Artifact:** `docs/media/ocular_dominance_stripes_growth.gif` (298,610 B, `64` tok, SHA-256 `66e24bdb5669a44c…`)

### Step 7 — `audit` (Dual-agent audit)
- **Artifact:** `morphoeval/audits/2026-10-19-ocular-dominance-stripes-audit.yaml` (8,864 B, `2,131` tok, SHA-256 `5830ce5ec19fb760…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-19-ocular-dominance-stripes-audit.measurements.yaml` (2,706 B, `864` tok, SHA-256 `558603c0df08785a…`)

### Step 10 — `publish` (Sign & verify)
- **Artifact:** `morphoeval/kg/2026-10-19-ocular-dominance-stripes-audit/published.json` (1,934 B, `652` tok, SHA-256 `48532a184bcd49b7…`)

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-19-ocular-dominance-stripes-audit/REPORT.md` (7,925 B, `1,999` tok, SHA-256 `98fa2d555f0440da…`)

### Step 12 — `remed` (Remediate & re-score)
- **Artifact:** `1989-miller-ocular-dominance-stripes.py` (39,294 B, `10,894` tok, SHA-256 `2163b3a1d236196c…`)
- **Artifact:** `docs/media/ocular_dominance_stripes_comparison.png` (1,765,310 B, `64` tok, SHA-256 `d9836de9bccac067…`)
- **Artifact:** `docs/media/ocular_dominance_stripes_growth.gif` (1,847,372 B, `64` tok, SHA-256 `1ef27c3ba5ecadd0…`)
- **Artifact:** `morphoeval/audits/2026-10-19-ocular-dominance-stripes-audit-v2.yaml` (13,571 B, `3,426` tok, SHA-256 `a71430fd25d3b01a…`)
- **Artifact:** `morphoeval/audits/2026-10-19-ocular-dominance-stripes-audit-v2.measurements.yaml` (7,797 B, `2,170` tok, SHA-256 `205c6fd118ed6c2f…`)
- **Artifact:** `morphoeval/kg/2026-10-19-ocular-dominance-stripes-audit-v2/published.json` (1,934 B, `659` tok, SHA-256 `1ae02b6b23ae5cbe…`)
