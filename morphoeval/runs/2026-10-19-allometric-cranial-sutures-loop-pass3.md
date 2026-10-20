# MorphoEval Full-Loop Telemetry — `2026-10-19-allometric-cranial-sutures-loop-pass3`

- **Target simulation:** `bailleul-allometric-cranial-sutures`
- **Primary paper / dataset:** Miura et al. (2009) PNAS — Mechanism of cranial sutureimos fractal interdigitation
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `d58a084`
- **Completed steps:** `7 / 12`
- **Total wall-clock time:** **1457.67 s** (24.29 min) — Tier A: `1401.95 s` · Tier B: `55.73 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **198,345** (input: `6,524`, output: `191,821`, subagent: `0`) — Tier A: `175,293` · Tier B: `23,052`
- **Total artifact tokens produced:** `85,346`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 1270.48 | 0.00 | 6,524 | 155,442 | 0 | **161,966** | 34,373 | `2020-bailleul-allometric-cranial-sutures.py`, `Loop_020_allometric_cranial_sutures.lean`, `bailleul-allometric-cranial-sutures.yaml`, +3 more |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 131.46 | 0.00 | 0 | 13,327 | 0 | **13,327** | 128 | `allometric_cranial_sutures_comparison.png`, `allometric_cranial_sutures_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 7.41 | 0.00 | 0 | 1,107 | 0 | **1,107** | 6,899 | `2026-10-19-allometric-cranial-sutures-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 20.55 | 0.00 | 0 | 11,297 | 0 | **11,297** | 3,601 | `2026-10-19-allometric-cranial-sutures-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 18.43 | 0.00 | 0 | 2,540 | 0 | **2,540** | 34,333 | `REPORT.md`, `morphoeval_2026-10-19-allometric-cranial-sutures-audit-v3.ttl` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 9.24 | 0.00 | 0 | 2,762 | 0 | **2,762** | 666 | `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 0.09 | 0.00 | 0 | 5,346 | 0 | **5,346** | 5,346 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `7/12` | **1457.67** | **0.00** | **6,524** | **191,821** | **0** | **198,345** | **85,346** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 remediation: quasi-3D ecto->endocranial layered KS solver + heterogeneous masticatory strain coupling across coronal/sagittal/lambdoid sutures; Pass 3 remediation: heterogeneous alpha(x) pseudo-spectral KS/KPZ, 5-layer ecto->endo plate stack with kappa_z coupling, reduced-order Kirchhoff-Love vault strain field for coronal/sagittal/lambdoid, simulated 15x15 morphospace, nuchal sweep, 2 new Lean theorems, 14-check self-test
- **Artifact:** `2020-bailleul-allometric-cranial-sutures.py` (87,426 B, `24,495` tok, SHA-256 `bda1e460fd932cd6…`)
- **Artifact:** `morphoeval/lean/loops/Loop_020_allometric_cranial_sutures.lean` (11,795 B, `64` tok, SHA-256 `fad9871415e981de…`)
- **Artifact:** `morphoeval/math_kg_loops/bailleul-allometric-cranial-sutures.yaml` (13,562 B, `3,519` tok, SHA-256 `f27a35986f61131d…`)
- **Artifact:** `datasets/allometric_cranial_sutures/allometric_cranial_sutures_reference.json` (12,066 B, `3,540` tok, SHA-256 `c41074e16d3025c9…`)
- **Artifact:** `datasets/allometric_cranial_sutures/paper_citations.bib` (3,217 B, `972` tok, SHA-256 `81d1b1b367ddb12c…`)
- **Artifact:** `datasets/allometric_cranial_sutures/README.md` (6,344 B, `1,783` tok, SHA-256 `bd6e1cd0807c01ae…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test 14/14 passed; 12-panel PNG (+row D: vault strain, alpha(s,zeta), quasi-3D layered cleft) and 24-frame GIF regenerated; specimen comparison visible in B1/B2/C1/C2/C3
- **Artifact:** `docs/media/allometric_cranial_sutures_comparison.png` (1,553,683 B, `64` tok, SHA-256 `7d403d848e058ca7…`)
- **Artifact:** `docs/media/allometric_cranial_sutures_growth.gif` (573,223 B, `64` tok, SHA-256 `27e6d90f28198a51…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** v3 audit: 86.5 overall (MF 84, SM 82, PE 90, MV 86, ST 92, CT 88, RP 92), supersedes v2 (92.4); 5 Pass 4 remediation items; --check valid
- **Artifact:** `morphoeval/audits/2026-10-19-allometric-cranial-sutures-audit-v3.yaml` (27,516 B, `6,899` tok, SHA-256 `a346a81c61a268f7…`)

### Step 8 — `measure` (Measure)
- **Notes:** 11 measurements, 2 outputs fingerprinted, quick_gate 100/100
- **Artifact:** `morphoeval/audits/2026-10-19-allometric-cranial-sutures-audit-v3.measurements.yaml` (13,071 B, `3,601` tok, SHA-256 `fc0f82490053f95e…`)

### Step 9 — `build` (Write as claims)
- **Notes:** KG built: 10 nanopubs, 713 triples, ProbKnow conformance OK, supersedes v2
- **Artifact:** `morphoeval/kg/2026-10-19-allometric-cranial-sutures-audit-v3/REPORT.md` (19,823 B, `5,346` tok, SHA-256 `90a6efa4c5aca1e8…`)
- **Artifact:** `morphoeval/kg/2026-10-19-allometric-cranial-sutures-audit-v3/morphoeval_2026-10-19-allometric-cranial-sutures-audit-v3.ttl` (103,391 B, `28,987` tok, SHA-256 `c2d3a2aa14c3c05c…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** 10 nanopubs signed, 10/10 verified by check_signed.mjs; NOT published
- **Artifact:** `morphoeval/kg/2026-10-19-allometric-cranial-sutures-audit-v3/published.json` (1,970 B, `666` tok, SHA-256 `867f546928b512d6…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md confirms overall 86.5 and supersedes v2
- **Artifact:** `morphoeval/kg/2026-10-19-allometric-cranial-sutures-audit-v3/REPORT.md` (19,823 B, `5,346` tok, SHA-256 `90a6efa4c5aca1e8…`)
