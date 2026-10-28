# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-010-the-rate-of-growth-loop`

- **Target simulation:** `thompson1917-t1917-010`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. III, pp. 50-155, Figs. 33-40
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `98a43c3a1`
- **Completed steps:** `11 / 12`
- **Total wall-clock time:** **1367.01 s** (22.78 min) — Tier A: `712.79 s` · Tier B: `654.23 s`
- **Total CPU time:** `1.94 s` user + `0.16 s` sys
- **Total LLM tokens:** **399,800** (input: `322,000`, output: `77,800`, subagent: `0`) — Tier A: `241,500` · Tier B: `158,300`
- **Total artifact tokens produced:** `56,710`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | cloud-agent (`claude-fable-5.1`) | `completed` | 316.78 | 0.00 | 60,000 | 6,000 | 0 | **66,000** | 1,913 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.09 | 0.00 | 25,000 | 7,000 | 0 | **32,000** | 5,910 | `t1917_010_the_rate_of_growth_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.09 | 0.00 | 20,000 | 5,000 | 0 | **25,000** | 1,747 | `lithobates_clamitans_tadpole_specimen.jpg`, `thompson_1917_p148_leaf168.jpg`, `dryad_n6n6495_fitzpatrick_et_al_data.csv` |
| 4 | A | **rules** (Pin the rules) | cloud-agent (`claude-fable-5.1`) | `completed` | 177.26 | 0.00 | 30,000 | 9,000 | 0 | **39,000** | 1,799 | `Thompson1917_T1917_010.lean`, `thompson1917-t1917-010.yaml` |
| 5 | A | **script** (Script) | cloud-agent (`claude-fable-5.1`) | `completed` | 215.22 | 0.00 | 48,000 | 16,000 | 0 | **64,000** | 14,452 | `t1917-010_the-rate-of-growth.py` |
| 6 | A | **output** (Output & compare) | cloud-agent (`claude-fable-5.1`) | `completed` | 3.35 | 0.00 | 13,000 | 2,500 | 0 | **15,500** | 2,148 | `t1917_010_the_rate_of_growth_comparison.png`, `t1917_010_the_rate_of_growth_growth.gif`, `result.json` |
| 7 | B | **audit** (Dual-agent audit) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.20 | 0.00 | 28,000 | 7,000 | 0 | **35,000** | 6,578 | `2026-10-28-t1917-010-the-rate-of-growth-audit.yaml`, `2026-10-28-t1917-010-the-rate-of-growth-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | cloud-agent (`claude-fable-5.1`) | `completed` | 0.79 | 1.27 | 11,000 | 2,300 | 0 | **13,300** | 3,817 | `2026-10-28-t1917-010-the-rate-of-growth-audit.measurements.yaml`, `2026-10-28-t1917-010-the-rate-of-growth-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | cloud-agent (`claude-fable-5.1`) | `completed` | 1.76 | 0.83 | 11,000 | 2,300 | 0 | **13,300** | 5,883 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | cloud-agent (`claude-fable-5.1`) | `skipped` | 0.20 | 0.00 | 12,000 | 2,300 | 0 | **14,300** | 0 | — |
| 11 | B | **graph** (Public record) | cloud-agent (`claude-fable-5.1`) | `completed` | 46.13 | 0.00 | 14,000 | 2,400 | 0 | **16,400** | 5,883 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | cloud-agent (`claude-fable-5.1`) | `completed` | 605.15 | 0.00 | 50,000 | 16,000 | 0 | **66,000** | 6,580 | `2026-10-28-t1917-010-the-rate-of-growth-audit-v2.yaml`, `REPORT.md` |
| — | **All** | **TOTAL** | — | `11/12` | **1367.01** | **2.10** | **322,000** | **77,800** | **0** | **399,800** | **56,710** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Verify Dryad n6n6495 vs claim; IA scan ongrowthform1917thom pp.138-149; iNat CC-BY Lithobates clamitans tadpole; Dryad n6n6495 fetched: DATASET_MISMATCH (skink telomeres, single endpoint). Verdict dataset = Thompson's own pp.139-148 tables from IA scan. Commons 503; iNat CC-BY L. clamitans tadpole vendored.
- **Artifact:** `datasets/thompson_1917/t1917-010/scout_manifest.json` (6,534 B, `1,913` tok, SHA-256 `a0e17d8bc22e3fe3…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations E1-E7, state vars, BCs, regimes, claims C1-C6 extracted from pp.138-149; Durbin 1909 original consulted for table identity
- **Artifact:** `datasets/thompson_1917/t1917-010/t1917_010_the_rate_of_growth_reference.json` (11,980 B, `3,765` tok, SHA-256 `8cf23ca99668fed8…`)
- **Artifact:** `datasets/thompson_1917/t1917-010/paper_citations.bib` (2,682 B, `808` tok, SHA-256 `65dce1714be1e93d…`)
- **Artifact:** `datasets/thompson_1917/t1917-010/README.md` (4,487 B, `1,337` tok, SHA-256 `5502a23e868ef1d4…`)

### Step 3 — `source` (Formal source)
- **Notes:** 4 digitised tables, 5 page scans, CC-BY specimen, Dryad CSV, OCR excerpt; 12 SHA-256 digests pinned in reference.json
- **Artifact:** `datasets/thompson_1917/t1917-010/lithobates_clamitans_tadpole_specimen.jpg` (64,739 B, `64` tok, SHA-256 `4a90d205c4f3ec5e…`)
- **Artifact:** `datasets/thompson_1917/t1917-010/thompson_1917_p148_leaf168.jpg` (255,626 B, `64` tok, SHA-256 `dec8fbe709f3068d…`)
- **Artifact:** `datasets/thompson_1917/t1917-010/dryad_n6n6495_fitzpatrick_et_al_data.csv` (2,476 B, `1,619` tok, SHA-256 `b6200e3593aefa85…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4.35.0 kernel-checked via elan: 17 theorems, 0 sorry (E1 telescoping, E2 hyperbolic antitone, E3 geometric decay, E5 Ellis strict mono, E6 cubic isochrone, E7 incomplete plateau); 6 KG equations
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_010.lean` (9,065 B, `64` tok, SHA-256 `36af9a4ae4b3f5de…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-010.yaml` (5,889 B, `1,735` tok, SHA-256 `f6fa15e110ae5bce…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2; Pass 2 script v2: sweeps, controls, CV, KS, reference values
- **Artifact:** `thompson/claims/t1917-010_the-rate-of-growth.py` (47,991 B, `14,452` tok, SHA-256 `4b73b6ef37fff30c…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2; Pass 2 self-test passed (25 guards); verdict supported, 3 caveats
- **Artifact:** `docs/media/t1917_010_the_rate_of_growth_comparison.png` (734,418 B, `64` tok, SHA-256 `cd24ca6582e91fdb…`)
- **Artifact:** `docs/media/t1917_010_the_rate_of_growth_growth.gif` (229,906 B, `64` tok, SHA-256 `7b19ed35899b5939…`)
- **Artifact:** `datasets/thompson_1917/t1917-010/result.json` (5,979 B, `2,020` tok, SHA-256 `c1c2ff9282f10003…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 superseding audit v2 (supersedes_audit_id set), 24 evidence items
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-010-the-rate-of-growth-audit.yaml` (12,144 B, `3,112` tok, SHA-256 `8482c6e87a99b18b…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-010-the-rate-of-growth-audit-v2.yaml` (13,035 B, `3,466` tok, SHA-256 `fe6e2175a23aaece…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2; Pass 2 measurements: 12 metrics, 8 with reference_value, quick_gate 100 validated
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-010-the-rate-of-growth-audit.measurements.yaml` (4,986 B, `1,564` tok, SHA-256 `f9bdb65de27ed8c3…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-010-the-rate-of-growth-audit-v2.measurements.yaml` (7,478 B, `2,253` tok, SHA-256 `a792335c12ce4889…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2; Pass 2 KG: 10 nanopubs, 644 triples, overall 78.9 mostly faithful; --supersede-from deferred (needs signed Pass 1 published.json)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-010-the-rate-of-growth-audit/REPORT.md` (9,981 B, `2,769` tok, SHA-256 `623af952312809cb…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-010-the-rate-of-growth-audit-v2/REPORT.md` (10,785 B, `3,114` tok, SHA-256 `87de214996647755…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 2; Pass 2 signing deferred: maintainer key required (see Pass 1 note); after signing Pass 1 rebuild v2 with --supersede-from

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2; Pass 2 REPORT.md verified: supersession chain audit -> audit-v2, overall 78.9 (from 55.0)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-010-the-rate-of-growth-audit/REPORT.md` (9,981 B, `2,769` tok, SHA-256 `623af952312809cb…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-010-the-rate-of-growth-audit-v2/REPORT.md` (10,785 B, `3,114` tok, SHA-256 `87de214996647755…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2: LF fix, sweep, controls, CV, KS; All 7 Pass 1 remediation items closed; overall 55.0 -> 78.9; verdict supported with caveats on Thompson's V t^2 gloss and 75% asymptote extrapolation
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-010-the-rate-of-growth-audit-v2.yaml` (13,035 B, `3,466` tok, SHA-256 `fe6e2175a23aaece…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-010-the-rate-of-growth-audit-v2/REPORT.md` (10,785 B, `3,114` tok, SHA-256 `87de214996647755…`)
