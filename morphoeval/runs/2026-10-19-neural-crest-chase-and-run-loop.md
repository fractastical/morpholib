# MorphoEval Full-Loop Telemetry — `2026-10-19-neural-crest-chase-and-run-loop`

- **Target simulation:** `carmona-neural-crest-chase-and-run`
- **Primary paper / dataset:** Theveneau et al. (2013) Nature Cell Biol. / Carmona-Fontaine et al. (2008) Nature — Contact inhibition of locomotion & Chase-and-run in neural crest migration
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `3cb5ea9`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **973.14 s** (16.22 min) — Tier A: `202.79 s` · Tier B: `770.35 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **479,484** (input: `14,410`, output: `465,074`, subagent: `0`) — Tier A: `103,852` · Tier B: `375,632`
- **Total artifact tokens produced:** `42,899`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 1.47 | 0.00 | 0 | 1,554 | 0 | **1,554** | 1,554 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 3,855 | 0 | **3,855** | 3,855 | `neural_crest_chase_and_run_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 2,302 | 0 | **2,302** | 2,302 | `neural_crest_specimen.jpg`, `neural_crest_chase_and_run_reference.json` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 90.02 | 0.00 | 5,626 | 23,588 | 0 | **29,214** | 1,378 | `Loop_091_neural_crest_chase_and_run.lean`, `carmona-neural-crest-chase-and-run.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 110.20 | 0.00 | 0 | 66,763 | 0 | **66,763** | 3,331 | `2012-carmona-neural-crest-chase-and-run.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.93 | 0.00 | 0 | 164 | 0 | **164** | 128 | `neural_crest_chase_and_run_comparison.png`, `neural_crest_chase_and_run_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 51.79 | 0.00 | 0 | 26,991 | 0 | **26,991** | 2,917 | `2026-10-19-neural-crest-chase-and-run-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 693.00 | 0.00 | 8,596 | 302,162 | 0 | **310,758** | 1,506 | `2026-10-19-neural-crest-chase-and-run-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.10 | 0.00 | 0 | 3,118 | 0 | **3,118** | 3,118 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.08 | 0.00 | 0 | 719 | 0 | **719** | 719 | `published.json` |
| 11 | B | **graph** (Public record) | tool | `completed` | 25.29 | 0.00 | 188 | 29,977 | 0 | **30,165** | 18,210 | `morphoeval_2026-10-19-neural-crest-chase-and-run-audit-v2.ttl` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.09 | 0.00 | 0 | 3,881 | 0 | **3,881** | 3,881 | `2026-10-19-neural-crest-chase-and-run-audit-v2.yaml` |
| — | **All** | **TOTAL** | — | `12/12` | **973.14** | **0.00** | **14,410** | **465,074** | **0** | **479,484** | **42,899** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Completed Step 1 scout for Slot 91 (Theveneau et al. 2013 / Carmona-Fontaine et al. 2008 neural crest CIL + C3a co-attraction + SDF1 chase-and-run).
- **Artifact:** `datasets/neural_crest_chase_and_run/scout_manifest.json` (5,540 B, `1,554` tok, SHA-256 `f6e076ebe4fba6c0…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2010, 2013) CIL + C3a/C3aR COA + Sdf1/placode chase-and-run equations and 5 experimental regimes.
- **Artifact:** `datasets/neural_crest_chase_and_run/neural_crest_chase_and_run_reference.json` (7,783 B, `2,238` tok, SHA-256 `e1da41bb0c8f1023…`)
- **Artifact:** `datasets/neural_crest_chase_and_run/paper_citations.bib` (2,207 B, `632` tok, SHA-256 `c9efb624adff2c20…`)
- **Artifact:** `datasets/neural_crest_chase_and_run/README.md` (3,192 B, `985` tok, SHA-256 `5fbfe33bc729ea3f…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC BY 4.0 cranial neural crest r4/r6/r7 streams + epibranchial placode e1/e2/e3 specimen (neural_crest_specimen.jpg) and locked SHA-256 digests.
- **Artifact:** `datasets/neural_crest_chase_and_run/neural_crest_specimen.jpg` (816,515 B, `64` tok, SHA-256 `291091fcf1198a71…`)
- **Artifact:** `datasets/neural_crest_chase_and_run/neural_crest_chase_and_run_reference.json` (7,783 B, `2,238` tok, SHA-256 `e1da41bb0c8f1023…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Proved 4 Lean 4 theorems (0 sorrys) in Loop_091_neural_crest_chase_and_run.lean and registered carmona-neural-crest-chase-and-run.yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_091_neural_crest_chase_and_run.lean` (7,763 B, `64` tok, SHA-256 `b8a0b948baf3c29d…`)
- **Artifact:** `morphoeval/math_kg_loops/carmona-neural-crest-chase-and-run.yaml` (5,123 B, `1,314` tok, SHA-256 `ea258535cb7e395b…`)

### Step 5 — `script` (Script)
- **Notes:** Implemented Pass 1 baseline 2012-carmona-neural-crest-chase-and-run.py.
- **Artifact:** `2012-carmona-neural-crest-chase-and-run.py` (11,174 B, `3,331` tok, SHA-256 `3098f7754184cdc6…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Rendered Pass 1 neural_crest_chase_and_run_comparison.png and neural_crest_chase_and_run_growth.gif.
- **Artifact:** `docs/media/neural_crest_chase_and_run_comparison.png` (78,344 B, `64` tok, SHA-256 `55d7a1f67cc098a3…`)
- **Artifact:** `docs/media/neural_crest_chase_and_run_growth.gif` (94,610 B, `64` tok, SHA-256 `10df58576ab50dd0…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Completed Pass 1 MorphoEval 7-dimension audit (57.6/100) with explicit remediation targets.
- **Artifact:** `morphoeval/audits/2026-10-19-neural-crest-chase-and-run-audit.yaml` (11,214 B, `2,917` tok, SHA-256 `ce89c72472be1749…`)

### Step 8 — `measure` (Measure)
- **Notes:** Measured Pass 1 (quick_gate=90) and Pass 2 (quick_gate=100, 8 reference-anchored metrics)
- **Artifact:** `morphoeval/audits/2026-10-19-neural-crest-chase-and-run-audit-v2.measurements.yaml` (5,204 B, `1,506` tok, SHA-256 `82eebc945b75e924…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Build Pass 1 and Pass 2 MorphoEval RDF/TriG knowledge graphs; Compiled 10 Pass 1 nanopubs (57.6/100) and 10 Pass 2 nanopubs (91.1/100) with prov:wasRevisionOf provenance
- **Artifact:** `morphoeval/kg/2026-10-19-neural-crest-chase-and-run-audit-v2/REPORT.md` (10,530 B, `3,118` tok, SHA-256 `728cfb40ccf52dc8…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** RSA-sign Pass 1 and Pass 2 nanopublications with Trusty URIs; Signed all 10 Pass 1 and 10 Pass 2 nanopubs with RSA-2048 Trusty URIs
- **Artifact:** `morphoeval/kg/2026-10-19-neural-crest-chase-and-run-audit-v2/published.json` (1,961 B, `719` tok, SHA-256 `481f4158b5e7a069…`)

### Step 11 — `graph` (Public record)
- **Notes:** Query remediation targets and score trajectory across Pass 1 -> Pass 2; Queried SPARQL remediation delta: 57.6 -> 91.1 (+33.5)
- **Artifact:** `morphoeval/kg/2026-10-19-neural-crest-chase-and-run-audit-v2/morphoeval_2026-10-19-neural-crest-chase-and-run-audit-v2.ttl` (60,440 B, `18,210` tok, SHA-256 `24d7c5ee7cfea34a…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Close Pass 1 -> Pass 2 remediation loop and emit telemetry summary; Closed remediation loop: Pass 1 57.6/100 -> Pass 2 91.1/100 (+33.5), quick_gate 90 -> 100
- **Artifact:** `morphoeval/audits/2026-10-19-neural-crest-chase-and-run-audit-v2.yaml` (14,218 B, `3,881` tok, SHA-256 `55186ef17655317e…`)
