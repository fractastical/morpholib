# MorphoEval Full-Loop Telemetry — `2026-10-19-coral-calice-septa-insertion-loop`

- **Target simulation:** `thomas-coral-calice-septa-insertion`
- **Primary paper / dataset:** Milne-Edwards & Haime (1848) / Kaandorp & Kübler (2001) / Chindapol et al. (2013) PLoS Comput. Biol. — Hexacorallian Milne-Edwards septal insertion cycles (6-12-24-48-96) and advection-diffusion-limited coral calice morphogenesis
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `e93f00e`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **305.25 s** (5.09 min) — Tier A: `1.44 s` · Tier B: `303.81 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **125,914** (input: `376`, output: `125,538`, subagent: `0`) — Tier A: `10,085` · Tier B: `115,829`
- **Total artifact tokens produced:** `36,204`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.08 | 0.00 | 0 | 1,355 | 0 | **1,355** | 1,355 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 572 | 0 | **572** | 572 | `paper_citations.bib` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 4,197 | 0 | **4,197** | 4,197 | `coral_calice_septa_insertion_reference.json`, `scleractinian_coral_septa_specimen.jpg`, `README.md` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 1,632 | 0 | **1,632** | 1,632 | `Loop_074_coral_calice_septa_insertion.lean`, `thomas-coral-calice-septa-insertion.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.32 | 0.00 | 0 | 2,201 | 0 | **2,201** | 2,201 | `2013-thomas-coral-calice-septa-insertion.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.78 | 0.00 | 0 | 128 | 0 | **128** | 128 | `coral_calice_septa_insertion_comparison.png`, `coral_calice_septa_insertion_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 2,201 | 0 | **2,201** | 2,201 | `2026-10-19-coral-calice-septa-insertion-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.66 | 0.00 | 0 | 736 | 0 | **736** | 736 | `2026-10-19-coral-calice-septa-insertion-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.38 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 24.30 | 0.00 | 0 | 1,619 | 0 | **1,619** | 692 | `published.json` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.10 | 0.00 | 0 | 1,931 | 0 | **1,931** | 1,931 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 278.28 | 0.00 | 376 | 108,966 | 0 | **109,342** | 20,559 | `2013-thomas-coral-calice-septa-insertion.py`, `coral_calice_septa_insertion_comparison.png`, `coral_calice_septa_insertion_growth.gif`, +3 more |
| — | **All** | **TOTAL** | — | `12/12` | **305.25** | **0.00** | **376** | **125,538** | **0** | **125,914** | **36,204** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/coral_calice_septa_insertion/scout_manifest.json` (4,866 B, `1,355` tok, SHA-256 `325761c7cb0d58db…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/coral_calice_septa_insertion/paper_citations.bib` (1,915 B, `572` tok, SHA-256 `156b2c3cca2ef939…`)

### Step 3 — `source` (Formal source)
- **Artifact:** `datasets/coral_calice_septa_insertion/coral_calice_septa_insertion_reference.json` (9,553 B, `2,807` tok, SHA-256 `9984d41d12285bd0…`)
- **Artifact:** `datasets/coral_calice_septa_insertion/scleractinian_coral_septa_specimen.jpg` (254,324 B, `64` tok, SHA-256 `904035c936843361…`)
- **Artifact:** `datasets/coral_calice_septa_insertion/README.md` (4,811 B, `1,326` tok, SHA-256 `d045377863514b33…`)

### Step 4 — `rules` (Pin the rules)
- **Artifact:** `morphoeval/lean/loops/Loop_074_coral_calice_septa_insertion.lean` (7,540 B, `64` tok, SHA-256 `5af5b48bee351626…`)
- **Artifact:** `morphoeval/math_kg_loops/thomas-coral-calice-septa-insertion.yaml` (5,758 B, `1,568` tok, SHA-256 `35a88c73e7589223…`)

### Step 5 — `script` (Script)
- **Artifact:** `2013-thomas-coral-calice-septa-insertion.py` (7,788 B, `2,201` tok, SHA-256 `bfc9347af2f5c73d…`)

### Step 6 — `output` (Output & compare)
- **Artifact:** `docs/media/coral_calice_septa_insertion_comparison.png` (529,340 B, `64` tok, SHA-256 `78be82bf5573c83c…`)
- **Artifact:** `docs/media/coral_calice_septa_insertion_growth.gif` (384,266 B, `64` tok, SHA-256 `56626097535a81d7…`)

### Step 7 — `audit` (Dual-agent audit)
- **Artifact:** `morphoeval/audits/2026-10-19-coral-calice-septa-insertion-audit.yaml` (8,858 B, `2,201` tok, SHA-256 `7b3bea8c80369bea…`)

### Step 8 — `measure` (Measure)
- **Artifact:** `morphoeval/audits/2026-10-19-coral-calice-septa-insertion-audit.measurements.yaml` (2,360 B, `736` tok, SHA-256 `c7301da0a4bf2298…`)

### Step 10 — `publish` (Sign & verify)
- **Artifact:** `morphoeval/kg/2026-10-19-coral-calice-septa-insertion-audit/published.json` (1,970 B, `692` tok, SHA-256 `46c853e4bbd395ba…`)

### Step 11 — `graph` (Public record)
- **Artifact:** `morphoeval/kg/2026-10-19-coral-calice-septa-insertion-audit/REPORT.md` (7,443 B, `1,931` tok, SHA-256 `3c48e70c1d06cb2f…`)

### Step 12 — `remed` (Remediate & re-score)
- **Artifact:** `2013-thomas-coral-calice-septa-insertion.py` (52,614 B, `14,241` tok, SHA-256 `43eb666dbe247cfe…`)
- **Artifact:** `docs/media/coral_calice_septa_insertion_comparison.png` (1,000,915 B, `64` tok, SHA-256 `d46e2020411ac23d…`)
- **Artifact:** `docs/media/coral_calice_septa_insertion_growth.gif` (215,409 B, `64` tok, SHA-256 `d53e16d853cad262…`)
- **Artifact:** `morphoeval/audits/2026-10-19-coral-calice-septa-insertion-audit-v2.yaml` (13,768 B, `3,684` tok, SHA-256 `22620290cfc14598…`)
- **Artifact:** `morphoeval/audits/2026-10-19-coral-calice-septa-insertion-audit-v2.measurements.yaml` (6,323 B, `1,820` tok, SHA-256 `10f094ad324e7197…`)
- **Artifact:** `morphoeval/kg/2026-10-19-coral-calice-septa-insertion-audit-v2/published.json` (1,970 B, `686` tok, SHA-256 `104f1f6b62c5daec…`)
