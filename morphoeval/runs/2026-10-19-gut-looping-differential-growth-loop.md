# MorphoEval Full-Loop Telemetry — `2026-10-19-gut-looping-differential-growth-loop`

- **Target simulation:** `savin-gut-looping-differential-growth`
- **Primary paper / dataset:** Savin et al. (2011) Nature — On the growth and form of the gut
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `45d4150`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **404.54 s** (6.74 min) — Tier A: `57.66 s` · Tier B: `346.88 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **287,246** (input: `29,815`, output: `257,431`, subagent: `0`) — Tier A: `48,275` · Tier B: `238,971`
- **Total artifact tokens produced:** `54,457`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 0.91 | 0.00 | 0 | 743 | 0 | **743** | 743 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 0.13 | 0.00 | 1,646 | 1,219 | 0 | **2,865** | 1,219 | `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.07 | 0.00 | 513 | 3,399 | 0 | **3,912** | 3,399 | `intestine_mesentery_specimen.jpg`, `gut_looping_differential_growth_reference.json` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.13 | 0.00 | 3,335 | 1,300 | 0 | **4,635** | 1,300 | `Loop_062_gut_looping_differential_growth.lean`, `savin-gut-looping-differential-growth.yaml` |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 55.40 | 0.00 | 3,399 | 28,989 | 0 | **32,388** | 3,604 | `2011-savin-gut-looping-differential-growth.py` |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 1.03 | 0.00 | 3,604 | 128 | 0 | **3,732** | 128 | `gut_looping_differential_growth_comparison.png`, `gut_looping_differential_growth_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 3,668 | 2,802 | 0 | **6,470** | 2,802 | `2026-10-19-gut-looping-differential-growth-audit.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.47 | 0.00 | 2,802 | 1,249 | 0 | **4,051** | 926 | `2026-10-19-gut-looping-differential-growth-audit.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.29 | 0.00 | 3,728 | 13,579 | 0 | **17,307** | 13,579 | `REPORT.md`, `morphoeval_2026-10-19-gut-looping-differential-growth-audit.ttl` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 14.38 | 0.00 | 42 | 3,008 | 0 | **3,050** | 708 | `published.json` |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.11 | 0.00 | 2,408 | 2,408 | 0 | **4,816** | 2,408 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 331.55 | 0.00 | 4,670 | 198,607 | 0 | **203,277** | 23,641 | `2011-savin-gut-looping-differential-growth.py`, `gut_looping_differential_growth_comparison.png`, `gut_looping_differential_growth_growth.gif`, +4 more |
| — | **All** | **TOTAL** | — | `12/12` | **404.54** | **0.00** | **29,815** | **257,431** | **0** | **287,246** | **54,457** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Scouted Savin et al. (2011) Nature 476:57-62 and Wikimedia Commons intestinal mesentery specimens
- **Artifact:** `datasets/gut_looping_differential_growth/scout_manifest.json` (2,594 B, `743` tok, SHA-256 `f7c39cdcaf077348…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Savin et al. (2011) Kirchhoff rod + neo-Hookean mesentery scaling laws and cross-species tables
- **Artifact:** `datasets/gut_looping_differential_growth/paper_citations.bib` (1,690 B, `513` tok, SHA-256 `21703c48df51dd52…`)
- **Artifact:** `datasets/gut_looping_differential_growth/README.md` (2,365 B, `706` tok, SHA-256 `1da32b807c44e2ed…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored Wikimedia Commons CC BY-SA 4.0 mesentery/intestine specimen and locked SHA-256 digests
- **Artifact:** `datasets/gut_looping_differential_growth/intestine_mesentery_specimen.jpg` (45,553 B, `64` tok, SHA-256 `e12fee9ac1fb5f3d…`)
- **Artifact:** `datasets/gut_looping_differential_growth/gut_looping_differential_growth_reference.json` (10,411 B, `3,335` tok, SHA-256 `f73121bb4140187f…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Proved 3 Lean 4 theorems for Savin et al. (2011) differential strain, cube-law wavelength stationarity, and annular inertia factorization (0 sorrys)
- **Artifact:** `morphoeval/lean/loops/Loop_062_gut_looping_differential_growth.lean` (4,772 B, `64` tok, SHA-256 `e8f7de2eef05180a…`)
- **Artifact:** `morphoeval/math_kg_loops/savin-gut-looping-differential-growth.yaml` (4,666 B, `1,236` tok, SHA-256 `d2217c423052ba4a…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 1 (v1) implementation of 2011-savin-gut-looping-differential-growth.py; Wrote Pass 1 (v1) baseline script
- **Artifact:** `2011-savin-gut-looping-differential-growth.py` (12,671 B, `3,604` tok, SHA-256 `4bf48928be4b1085…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Rendered Pass 1 comparison PNG and growth GIF
- **Artifact:** `docs/media/gut_looping_differential_growth_comparison.png` (242,127 B, `64` tok, SHA-256 `a509fa903edf6a28…`)
- **Artifact:** `docs/media/gut_looping_differential_growth_growth.gif` (213,780 B, `64` tok, SHA-256 `170b42d0a5576991…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 audit (56.4/100) identifying 4 concrete remediation targets
- **Artifact:** `morphoeval/audits/2026-10-19-gut-looping-differential-growth-audit.yaml` (11,296 B, `2,802` tok, SHA-256 `b4272c4ec95238b6…`)

### Step 8 — `measure` (Measure)
- **Notes:** Measured Pass 1 outputs and sidecar morphometrics
- **Artifact:** `morphoeval/audits/2026-10-19-gut-looping-differential-growth-audit.measurements.yaml` (2,871 B, `926` tok, SHA-256 `d5f5dff089078344…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Built Pass 1 ProbKnow-conformant nanopublications
- **Artifact:** `morphoeval/kg/2026-10-19-gut-looping-differential-growth-audit/REPORT.md` (9,046 B, `2,408` tok, SHA-256 `723a10b1880f7b2e…`)
- **Artifact:** `morphoeval/kg/2026-10-19-gut-looping-differential-growth-audit/morphoeval_2026-10-19-gut-looping-differential-growth-audit.ttl` (38,575 B, `11,171` tok, SHA-256 `a1c054c3d6317acb…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signed and verified all 10 Pass 1 nanopublications
- **Artifact:** `morphoeval/kg/2026-10-19-gut-looping-differential-growth-audit/published.json` (1,988 B, `708` tok, SHA-256 `27da7769a8be3ad5…`)

### Step 11 — `graph` (Public record)
- **Notes:** Synthesized Pass 1 REPORT.md
- **Artifact:** `morphoeval/kg/2026-10-19-gut-looping-differential-growth-audit/REPORT.md` (9,046 B, `2,408` tok, SHA-256 `723a10b1880f7b2e…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Remediating all 4 Pass 1 audit findings in 2011-savin-gut-looping-differential-growth.py (v2); Completed Pass 2 remediation: 3D Kirchhoff rod L-BFGS-B relaxation, 4-species collapse (r=0.9999), BMP2/Noggin/ablation controls, specimen CV segmentation, and superseding signed nanopubs (score 56.4 -> 92.6)
- **Artifact:** `2011-savin-gut-looping-differential-growth.py` (54,333 B, `14,347` tok, SHA-256 `037030eb8d32e782…`)
- **Artifact:** `docs/media/gut_looping_differential_growth_comparison.png` (952,385 B, `64` tok, SHA-256 `cb61655dfbcbf3d2…`)
- **Artifact:** `docs/media/gut_looping_differential_growth_growth.gif` (317,729 B, `64` tok, SHA-256 `7266ce228fe24dc5…`)
- **Artifact:** `morphoeval/audits/2026-10-19-gut-looping-differential-growth-audit-v2.yaml` (12,829 B, `3,457` tok, SHA-256 `ffcb59a43dc11048…`)
- **Artifact:** `morphoeval/audits/2026-10-19-gut-looping-differential-growth-audit-v2.measurements.yaml` (5,991 B, `1,821` tok, SHA-256 `676b2df4af6bbaba…`)
- **Artifact:** `morphoeval/kg/2026-10-19-gut-looping-differential-growth-audit-v2/REPORT.md` (11,015 B, `3,192` tok, SHA-256 `21304ac790f9985b…`)
- **Artifact:** `morphoeval/kg/2026-10-19-gut-looping-differential-growth-audit-v2/published.json` (1,988 B, `696` tok, SHA-256 `422ecf3fb3f662ff…`)
