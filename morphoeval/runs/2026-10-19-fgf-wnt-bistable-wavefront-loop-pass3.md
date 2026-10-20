# MorphoEval Full-Loop Telemetry — `2026-10-19-fgf-wnt-bistable-wavefront-loop-pass3`

- **Target simulation:** `baker-fgf-wnt-bistable-wavefront`
- **Primary paper / dataset:** Baker et al. (2006) Dev. Biol. — A clock and wavefront mechanism for somite formation
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `d58a084`
- **Completed steps:** `0 / 12`
- **Total wall-clock time:** **1997.90 s** (33.30 min) — Tier A: `1933.84 s` · Tier B: `64.06 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **204,686** (input: `6,712`, output: `197,974`, subagent: `0`) — Tier A: `189,075` · Tier B: `15,611`
- **Total artifact tokens produced:** `100,973`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 1923.68 | 0.00 | 6,712 | 182,159 | 0 | **188,871** | 40,971 | `2012-baker-fgf-wnt-bistable-wavefront.py`, `fgf_wnt_clock_lr_reference.json`, `README.md`, +3 more |
| 6 | A | **output** (Output & compare) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 10.16 | 0.00 | 0 | 204 | 0 | **204** | 128 | `fgf_wnt_bistable_wavefront_comparison.png`, `fgf_wnt_bistable_wavefront_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | assessor-agent (`claude-fable-5-1-thinking-high`) | `done` | 14.87 | 0.00 | 0 | 3,975 | 0 | **3,975** | 7,169 | `2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 13.32 | 0.00 | 0 | 2,458 | 0 | **2,458** | 3,712 | `2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 10.81 | 0.00 | 0 | 327 | 0 | **327** | 35,842 | `REPORT.md`, `morphoeval_2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.ttl` |
| 10 | B | **publish** (Sign & verify) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 10.37 | 0.00 | 0 | 523 | 0 | **523** | 684 | `published.json` |
| 11 | B | **graph** (Public record) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 14.60 | 0.00 | 0 | 1,159 | 0 | **1,159** | 5,298 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-agent (`claude-fable-5-1-thinking-high`) | `done` | 0.09 | 0.00 | 0 | 7,169 | 0 | **7,169** | 7,169 | `2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.yaml` |
| — | **All** | **TOTAL** | — | `0/12` | **1997.90** | **0.00** | **6,712** | **197,974** | **0** | **204,686** | **100,973** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 Step 12->5: couple bistable switch to Notch-Delta/Her1 clock lattice (saltatory arrest) + 2D bilateral LR strips (Vermot & Pourquie 2005); Pass 3 remediation: Lewis (2003) delayed Her1/Hes7 clock lattice with Notch-Delta coupling gating the bistable switch (saltatory cohort arrest); 2D bilateral paraxial strips with RA-buffered LR bias (Vermot & Pourquie 2005), 4 LR regimes; 6 new measurements; 3 new Lean theorems (6 total, 0 sorry); math_kg eqs 4-6; CLI flags --clock-delay/--notch-delay/--notch-coupling/--lr-bias/--situs/--print-pass3
- **Artifact:** `2012-baker-fgf-wnt-bistable-wavefront.py` (108,216 B, `29,655` tok, SHA-256 `6626e1777c95fcaa…`)
- **Artifact:** `datasets/fgf_wnt_bistable_wavefront/fgf_wnt_clock_lr_reference.json` (10,121 B, `2,714` tok, SHA-256 `16deab8565d154cb…`)
- **Artifact:** `datasets/fgf_wnt_bistable_wavefront/README.md` (7,957 B, `2,511` tok, SHA-256 `fe583658924993d4…`)
- **Artifact:** `datasets/fgf_wnt_bistable_wavefront/paper_citations.bib` (3,896 B, `1,161` tok, SHA-256 `9824d167487ab303…`)
- **Artifact:** `morphoeval/lean/loops/Loop_025_fgf_wnt_bistable_wavefront.lean` (9,360 B, `64` tok, SHA-256 `aeb61e9bd1e608ba…`)
- **Artifact:** `morphoeval/math_kg_loops/baker-fgf-wnt-bistable-wavefront.yaml` (16,663 B, `4,866` tok, SHA-256 `dcf9179cd1241998…`)

### Step 6 — `output` (Output & compare)
- **Notes:** self-test PASSED (7 sections, 4 pinned/verified dataset digests); 3x3 figure (panels G-I new) + bilateral GIF rendered in ~6 s
- **Artifact:** `docs/media/fgf_wnt_bistable_wavefront_comparison.png` (1,793,783 B, `64` tok, SHA-256 `ce04ae4bd568fab0…`)
- **Artifact:** `docs/media/fgf_wnt_bistable_wavefront_growth.gif` (623,469 B, `64` tok, SHA-256 `2278580ac2349f32…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** v3 audit: 92.0 overall (93/89/94/93/94/92/92), 6 new line-pinned Pass 4 remediation items, --check valid
- **Artifact:** `morphoeval/audits/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.yaml` (27,251 B, `7,169` tok, SHA-256 `95f8186b3c8fe4a9…`)

### Step 8 — `measure` (Measure)
- **Notes:** 11 measurements, 2 outputs fingerprinted, quick_gate 100/100
- **Artifact:** `morphoeval/audits/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.measurements.yaml` (13,125 B, `3,712` tok, SHA-256 `559d29c350bb6958…`)

### Step 9 — `build` (Write as claims)
- **Notes:** 10 nanopubs, 765 triples, ProbKnow conformance OK, supersedes v2
- **Artifact:** `morphoeval/kg/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3/REPORT.md` (18,921 B, `5,298` tok, SHA-256 `4ae1432a058873db…`)
- **Artifact:** `morphoeval/kg/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3/morphoeval_2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.ttl` (103,295 B, `30,544` tok, SHA-256 `6bbc7263963d1d88…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** sign only (not published): 10 nanopubs signed, check_signed.mjs 10 ok / 0 failed
- **Artifact:** `morphoeval/kg/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3/published.json` (1,943 B, `684` tok, SHA-256 `fff22d870256153f…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md confirms overall 92.0 (faithful) and supersession chain audit -> v2 -> v3
- **Artifact:** `morphoeval/kg/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3/REPORT.md` (18,921 B, `5,298` tok, SHA-256 `4ae1432a058873db…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** 6 new line-pinned Pass 4 remediation items recorded in the v3 audit (cohort CV 0.69, 2D vs 1D WT front, fitted b_LR, domain unification, explicit deltaC, CLI wiring)
- **Artifact:** `morphoeval/audits/2026-10-19-fgf-wnt-bistable-wavefront-audit-v3.yaml` (27,251 B, `7,169` tok, SHA-256 `95f8186b3c8fe4a9…`)
