# MorphoEval Full-Loop Telemetry — `2026-10-19-limb-bud-elongation-pushing-loop-pass3`

- **Target simulation:** `boehm-limb-bud-elongation-pushing`
- **Primary paper / dataset:** Boehm et al. (2010) PLoS Biol. — The role of oriented cell divisions in vertebrate limb bud morphogenesis
- **Builder model:** `gemini-4-argon`
- **Assessor models:** `gemini-4-argon` + `claude-sonnet-5-5-high`
- **Start commit:** `d58a084`
- **Completed steps:** `7 / 12`
- **Total wall-clock time:** **2130.71 s** (35.51 min) — Tier A: `1819.58 s` · Tier B: `311.13 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **263,319** (input: `7,228`, output: `256,091`, subagent: `0`) — Tier A: `208,441` · Tier B: `54,878`
- **Total artifact tokens produced:** `99,488`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | claude-fable-5-1 (`gemini-4-argon`) | `completed` | 1723.82 | 0.00 | 6,900 | 179,703 | 0 | **186,603** | 40,691 | `2015-boehm-limb-bud-elongation-pushing.py`, `Loop_017_limb_bud_elongation_pushing.lean`, `boehm-limb-bud-elongation-pushing.yaml`, +1 more |
| 6 | A | **output** (Output & compare) | claude-fable-5-1 (`gemini-4-argon`) | `completed` | 95.76 | 0.00 | 0 | 21,838 | 0 | **21,838** | 192 | `limb_bud_elongation_pushing_comparison.png`, `limb_bud_elongation_pushing_fem3d.png`, `limb_bud_elongation_pushing_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 | `completed` | 246.26 | 0.00 | 328 | 21,851 | 0 | **22,179** | 7,135 | `2026-10-19-limb-bud-elongation-pushing-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5-1 | `completed` | 19.94 | 0.00 | 0 | 30,518 | 0 | **30,518** | 4,235 | `2026-10-19-limb-bud-elongation-pushing-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5-1 | `completed` | 11.99 | 0.00 | 0 | 883 | 0 | **883** | 40,951 | `REPORT.md`, `morphoeval_2026-10-19-limb-bud-elongation-pushing-audit-v3.ttl` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5-1 | `completed` | 10.14 | 0.00 | 0 | 344 | 0 | **344** | 688 | `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5-1 | `completed` | 22.80 | 0.00 | 0 | 954 | 0 | **954** | 5,596 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `7/12` | **2130.71** | **0.00** | **7,228** | **256,091** | **0** | **263,319** | **99,488** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 remediation: 3D P1 tetrahedral FEM pressure-Poisson on an OPT visual-hull mesh reconstructed from the specimen's orthogonal iso-surface silhouettes; Pass 3: specimen visual-hull reconstruction + structured P1 tet mesh + P1-P1 stabilised 3D Stokes growth FEM (extensile OCD stress, FGF8 body force, ALE flank inflow); manufactured-solution oracles 8e-14; 2-gain calibration; flank-influx volume closure; silhouette IoU vs specimen St1; uniform-control Tc 12.1->16.5 h fix; Lean Theorems 4-5 (0 sorrys); 14-check self-test; 20 measurements
- **Artifact:** `2015-boehm-limb-bud-elongation-pushing.py` (119,023 B, `35,008` tok, SHA-256 `d81a723d4e504e88…`)
- **Artifact:** `morphoeval/lean/loops/Loop_017_limb_bud_elongation_pushing.lean` (13,694 B, `64` tok, SHA-256 `7f1474f5d1111b7c…`)
- **Artifact:** `morphoeval/math_kg_loops/boehm-limb-bud-elongation-pushing.yaml` (10,811 B, `2,933` tok, SHA-256 `8c2642b12587d759…`)
- **Artifact:** `datasets/limb_bud_elongation_pushing/README.md` (8,316 B, `2,686` tok, SHA-256 `07cb0f559359cfef…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 3 self-test + render; self-test 14/14 PASS (20 s); full render 23.7 s -> comparison.png (10 panels), fem3d.png (6 panels K-P, new), growth.gif (4 panels); CLI --flank-influx 61.4 verified
- **Artifact:** `docs/media/limb_bud_elongation_pushing_comparison.png` (1,024,457 B, `64` tok, SHA-256 `1b3c6fa5a82c58c7…`)
- **Artifact:** `docs/media/limb_bud_elongation_pushing_fem3d.png` (716,383 B, `64` tok, SHA-256 `72228a70b79bbcaf…`)
- **Artifact:** `docs/media/limb_bud_elongation_pushing_growth.gif` (597,749 B, `64` tok, SHA-256 `46c037dbd45ad054…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** write audit v3; audit v3 written and validated (--check OK); 7 dims re-scored 84/86/92/90/94/93/94, 5 Pass-4 remediation items, 3 new references
- **Artifact:** `morphoeval/audits/2026-10-19-limb-bud-elongation-pushing-audit-v3.yaml` (26,591 B, `7,135` tok, SHA-256 `0928da96fe3e963d…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs; measurements yaml generated (20 measurements)
- **Artifact:** `morphoeval/audits/2026-10-19-limb-bud-elongation-pushing-audit-v3.measurements.yaml` (14,717 B, `4,235` tok, SHA-256 `6063d710fb1b731f…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build KG with supersession; 10 nanopubs built, 948 triples, ProbKnow conformance OK, supersedes v2 KG
- **Artifact:** `morphoeval/kg/2026-10-19-limb-bud-elongation-pushing-audit-v3/REPORT.md` (19,662 B, `5,596` tok, SHA-256 `2c6a2f8c513c27c2…`)
- **Artifact:** `morphoeval/kg/2026-10-19-limb-bud-elongation-pushing-audit-v3/morphoeval_2026-10-19-limb-bud-elongation-pushing-audit-v3.ttl` (117,725 B, `35,355` tok, SHA-256 `9c8671aa0856cd1f…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** sign + verify (no publish); 10 nanopubs signed; check_signed.mjs 10 ok / 0 failed; NOT published (isolated)
- **Artifact:** `morphoeval/kg/2026-10-19-limb-bud-elongation-pushing-audit-v3/published.json` (1,952 B, `688` tok, SHA-256 `efcc10a80a3cb48e…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md check; REPORT.md: overall 89.1 (faithful), dims 84/86/92/90/94/93/94, supersedes v2 confirmed in claim+overall trig
- **Artifact:** `morphoeval/kg/2026-10-19-limb-bud-elongation-pushing-audit-v3/REPORT.md` (19,662 B, `5,596` tok, SHA-256 `2c6a2f8c513c27c2…`)
