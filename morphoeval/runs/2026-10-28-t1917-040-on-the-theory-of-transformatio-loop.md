# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-040-on-the-theory-of-transformatio-loop`

- **Target simulation:** `thompson1917-t1917-040`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. XVII, pp. ~724-735, Figs. 353-358
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `6f8773168`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **3079.43 s** (51.32 min) — Tier A: `2145.52 s` · Tier B: `933.91 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **114,098** (input: `0`, output: `114,098`, subagent: `0`) — Tier A: `62,718` · Tier B: `51,380`
- **Total artifact tokens produced:** `98,554`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | cloud-agent (`claude-fable-5-1`) | `completed` | 1039.05 | 0.00 | 0 | 7,167 | 0 | **7,167** | 7,167 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | cloud-agent (`claude-fable-5-1`) | `completed` | 70.41 | 0.00 | 0 | 8,760 | 0 | **8,760** | 8,760 | `t1917_040_on_the_theory_of_transformatio_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | cloud-agent (`claude-fable-5-1`) | `completed` | 0.08 | 0.00 | 0 | 704 | 0 | **704** | 704 | `helen_keller_1904_profile.jpg`, `romy_schneider_profile.jpg`, `roberts_2013_profile.jpg`, +8 more |
| 4 | A | **rules** (Pin the rules) | cloud-agent (`claude-fable-5-1`) | `completed` | 188.65 | 0.00 | 0 | 2,664 | 0 | **2,664** | 2,664 | `Thompson1917_T1917_040.lean`, `thompson1917-t1917-040.yaml` |
| 5 | A | **script** (Script) | cloud-agent (`claude-fable-5-1`) | `completed` | 834.51 | 0.00 | 0 | 36,175 | 0 | **36,175** | 22,804 | `t1917-040_on-the-theory-of-transformations.py` |
| 6 | A | **output** (Output & compare) | cloud-agent (`claude-fable-5-1`) | `completed` | 12.82 | 0.00 | 0 | 7,248 | 0 | **7,248** | 5,075 | `t1917_040_on_the_theory_of_transformatio_comparison.png`, `t1917_040_on_the_theory_of_transformatio_growth.gif`, `t1917_040_metrics.json`, +1 more |
| 7 | B | **audit** (Dual-agent audit) | cloud-agent (`claude-fable-5-1`) | `completed` | 123.63 | 0.00 | 0 | 8,488 | 0 | **8,488** | 8,488 | `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit.yaml`, `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | cloud-agent (`claude-fable-5-1`) | `completed` | 1.75 | 0.00 | 0 | 6,389 | 0 | **6,389** | 6,389 | `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit.measurements.yaml`, `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | cloud-agent (`claude-fable-5-1`) | `completed` | 89.52 | 0.00 | 0 | 4,331 | 0 | **4,331** | 4,331 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | cloud-agent (`claude-fable-5-1`) | `completed` | 66.16 | 0.00 | 0 | 1,247 | 0 | **1,247** | 1,247 | `published.json`, `published.json` |
| 11 | B | **graph** (Public record) | cloud-agent (`claude-fable-5-1`) | `completed` | 13.03 | 0.00 | 0 | 8,121 | 0 | **8,121** | 8,121 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | cloud-agent (`claude-fable-5-1`) | `completed` | 639.83 | 0.00 | 0 | 22,804 | 0 | **22,804** | 22,804 | `t1917-040_on-the-theory-of-transformations.py` |
| — | **All** | **TOTAL** | — | `12/12` | **3079.43** | **0.00** | **0** | **114,098** | **0** | **114,098** | **98,554** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Dryad pointer (cichlid TSA GM) = DATASET_MISMATCH, unused; IA scan ongrowthform1917thom leaves 748-750,759,761 pinned (Durer heads Figs 364/366 at pp. 740/742); 9 CC/PD Commons profile photos vendored, 6 rejected; OpenAlex 10+6 hits
- **Artifact:** `datasets/thompson_1917/t1917-040/scout_manifest.json` (24,023 B, `7,167` tok, SHA-256 `c738aa8c422f8f96…`)

### Step 2 — `paper` (Paper)
- **Notes:** E1-E5 (stretch, log-stretch, shear x=xi-eta cot w, affine, facial-angle shear law), claims C1-C3, parameter regime table, page/figure correction (Durer heads = Figs 364/366 pp 740/742)
- **Artifact:** `datasets/thompson_1917/t1917-040/t1917_040_on_the_theory_of_transformatio_reference.json` (21,482 B, `6,632` tok, SHA-256 `07745a7171b7648f…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/paper_citations.bib` (4,180 B, `1,126` tok, SHA-256 `1279402c7d3411df…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/README.md` (3,630 B, `1,002` tok, SHA-256 `e6c1038f13517e28…`)

### Step 3 — `source` (Formal source)
- **Notes:** 9 CC/PD Commons profile photographs + 5 IA page scans vendored; SHA-256 pinned in reference JSON; Dryad pointer = mismatch (fish), unused
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/helen_keller_1904_profile.jpg` (189,614 B, `64` tok, SHA-256 `7fa2a48b647c3b09…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/romy_schneider_profile.jpg` (113,680 B, `64` tok, SHA-256 `0857743562b65c3a…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/roberts_2013_profile.jpg` (117,919 B, `64` tok, SHA-256 `643f14a150df84e0…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/nell_walden_1905_profile.png` (230,036 B, `64` tok, SHA-256 `b1f49c57c7aeefee…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/nasta_rojc_profile.jpg` (45,469 B, `64` tok, SHA-256 `4b5ed492fb19ec93…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/winifred_odunoku_profile.jpg` (30,618 B, `64` tok, SHA-256 `b48ece4b38e69468…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/margaret_of_hesse_profile.jpg` (192,366 B, `64` tok, SHA-256 `1b38e84fc0eb1594…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/kubra_gumusay_2023_profile.jpg` (71,115 B, `64` tok, SHA-256 `aaa06c6f5c336da7…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/specimens/beverly_sills_profile.jpg` (69,726 B, `64` tok, SHA-256 `679a08ff1f459089…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/thompson_1917_pages/ongrowthform1917thom_leaf759_p740.jpg` (518,179 B, `64` tok, SHA-256 `9e583cdcbd5c7ced…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/thompson_1917_pages/ongrowthform1917thom_leaf761_p742.jpg` (539,306 B, `64` tok, SHA-256 `214db8a09bd3fdfe…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** 6 theorems (shear group law, whole-grid shear, Camper facial-angle index, stretch circle->ellipse, affine closure, oblique transposition invertible); lean 4.35.0 exits 0, 0 sorry; KG yaml registers 6 laws
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_040.lean` (7,618 B, `64` tok, SHA-256 `9516621d378fb0ae…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-040.yaml` (8,877 B, `2,600` tok, SHA-256 `4ef3b0c7f4761102…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2; Pass 2 script v2.0 (R2-R8 remediated)
- **Artifact:** `thompson/claims/t1917-040_on-the-theory-of-transformations.py` (76,685 B, `22,804` tok, SHA-256 `a720822197e12ca3…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2; Pass 2 outputs rendered (~6 s)
- **Artifact:** `docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png` (608,335 B, `64` tok, SHA-256 `fe2b61061643bce3…`)
- **Artifact:** `docs/media/t1917_040_on_the_theory_of_transformatio_growth.gif` (255,591 B, `64` tok, SHA-256 `f3798ef508dfe7c0…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/t1917_040_metrics.json` (8,867 B, `2,947` tok, SHA-256 `2b1d42d3c1059487…`)
- **Artifact:** `datasets/thompson_1917/t1917-040/result.json` (6,044 B, `2,000` tok, SHA-256 `dd154e282abbc5c2…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 audit v2 supersedes v1; 7 dimensions re-scored with Pass-2 evidence
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit.yaml` (15,208 B, `3,978` tok, SHA-256 `dfb11a9e7c428eda…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2.yaml` (16,848 B, `4,510` tok, SHA-256 `69379e4d23d5a02a…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit.measurements.yaml` (7,945 B, `2,805` tok, SHA-256 `050382a8e4efb97c…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2.measurements.yaml` (11,086 B, `3,584` tok, SHA-256 `0f15be861d64bca0…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2; Pass 2 KG built --supersede-from v1 --no-resolve-heads --conform; 9 nanopubs, 617 triples, conformance OK
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2/REPORT.md` (14,891 B, `4,331` tok, SHA-256 `48b66a96b24c34d4…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 2; Pass 2: 9 nanopubs signed (local ephemeral key), check_signed 9 ok
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit/published.json` (1,694 B, `631` tok, SHA-256 `0a86801f51467b0e…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2/published.json` (1,694 B, `616` tok, SHA-256 `e12f47ad29942a07…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit/REPORT.md` (13,291 B, `3,790` tok, SHA-256 `dcd8b7e5a0d4e1ac…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2/REPORT.md` (14,891 B, `4,331` tok, SHA-256 `48b66a96b24c34d4…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 v2.0: Camper-frame null + Durer 5-dof fit + affine bound, dense semilandmarks, LOO-CV, noise floor, positive/negative/ablation controls, pose sensitivity, E5 r/slope, KS, omega/c sweep, Hough digitisation of Fig. 366 (81/99 deg); verdict mixed (strong form not supported)
- **Artifact:** `thompson/claims/t1917-040_on-the-theory-of-transformations.py` (76,685 B, `22,804` tok, SHA-256 `a720822197e12ca3…`)
