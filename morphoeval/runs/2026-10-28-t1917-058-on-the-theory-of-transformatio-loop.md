# MorphoEval Full-Loop Telemetry — `2026-10-28-t1917-058-on-the-theory-of-transformatio-loop`

- **Target simulation:** `thompson1917-t1917-058`
- **Primary paper / dataset:** Thompson, D.W. (1917) On Growth and Form, Ch. XVII, pp. 770-773, Figs. 404-408
- **Builder model:** `claude-fable-5.1`
- **Assessor models:** `claude-fable-5.1` + `pending (R6: independent cross-provider assessor not yet run)`
- **Start commit:** `fd43e5aba`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **3177.71 s** (52.96 min) — Tier A: `2255.29 s` · Tier B: `922.42 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **124,144** (input: `0`, output: `124,144`, subagent: `0`) — Tier A: `75,036` · Tier B: `49,108`
- **Total artifact tokens produced:** `105,387`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | claude-fable-5.1 (`claude-fable-5.1`) | `completed` | 1428.01 | 0.00 | 0 | 4,909 | 0 | **4,909** | 4,909 | `scout_manifest.json`, `dryad_7r85116_metadata.json` |
| 2 | A | **paper** (Paper) | claude-fable-5.1 (`claude-fable-5.1`) | `completed` | 140.68 | 0.00 | 0 | 6,800 | 0 | **6,800** | 6,800 | `t1917_058_on_the_theory_of_transformatio_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | claude-fable-5.1 (`claude-fable-5.1`) | `completed` | 0.08 | 0.00 | 0 | 448 | 0 | **448** | 448 | `pan_troglodytes_lateral.jpg`, `papio_ursinus_lateral.jpg`, `canis_familiaris_lateral.jpg`, +4 more |
| 4 | A | **rules** (Pin the rules) | claude-fable-5.1 (`claude-fable-5.1`) | `completed` | 278.25 | 0.00 | 0 | 2,875 | 0 | **2,875** | 2,875 | `Thompson1917_T1917_058.lean`, `thompson1917-t1917-058.yaml` |
| 5 | A | **script** (Script) | claude-fable-5.1 (`claude-fable-5.1`) | `completed` | 385.75 | 0.00 | 0 | 35,686 | 0 | **35,686** | 22,763 | `t1917-058_on-the-theory-of-transformations.py` |
| 6 | A | **output** (Output & compare) | claude-fable-5.1 (`claude-fable-5.1`) | `completed` | 22.53 | 0.00 | 0 | 24,318 | 0 | **24,318** | 18,484 | `t1917_058_on_the_theory_of_transformatio_comparison.png`, `t1917_058_on_the_theory_of_transformatio_growth.gif`, `t1917_058_metrics.json`, +1 more |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5.1 | `completed` | 0.17 | 0.00 | 0 | 9,328 | 0 | **9,328** | 9,328 | `2026-10-28-t1917-058-on-the-theory-of-transformatio-audit.yaml`, `2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | claude-fable-5.1 | `completed` | 77.23 | 0.00 | 0 | 9,637 | 0 | **9,637** | 9,637 | `2026-10-28-t1917-058-on-the-theory-of-transformatio-audit.measurements.yaml`, `2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | claude-fable-5.1 | `completed` | 114.73 | 0.00 | 0 | 5,863 | 0 | **5,863** | 5,863 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | claude-fable-5.1 | `completed` | 52.29 | 0.00 | 0 | 1,240 | 0 | **1,240** | 1,240 | `published.json`, `published.json` |
| 11 | B | **graph** (Public record) | claude-fable-5.1 | `completed` | 45.60 | 0.00 | 0 | 10,518 | 0 | **10,518** | 10,518 | `REPORT.md`, `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | claude-fable-5.1 | `completed` | 632.40 | 0.00 | 0 | 12,522 | 0 | **12,522** | 12,522 | `2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2.yaml`, `REPORT.md`, `result.json` |
| — | **All** | **TOTAL** | — | `12/12` | **3177.71** | **0.00** | **0** | **124,144** | **0** | **124,144** | **105,387** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** Named Dryad dataset 7r85116 = platyrrhines only -> DATASET_MISMATCH (metadata pinned, download blocked by bearer-token/Anubis). Verdict datasets: Thompson 1917 Figs 404-408 (IA scan, PD) + 4 CC Commons lateral skulls of the exact taxa.
- **Artifact:** `datasets/thompson_1917/t1917-058/scout_manifest.json` (13,738 B, `3,715` tok, SHA-256 `4aab29fefc62323b…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/dryad_7r85116_metadata.json` (5,126 B, `1,194` tok, SHA-256 `3a9d4e15adcd608a…`)

### Step 2 — `paper` (Paper)
- **Notes:** Equations E1-E7 (TPS whole-diagram map, affine/bending decomposition, cell-area log gradient, Procrustes explained fraction, facial proportions); six sub-claims C1-C6 extracted from pp. 770-773
- **Artifact:** `datasets/thompson_1917/t1917-058/t1917_058_on_the_theory_of_transformatio_reference.json` (14,074 B, `4,425` tok, SHA-256 `4fdcd4cba2786813…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/paper_citations.bib` (4,057 B, `1,135` tok, SHA-256 `f2b7ee8ec50937e4…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/README.md` (4,501 B, `1,240` tok, SHA-256 `cc71f9b0b580d13b…`)

### Step 3 — `source` (Formal source)
- **Notes:** 3 IA scan leaves (PD) + 5 figure crops + 4 CC BY-SA 3.0 Commons lateral skulls vendored; SHA-256 of all 13 files pinned in reference JSON; Fig.405 net digitised (30 nodes), 11 landmarks x 6 configurations
- **Artifact:** `datasets/thompson_1917/t1917-058/pan_troglodytes_lateral.jpg` (112,871 B, `64` tok, SHA-256 `04455ef234b6ac4a…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/papio_ursinus_lateral.jpg` (125,077 B, `64` tok, SHA-256 `02e80d49035d3940…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/canis_familiaris_lateral.jpg` (134,261 B, `64` tok, SHA-256 `e5b1eab566faf5ed…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/homo_sapiens_lateral.jpg` (131,964 B, `64` tok, SHA-256 `3457d096022c4b4f…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/thompson_1917_leaf789_p770_figs404_405.jpg` (504,989 B, `64` tok, SHA-256 `a0235719db9caefe…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/thompson_1917_leaf790_p771_figs406_407.jpg` (693,490 B, `64` tok, SHA-256 `91e1e4e75acb9961…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/thompson_1917_leaf792_p773_fig408.jpg` (691,050 B, `64` tok, SHA-256 `9c1e561e91602cca…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean 4.15.0 (elan installed in-run): 4 laws / 6 theorems, 0 sorry, lean exits 0 (affine net congruence + flat area field; log-area gradient monotonicity; intensity family T_k monotone area; rectangular vs triangular taper)
- **Artifact:** `morphoeval/lean/loops/Thompson1917_T1917_058.lean` (12,653 B, `64` tok, SHA-256 `679d28a3654acb4d…`)
- **Artifact:** `morphoeval/math_kg_loops/thompson1917-t1917-058.yaml` (9,703 B, `2,811` tok, SHA-256 `26aad5d9b6af472c…`)

### Step 5 — `script` (Script)
- **Notes:** Pass 2 (v2): R1-R5 remediation; Pass 2 v2: jitter bootstrap (1000) + permutation null (2000) + r/RMSE/KS/Wilcoxon, lambda & k sweeps with k* scan, mirrored-net & specificity controls, silhouette segmentation taper/face fraction, Fig.404-reference robustness
- **Artifact:** `thompson/claims/t1917-058_on-the-theory-of-transformations.py` (80,631 B, `22,763` tok, SHA-256 `aa9dac3afd5912fc…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 2; Pass 2 outputs: self-test OK (2.7 s), full run 7.5 s; E_chimp 0.351, bootstrap CI [0.19,0.41], permutation p=0.0005, r 0.92->0.97, RMSE 0.116->0.075, mirrored net -0.60, k* chimp 0.85 / baboon 1.15 / dog 1.30, dog taper 0.88 lowest; 16/17 checks (cos chimp-baboon 0.78 < 0.8 fails)
- **Artifact:** `docs/media/t1917_058_on_the_theory_of_transformatio_comparison.png` (1,145,183 B, `64` tok, SHA-256 `93e8738afc53ea10…`)
- **Artifact:** `docs/media/t1917_058_on_the_theory_of_transformatio_growth.gif` (1,179,678 B, `64` tok, SHA-256 `0f3e8a28771947c1…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/t1917_058_metrics.json` (49,339 B, `16,620` tok, SHA-256 `e96b962e952b3dbd…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/result.json` (5,252 B, `1,736` tok, SHA-256 `21c63895002e39ec…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 2; Pass 2 audit v2 supersedes Pass 1: overall 72.7 (MF74 SM66 PE72 MV70 ST85 CT78 RP74); C1,C3,C4,C5,C6 supported, C2 mixed
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit.yaml` (17,225 B, `4,405` tok, SHA-256 `eebfd564d11d481f…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2.yaml` (18,256 B, `4,923` tok, SHA-256 `f73d309e14cb5adc…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 2; Pass 2: 43 measurements
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit.measurements.yaml` (14,300 B, `3,926` tok, SHA-256 `bf6efb34aa102319…`)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2.measurements.yaml` (21,306 B, `5,711` tok, SHA-256 `3a9efd28bc8be7cf…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 2; Pass 2: build --supersede-from <pass1 kg> --no-resolve-heads --conform (--rubric-from bos-taurus-v7 head): 9 nanopubs, 1302 triples, ProbKnow conformance OK; Pass 1 nanopubs were signed-only (never published) so npx:supersedes cannot be attached - supersession recorded at audit level (supersedes_audit_id)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2/REPORT.md` (21,051 B, `5,863` tok, SHA-256 `d41d11c0e2dc1f92…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 2; Pass 2: 9 nanopubs signed (local key, sign-only, not published to the registry); check_signed.mjs 9 ok 0 failed
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit/published.json` (1,694 B, `620` tok, SHA-256 `acd17a42037bc02e…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2/published.json` (1,694 B, `620` tok, SHA-256 `813836b496918c25…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2; Pass 2 REPORT.md verified: overall 72.6 (mostly faithful), supersession chain pass1 -> v2, 1302 triples, 43 measurements, ProbKnow conformance OK
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit/REPORT.md` (17,195 B, `4,655` tok, SHA-256 `bc20745378c2e657…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2/REPORT.md` (21,051 B, `5,863` tok, SHA-256 `3c75eb07eb3b8a86…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 remediation R1-R6; Pass 2 closed: R1-R5 implemented (62.1 -> 72.6); open: R6 second assessor, R7 population sample, R8 Fig.407/408 nets, R9 real human photo. Verdict SUPPORTED (C1,C3,C4,C5,C6 supported; C2 mixed)
- **Artifact:** `morphoeval/audits/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2.yaml` (18,256 B, `4,923` tok, SHA-256 `cf2ebc179078811c…`)
- **Artifact:** `morphoeval/kg/2026-10-28-t1917-058-on-the-theory-of-transformatio-audit-v2/REPORT.md` (21,051 B, `5,863` tok, SHA-256 `3c75eb07eb3b8a86…`)
- **Artifact:** `datasets/thompson_1917/t1917-058/result.json` (5,252 B, `1,736` tok, SHA-256 `21c63895002e39ec…`)
