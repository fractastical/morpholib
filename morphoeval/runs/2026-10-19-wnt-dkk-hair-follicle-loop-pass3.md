# MorphoEval Full-Loop Telemetry — `2026-10-19-wnt-dkk-hair-follicle-loop-pass3`

- **Target simulation:** `sick-wnt-dkk-hair-follicle`
- **Primary paper / dataset:** Sick et al. (2006) Science — WNT and DKK determine hair follicle spacing through a reaction-diffusion mechanism
- **Builder model:** `claude-fable-5-1`
- **Assessor models:** `claude-fable-5-1` + `claude-sonnet-5-5-high`
- **Start commit:** `64526a0`
- **Completed steps:** `8 / 12`
- **Total wall-clock time:** **1031.90 s** (17.20 min) — Tier A: `990.58 s` · Tier B: `41.32 s`
- **Total CPU time:** `114.40 s` user + `7.59 s` sys
- **Total LLM tokens:** **269,150** (input: `37,213`, output: `231,937`, subagent: `0`) — Tier A: `223,916` · Tier B: `45,234`
- **Total artifact tokens produced:** `65,065`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 2 | A | **paper** (Paper) | builder-agent (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 3 | A | **source** (Formal source) | builder-agent+tool (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`claude-fable-5-1`) | `pending` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`claude-fable-5-1`) | `completed` | 892.35 | 0.00 | 166 | 223,166 | 0 | **223,332** | 35,995 | `2002-sick-wnt-dkk-hair-follicle.py`, `Loop_006_wnt_dkk_hair_follicle.lean`, `sick-wnt-dkk-hair-follicle.yaml`, +3 more |
| 6 | A | **output** (Output & compare) | tool (`deterministic-python`) | `completed` | 98.23 | 96.98 | 100 | 484 | 0 | **584** | 128 | `wnt_dkk_hair_follicle_comparison.png`, `wnt_dkk_hair_follicle_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | claude-fable-5-1 (`claude-fable-5-1`) | `completed` | 8.68 | 0.00 | 32,693 | 1,204 | 0 | **33,897** | 8,231 | `2026-10-19-wnt-dkk-hair-follicle-audit-v3.yaml` |
| 8 | B | **measure** (Measure) | measure_outputs.py (`deterministic-python`) | `completed` | 22.88 | 22.57 | 38 | 449 | 0 | **487** | 3,810 | `2026-10-19-wnt-dkk-hair-follicle-audit-v3.measurements.yaml` |
| 9 | B | **build** (Write as claims) | build_morphoeval_kg.py (`deterministic-rdflib`) | `completed` | 1.89 | 1.68 | 74 | 160 | 0 | **234** | 6,078 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | sign_morphoeval.py+check_signed.mjs (`rsa-sha256-nanopub`) | `completed` | 0.70 | 0.76 | 74 | 942 | 0 | **1,016** | 677 | `published.json` |
| 11 | B | **graph** (Public record) | nanopub-kg-index (`deterministic-rdflib`) | `completed` | 7.08 | 0.00 | 0 | 1,464 | 0 | **1,464** | 6,078 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`claude-fable-5-1`) | `completed` | 0.09 | 0.00 | 4,068 | 4,068 | 0 | **8,136** | 4,068 | `2026-10-19-wnt-dkk-hair-follicle-audit-v2.yaml` |
| — | **All** | **TOTAL** | — | `8/12` | **1031.90** | **121.99** | **37,213** | **231,937** | **0** | **269,150** | **65,065** | — |

## Step Notes & Provenance

### Step 5 — `script` (Script)
- **Notes:** Pass 3 Step 5: implement 3D dermal condensate chemotaxis + hair peg downgrowth coupling, new self-test checks, new measurements, new Lean theorems; Pass 3 Step 5: added Section 3b (3D 42x42x12 dermal slab, Keller-Segel FGF20 chemotaxis + BMP4 repulsion with volume filling, conservative face-flux scheme, per-placode hair peg depth ODE, WT/Fgf20-null/BMP4-excess/chemotaxis-null genotypes, 5-point chi_f sweep, dermal linear stability), 12-panel figure (+row D) and Phase 3 GIF frames, 6 new measurements (13 total), self-test 8->12 checks, CLI --chi-f; 3 new Lean laws (dermal_subsystem_instructed_stable, keller_segel_fourier_det_sign + ks_det_neg_of_supercritical, hair_peg_depth_monotone + closed_form; lean exit 0, 0 sorrys) registered in math_kg yaml; fixed math_kg/README parameter parity (D_w 0.015, D_h 0.45, kappa 0.02, d 30); 7 references appended to bib and SHA re-pinned
- **Artifact:** `2002-sick-wnt-dkk-hair-follicle.py` (89,596 B, `25,935` tok, SHA-256 `873dc24606fd7fe8…`)
- **Artifact:** `morphoeval/lean/loops/Loop_006_wnt_dkk_hair_follicle.lean` (9,469 B, `64` tok, SHA-256 `a50e59f58a5c174a…`)
- **Artifact:** `morphoeval/math_kg_loops/sick-wnt-dkk-hair-follicle.yaml` (11,386 B, `3,331` tok, SHA-256 `770b60d61baa43e0…`)
- **Artifact:** `datasets/wnt_dkk_hair_follicle/wnt_dkk_hair_follicle_reference.json` (12,302 B, `3,640` tok, SHA-256 `e2081b983d742461…`)
- **Artifact:** `datasets/wnt_dkk_hair_follicle/paper_citations.bib` (4,516 B, `1,379` tok, SHA-256 `7843d2b728a0e537…`)
- **Artifact:** `datasets/wnt_dkk_hair_follicle/README.md` (5,989 B, `1,646` tok, SHA-256 `9f32e77d0f1476fb…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Pass 3 Step 6: --self-test (12/12) then full render of 12-panel PNG + 45-frame GIF; Pass 3 Step 6 (re-run after the seam-aware placode counting fix found while auditing: WT 16->15 periodic local maxima, K14-Wnt 32->30, enlarged 12->10): --self-test (12/12) then full render of 12-panel PNG + 45-frame GIF
- **Artifact:** `docs/media/wnt_dkk_hair_follicle_comparison.png` (1,017,546 B, `64` tok, SHA-256 `2ede41f73dae7b11…`)
- **Artifact:** `docs/media/wnt_dkk_hair_follicle_growth.gif` (1,084,034 B, `64` tok, SHA-256 `d8761a5a0efcd16f…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 3 Step 7: superseding v3 audit, re-scored on all 7 dimensions with line-pinned evidence; Pass 3 Step 7: v3 audit 85.9 (MF 86, SM 80, PE 86, MV 84, ST 94, CT 90, RP 91; down from 91.9 after exposing the seam double-counting artifact, confounded K14-Wnt preset and loose vibrissa specimen); 7 line-pinned Pass 4 remediation items; --check valid
- **Artifact:** `morphoeval/audits/2026-10-19-wnt-dkk-hair-follicle-audit-v3.yaml` (30,988 B, `8,231` tok, SHA-256 `e35295d3bcad44c8…`)

### Step 8 — `measure` (Measure)
- **Notes:** Pass 3 Step 8: fingerprint rendered PNG/GIF and run measure_morphoeval() (13 measurements)
- **Artifact:** `morphoeval/audits/2026-10-19-wnt-dkk-hair-follicle-audit-v3.measurements.yaml` (13,643 B, `3,810` tok, SHA-256 `4bd008a2a5a18aa9…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Pass 3 Step 9: build superseding ProbKnow nanopubs from v3 audit (--supersede-from v2, --no-resolve-heads --conform)
- **Artifact:** `morphoeval/kg/2026-10-19-wnt-dkk-hair-follicle-audit-v3/REPORT.md` (21,421 B, `6,078` tok, SHA-256 `8d8a66b74200a809…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Pass 3 Step 10: sign only (no publish) and verify all signed TriG
- **Artifact:** `morphoeval/kg/2026-10-19-wnt-dkk-hair-follicle-audit-v3/published.json` (1,889 B, `677` tok, SHA-256 `e6fa3c45e7912c41…`)

### Step 11 — `graph` (Public record)
- **Notes:** Pass 3 Step 11: confirm overall score and superseded audit id in REPORT.md; Pass 3 Step 11: REPORT.md confirms overall 85.9 (faithful) and supersession chain audit -> v2 -> v3; 10 nanopubs signed and verified (10 ok, 0 failed), not published
- **Artifact:** `morphoeval/kg/2026-10-19-wnt-dkk-hair-follicle-audit-v3/REPORT.md` (21,421 B, `6,078` tok, SHA-256 `8d8a66b74200a809…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 3 Step 12 triage: 1 open Pass 2 remediation item (3D dermal condensate FGF20/BMP4 chemotaxis + hair peg downgrowth) plus README/math_kg parameter parity defects found on read; Pass 3 Step 12: triaged the single open Pass 2 item (couple 2D epidermal sheet to 3D dermal condensate FGF20/BMP4 chemotaxis + hair peg downgrowth) and two parity defects (datasets README D_w/D_h/d and math_kg yaml D_w/D_h/kappa disagree with the vendored JSON and the script); re-entering at Step 5
- **Artifact:** `morphoeval/audits/2026-10-19-wnt-dkk-hair-follicle-audit-v2.yaml` (14,866 B, `4,068` tok, SHA-256 `7f48365854ceda93…`)
