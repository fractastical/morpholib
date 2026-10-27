# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-ficusphila-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-ficusphila-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila ficusphila Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` (single assessor; no second provider was available)
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1046.62 s** (17.44 min) — Tier A: `1046.07 s` · Tier B: `0.55 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **2,995** (input: `0`, output: `2,995`, subagent: `0`) — Tier A: `2,995` · Tier B: `0`
- **Total artifact tokens produced:** `2,995`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`gemini-4-argon`) | `completed` | 369.67 | 0.00 | 0 | 200 | 0 | **200** | 200 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`gemini-4-argon`) | `completed` | 676.04 | 0.00 | 0 | 2,795 | 0 | **2,795** | 2,795 | `drosophila_ficusphila_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 4 | A | **rules** (Pin the rules) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 5 | A | **script** (Script) | builder-agent (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 6 | A | **output** (Output & compare) | tool (`gemini-4-argon`) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 0.09 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor | `completed` | 0.10 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| — | **All** | **TOTAL** | — | `12/12` | **1046.62** | **0.00** | **0** | **2,995** | **0** | **2,995** | **2,995** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Artifact:** `datasets/drosophila_ficusphila_turing_wavelength_s0/scout_manifest.json` (780 B, `200` tok, SHA-256 `0a7fb1641ea126b0…`)

### Step 2 — `paper` (Paper)
- **Artifact:** `datasets/drosophila_ficusphila_turing_wavelength_s0/drosophila_ficusphila_turing_wavelength_s0_reference.json` (4,940 B, `1,377` tok, SHA-256 `bb3981fcb66333ba…`)
- **Artifact:** `datasets/drosophila_ficusphila_turing_wavelength_s0/paper_citations.bib` (2,032 B, `650` tok, SHA-256 `26d1e013287e2844…`)
- **Artifact:** `datasets/drosophila_ficusphila_turing_wavelength_s0/README.md` (2,736 B, `768` tok, SHA-256 `21c38422aa9196e1…`)

## Result (pass 2)

Schnakenberg–Turing wavelength for the *Drosophila ficusphila* longitudinal sex-comb ridge at **2.1 µm (2.072e-6 m, log10 = −5.684)** and **0 Ma**. Predecessor *D. eugracilis* λ = 3.108 µm (density 2/3). Successor *D. grimshawi* λ = 4.144 µm (density 1/2). Same Jacobian, different D_u. These are not the 100 µm sibling parameters.

- Audit v2 overall **88.8** (pass 1 was 80.7). Supersedes `2026-10-27-drosophila-ficusphila-turing-wavelength-s0-audit`.
- image_fitness **83.7** = 0.55×81.9 + 0.30×84 + 0.15×90. CV similarity **81.9**. GIF 16 frames, 800×600.
- Pearson r of analytic vs Euler σ(k) = 0.99999994. RMSE = 5.2e-4. Equal-diffusion std ratio = 0.018. KS = 0.1879. FFT centroid = 1.9175 µm (reference 2.072 µm).
- Gates after pass 2 all hold. Pass 3 was not required.
- Signing: `~/.nanopub/profile.yml` is absent. Both KG directories passed `build_morphoeval_kg.py --conform`. No nanopub was signed and no trusty URI was invented.
- R5 stays open: one assessor only.
- Step wall-clock for steps 3–12 in the table above is the bookkeeping close, not the render time. The script, GIF, audits and KG were produced before those rows were closed.
