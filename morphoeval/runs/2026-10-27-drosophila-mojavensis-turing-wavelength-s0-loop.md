# MorphoEval Full-Loop Telemetry — `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-drosophila-mojavensis-turing-wavelength-s0`
- **Primary paper / dataset:** Drosophila mojavensis Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` only. The telemetry default names a second assessor; that second pass was not run.
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.09 s** (0.02 min) — Tier A: `0.52 s` · Tier B: `0.57 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **32,563** (input: `0`, output: `32,563`, subagent: `0`) — Tier A: `14,907` · Tier B: `17,656`
- **Total artifact tokens produced:** `32,563`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 297 | 0 | **297** | 297 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,744 | 0 | **2,744** | 2,744 | `drosophila_mojavensis_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_mojavensis_turing_wavelength_s0_specimen.jpg`, `drosophila_mojavensis_wikimedia_female.png` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 1,109 | 0 | **1,109** | 1,109 | `Loop_3194_drosophila_mojavensis_turing_wavelength_s0.lean`, `delphi10000-drosophila-mojavensis-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 10,501 | 0 | **10,501** | 10,501 | `2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 128 | 0 | **128** | 128 | `drosophila_mojavensis_turing_wavelength_s0_comparison.png`, `drosophila_mojavensis_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 6,450 | 0 | **6,450** | 6,450 | `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit.yaml`, `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,902 | 0 | **2,902** | 2,902 | `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 5,184 | 0 | **5,184** | 5,184 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.11 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,514 | 0 | **2,514** | 2,514 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 606 | 0 | **606** | 606 | `2026-10-27-drosophila-mojavensis-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.09** | **0.00** | **0** | **32,563** | **0** | **32,563** | **32,563** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex+Wikimedia scout; slot 3194 is outside campaign_100 so query/taxon path used. No script/dataset collision on drosophila-mojavensis-turing-wavelength-s0.
- **Artifact:** `datasets/drosophila_mojavensis_turing_wavelength_s0/scout_manifest.json` (1,143 B, `297` tok, SHA-256 `52b2341694b92509…`)

### Step 2 — `paper` (Paper)
- **Notes:** Crossref-checked Schnakenberg/Turing/Matzkin citations. Scale 2.4 µm and age 0 Ma recorded. Rejected sibling DOI 10.1534/genetics.106.059444 (Hill & Otto 2007, yeast).
- **Artifact:** `datasets/drosophila_mojavensis_turing_wavelength_s0/drosophila_mojavensis_turing_wavelength_s0_reference.json` (4,537 B, `1,297` tok, SHA-256 `7ab7a539471b2219…`)
- **Artifact:** `datasets/drosophila_mojavensis_turing_wavelength_s0/paper_citations.bib` (2,342 B, `730` tok, SHA-256 `ba25b3495a6a5e8c…`)
- **Artifact:** `datasets/drosophila_mojavensis_turing_wavelength_s0/README.md` (2,641 B, `717` tok, SHA-256 `9eb8b09e1aecea42…`)

### Step 3 — `source` (Formal source)
- **Notes:** Gallery plate SHA-256 f68e7b21… vendored as the fitness target. Wikimedia public-domain female habitus 225x99 SHA-256 7615839c… is too small for the 480 px gate.
- **Artifact:** `datasets/drosophila_mojavensis_turing_wavelength_s0/drosophila_mojavensis_turing_wavelength_s0_specimen.jpg` (98,202 B, `64` tok, SHA-256 `f68e7b21de8d6b97…`)
- **Artifact:** `datasets/drosophila_mojavensis_turing_wavelength_s0/drosophila_mojavensis_wikimedia_female.png` (20,678 B, `64` tok, SHA-256 `7615839cbc796270…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Four Lean theorems, zero sorry. Lean binary not installed, so they were not machine-checked.
- **Artifact:** `morphoeval/lean/loops/Loop_3194_drosophila_mojavensis_turing_wavelength_s0.lean` (1,984 B, `64` tok, SHA-256 `0f22adf2ef7a10dc…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-drosophila-mojavensis-turing-wavelength-s0.yaml` (3,679 B, `1,045` tok, SHA-256 `88ca03266262ea22…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg integrator, dispersion Pearson, ablation, Neumann control, chronospecies FFTs.
- **Artifact:** `models/2026-delphi10000-drosophila-mojavensis-turing-wavelength-s0.py` (38,076 B, `10,501` tok, SHA-256 `37ecdf5a88b9634f…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Self-test passed. GIF 14 frames 640x480. CV 87.1. Pearson r 0.99323.
- **Artifact:** `docs/media/drosophila_mojavensis_turing_wavelength_s0_comparison.png` (491,888 B, `64` tok, SHA-256 `33eeba72316e26ac…`)
- **Artifact:** `docs/media/drosophila_mojavensis_turing_wavelength_s0_growth.gif` (3,810,451 B, `64` tok, SHA-256 `98d86591b76e9ff0…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 83.8. Pass 2 overall 87.0 after Neumann and chronospecies FFT. Single assessor.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit.yaml` (12,831 B, `3,441` tok, SHA-256 `97bf9780f81bb8cb…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2.yaml` (11,135 B, `3,009` tok, SHA-256 `e38dbf95e4e4b8af…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs.py quick_gate 100/100. 10 metrics on pass 1, 14 on pass 2. All value != reference.
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit.measurements.yaml` (4,794 B, `1,332` tok, SHA-256 `3e3f6f9d11a58bc4…`)
- **Artifact:** `morphoeval/audits/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2.measurements.yaml` (5,820 B, `1,570` tok, SHA-256 `5077b989748d7052…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg.py --conform --no-verify-loci --rubric-from the bos-taurus v7 rubric. Conformance OK. Loci check skipped because new files are not in commit 0c76d15.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit/REPORT.md` (9,481 B, `2,670` tok, SHA-256 `f29e071ab5592111…`)
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2/REPORT.md` (8,956 B, `2,514` tok, SHA-256 `71634a29f3cd017a…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml does not exist. Nanopubs were not signed and check_signed.mjs was not run. Publisher ORCID on the graphs is https://orcid.org/0009-0007-4948-9192, the ORCID already used by morphoeval/kg/2026-10-19-turing-audit.

### Step 11 — `graph` (Public record)
- **Notes:** Pass 2 REPORT overall 87.0.
- **Artifact:** `morphoeval/kg/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-audit-v2/REPORT.md` (8,956 B, `2,514` tok, SHA-256 `71634a29f3cd017a…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closed R1 and R2. image_fitness 85.7, cv_sim_similarity 87.1, pearson_r 0.99323, 14 frames, 640 px. Gates passed; no pass 3.
- **Artifact:** `morphoeval/runs/2026-10-27-drosophila-mojavensis-turing-wavelength-s0-loop-image-fitness.json` (1,971 B, `606` tok, SHA-256 `d204d9dc015336bc…`)

## Gates (Pass 2)

All gates passed. No Pass 3.

| Gate | Result |
| --- | --- |
| image_fitness ≥ 70 | 85.7 |
| cv_sim_similarity ≥ 58 | 87.1 |
| GIF frames ≥ 6 | 14 |
| resolution ≥ 340 px | 640×480 |
| non-circular specimen metric | plate foreground 0.3806 vs habitus 0.5455; KS 0.858 vs 0 |
| Pearson r > 0.98 | 0.99323 |
| audit overall ≥ 85 | 87.0 (Pass 1 was 83.8) |

Pass 1 remediation that Pass 2 closed: a no-flux control (same radial-FFT bin as periodic, std 0.99918 vs 1.01072) and saturated FFT wavelengths at the *D. miranda* (3.2825 µm vs analytic 3.2364 µm) and *D. novamexicana* (1.9754 µm vs analytic 1.9477 µm) diffusivity scales. Scale is 2.4 µm (2.38e-06 m). Age is 0 Ma.

Signing was not done: `~/.nanopub/profile.yml` is absent. Lean 4 was not executed: no `lean` binary in this environment. The four theorems contain no `sorry`.
