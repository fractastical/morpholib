# MorphoEval Full-Loop Telemetry — `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-acanthostega-gunnari-turing-wavelength-s0`
- **Primary paper / dataset:** Acanthostega gunnari Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `dff4f2b5`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **1.01 s** (0.02 min) — Tier A: `0.50 s` · Tier B: `0.51 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **30,642** (input: `0`, output: `30,642`, subagent: `0`) — Tier A: `16,881` · Tier B: `13,761`
- **Total artifact tokens produced:** `30,642`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | human+scout-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,239 | 0 | **2,239** | 2,239 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 3,436 | 0 | **3,436** | 3,436 | `acanthostega_gunnari_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent+tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 64 | 0 | **64** | 64 | `acanthostega_gunnari_brussels_head_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,586 | 0 | **1,586** | 1,586 | `Loop_3102_acanthostega_gunnari_turing_wavelength_s0.lean`, `delphi10000-acanthostega-gunnari-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 9,428 | 0 | **9,428** | 9,428 | `2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 128 | 0 | **128** | 128 | `acanthostega_gunnari_turing_wavelength_s0_comparison.png`, `acanthostega_gunnari_turing_wavelength_s0_growth.gif` |
| 7 | B | **audit** (Dual-agent audit) | dual-assessors (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,966 | 0 | **2,966** | 2,966 | `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2.yaml` |
| 8 | B | **measure** (Measure) | tool (measure_outputs.py) (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 1,984 | 0 | **1,984** | 1,984 | `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool (build_morphoeval_kg.py) (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,707 | 0 | **2,707** | 2,707 | `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool (sign_morphoeval.py + check_signed.mjs) (`grok-4.7`) | `completed` | 0.08 | 0.00 | 0 | 2,707 | 0 | **2,707** | 2,707 | `REPORT.md` |
| 11 | B | **graph** (Public record) | tool (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 2,707 | 0 | **2,707** | 2,707 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder+assessor (`grok-4.7`) | `completed` | 0.09 | 0.00 | 0 | 690 | 0 | **690** | 690 | `2026-10-27-acanthostega-gunnari-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **1.01** | **0.00** | **0** | **30,642** | **0** | **30,642** | **30,642** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex/Wikimedia scout. Slot 3102 is outside campaign_100_slots.yaml. No slug collision.; OpenAlex/Wikimedia scout. Slot 3102 is outside campaign_100_slots.yaml. No slug collision.
- **Artifact:** `datasets/acanthostega_gunnari_turing_wavelength_s0/scout_manifest.json` (7,698 B, `2,239` tok, SHA-256 `e931a1606fa6cf4d…`)

### Step 2 — `paper` (Paper)
- **Notes:** Extracted Schnakenberg k_c, 105 µm, 365 Ma, chronoseries vs Tiktaalik and Ichthyostega.; Extracted Schnakenberg k_c, 105 µm, 365 Ma, chronoseries vs Tiktaalik and Ichthyostega.
- **Artifact:** `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_turing_wavelength_s0_reference.json` (4,989 B, `1,471` tok, SHA-256 `dc929ab5cb065aa6…`)
- **Artifact:** `datasets/acanthostega_gunnari_turing_wavelength_s0/paper_citations.bib` (3,280 B, `1,036` tok, SHA-256 `7a351b92a6a7092a…`)
- **Artifact:** `datasets/acanthostega_gunnari_turing_wavelength_s0/README.md` (3,494 B, `929` tok, SHA-256 `e536875ac09df1a8…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored CC BY-SA 3.0 Brussels skull, SHA-256 pinned.; Vendored CC BY-SA 3.0 Brussels skull, SHA-256 pinned.
- **Artifact:** `datasets/acanthostega_gunnari_turing_wavelength_s0/acanthostega_gunnari_brussels_head_specimen.jpg` (241,730 B, `64` tok, SHA-256 `6eae655c40f0444c…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean Loop_3102, five theorems, zero sorry, plus math KG yaml.; Lean Loop_3102, five theorems, zero sorry, plus math KG yaml.
- **Artifact:** `morphoeval/lean/loops/Loop_3102_acanthostega_gunnari_turing_wavelength_s0.lean` (1,717 B, `64` tok, SHA-256 `a72ca5de4cf9734a…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-acanthostega-gunnari-turing-wavelength-s0.yaml` (5,414 B, `1,522` tok, SHA-256 `bb44f611c6c714a0…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg solver, dispersion fit, sweep, ablation, specimen CV, self-test.; Schnakenberg solver, dispersion fit, sweep, ablation, specimen CV, self-test.
- **Artifact:** `models/2026-delphi10000-acanthostega-gunnari-turing-wavelength-s0.py` (32,260 B, `9,428` tok, SHA-256 `336a82a5c140e27a…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Comparison PNG and 12-frame 640x480 GIF. Self-test passed.; Comparison PNG and 12-frame 640x480 GIF. Self-test passed.
- **Artifact:** `docs/media/acanthostega_gunnari_turing_wavelength_s0_comparison.png` (557,252 B, `64` tok, SHA-256 `ee8fcd1b8df7a3e3…`)
- **Artifact:** `docs/media/acanthostega_gunnari_turing_wavelength_s0_growth.gif` (3,408,681 B, `64` tok, SHA-256 `9161f32ecf0fd605…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 77.5 at r=16 (FFT 125.53 µm). Pass 2 overall 87.1 at r=10.; Pass 1 overall 77.5 at r=16 (FFT 125.53 µm). Pass 2 overall 87.1 at r=10.
- **Artifact:** `morphoeval/audits/2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2.yaml` (10,858 B, `2,966` tok, SHA-256 `1b71102d1a3b0123…`)

### Step 8 — `measure` (Measure)
- **Notes:** 12 non-circular measurements. Pearson r=0.9999. Quick gate 90/100.; 12 non-circular measurements. Pearson r=0.9999. Quick gate 90/100.
- **Artifact:** `morphoeval/audits/2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2.measurements.yaml` (7,446 B, `1,984` tok, SHA-256 `42e2630c0444bb14…`)

### Step 9 — `build` (Write as claims)
- **Notes:** Unsigned TriG. ProbKnow conformance OK. Rubric reintroduction allowed locally; not signed.; Unsigned TriG. ProbKnow conformance OK. Rubric reintroduction allowed locally; not signed.
- **Artifact:** `morphoeval/kg/2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2/REPORT.md` (9,691 B, `2,707` tok, SHA-256 `ea76e6b30e7570eb…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** No ~/.nanopub/profile.yml. sign_morphoeval.py not run. No trusty URI fabricated.; No ~/.nanopub/profile.yml. sign_morphoeval.py not run. No trusty URI fabricated.
- **Artifact:** `morphoeval/kg/2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2/REPORT.md` (9,691 B, `2,707` tok, SHA-256 `ea76e6b30e7570eb…`)

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md overall 87.1 faithful. Supersession chain audit -> audit-v2.; REPORT.md overall 87.1 faithful. Supersession chain audit -> audit-v2.
- **Artifact:** `morphoeval/kg/2026-10-27-acanthostega-gunnari-turing-wavelength-s0-audit-v2/REPORT.md` (9,691 B, `2,707` tok, SHA-256 `ea76e6b30e7570eb…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 retuned r from 16 to 10, restricted the dispersion band, and marked Tiktaalik cross-ray as subcritical. All gates passed, so pass 3 was not run.; Pass 2 retuned r from 16 to 10, restricted the dispersion band, and marked Tiktaalik cross-ray as subcritical. All gates passed, so pass 3 was not run.
- **Artifact:** `morphoeval/runs/2026-10-27-acanthostega-gunnari-turing-wavelength-s0-loop-image-fitness.json` (2,352 B, `690` tok, SHA-256 `441d0d0975ca1091…`)

## Pass 2 gates

Pass 3 was not run. Every gate cleared after the r = 10 retune.

| Gate | Result |
| --- | --- |
| image_fitness | 90.5 (>= 70) |
| cv_sim_similarity | 96.0 (>= 58) |
| GIF frames | 12 (>= 6, and >= 10 required by the render) |
| resolution | 640×480 (>= 340, and >= 480 required by the render) |
| non-circular specimen metrics | 5 (wavelength, foreground, coherency, circularity, KS; none copy the reference) |
| Pearson r | 0.9999 (> 0.98) |
| audit overall | 87.1 (>= 85) |

Signing: `~/.nanopub/profile.yml` is absent. Nanopublications were built and passed the offline ProbKnow conformance check. They were not signed, and no trusty URI was invented. The publisher IRI on the unsigned graphs is the repository ORCID in `.zenodo.json`.

Pass 1 (r = 16) left the saturated wavelength at 125.53 µm (overall 77.5). Pass 2 moved to r = 10 (FFT 106.93 µm), kept the dispersion comparison on modes that stay linear, and recorded Tiktaalik's cross-ray mode as subcritical (σ = −0.621).
