# MorphoEval Full-Loop Telemetry — `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-loop`

- **Target simulation:** `delphi10000-dictyostelium-discoideum-turing-wavelength-s0`
- **Primary paper / dataset:** Dictyostelium discoideum Turing Reaction-Diffusion Wavelength Selection
- **Builder model:** `grok-4.7`
- **Assessor models:** `grok-4.7` + `not-run`
- **Start commit:** `0c76d15d`
- **Completed steps:** `12 / 12`
- **Total wall-clock time:** **41.00 s** (0.68 min) — Tier A: `32.00 s` · Tier B: `9.00 s`
- **Total CPU time:** `0.00 s` user + `0.00 s` sys
- **Total LLM tokens:** **34,889** (input: `0`, output: `34,889`, subagent: `0`) — Tier A: `17,398` · Tier B: `17,491`
- **Total artifact tokens produced:** `34,889`

## Per-Step Breakdown (12-Step MorphoEval Loop)

| # | Tier | Step | Actor / Model | Status | Wall (s) | CPU (s) | Input tok | Output tok | Subagent tok | Total LLM tok | Artifact tok | Key Artifacts |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A | **scout** (Scout) | builder-agent (`grok-4.7`) | `completed` | 17.00 | 0.00 | 0 | 1,846 | 0 | **1,846** | 1,846 | `scout_manifest.json` |
| 2 | A | **paper** (Paper) | builder-agent (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 3,865 | 0 | **3,865** | 3,865 | `dictyostelium_discoideum_turing_wavelength_s0_reference.json`, `paper_citations.bib`, `README.md` |
| 3 | A | **source** (Formal source) | builder-agent (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 64 | 0 | **64** | 64 | `dictyostelium_discoideum_turing_wavelength_s0_specimen.jpg` |
| 4 | A | **rules** (Pin the rules) | builder-agent (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 1,191 | 0 | **1,191** | 1,191 | `Loop_3168_dictyostelium_discoideum_turing_wavelength_s0.lean`, `delphi10000-dictyostelium-discoideum-turing-wavelength-s0.yaml` |
| 5 | A | **script** (Script) | builder-agent (`grok-4.7`) | `completed` | 3.00 | 0.00 | 0 | 7,874 | 0 | **7,874** | 7,874 | `2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` |
| 6 | A | **output** (Output & compare) | tool (`grok-4.7`) | `completed` | 6.00 | 0.00 | 0 | 2,558 | 0 | **2,558** | 2,558 | `dictyostelium_discoideum_turing_wavelength_s0_comparison.png`, `dictyostelium_discoideum_turing_wavelength_s0_growth.gif`, `dictyostelium_discoideum_turing_wavelength_s0_metrics.json` |
| 7 | B | **audit** (Dual-agent audit) | builder-assessor (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 2,166 | 0 | **2,166** | 2,166 | `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit.yaml` |
| 8 | B | **measure** (Measure) | tool | `completed` | 1.00 | 0.00 | 0 | 3,694 | 0 | **3,694** | 3,694 | `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit.measurements.yaml`, `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2.measurements.yaml` |
| 9 | B | **build** (Write as claims) | tool | `completed` | 3.00 | 0.00 | 0 | 4,960 | 0 | **4,960** | 4,960 | `REPORT.md`, `REPORT.md` |
| 10 | B | **publish** (Sign & verify) | tool | `completed` | 0.00 | 0.00 | 0 | 0 | 0 | **0** | 0 | — |
| 11 | B | **graph** (Public record) | tool | `completed` | 1.00 | 0.00 | 0 | 2,771 | 0 | **2,771** | 2,771 | `REPORT.md` |
| 12 | B->A | **remed** (Remediate & re-score) | builder-assessor (`grok-4.7`) | `completed` | 2.00 | 0.00 | 0 | 3,900 | 0 | **3,900** | 3,900 | `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2.yaml`, `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-loop-image-fitness.json` |
| — | **All** | **TOTAL** | — | `12/12` | **41.00** | **0.00** | **0** | **34,889** | **0** | **34,889** | **34,889** | — |

## Step Notes & Provenance

### Step 1 — `scout` (Scout)
- **Notes:** OpenAlex and Wikimedia scout. Slot 3168 is outside campaign_100_slots.yaml. No collision with dictyostelium-discoideum-turing-wavelength-s0.
- **Artifact:** `datasets/dictyostelium_discoideum_turing_wavelength_s0/scout_manifest.json` (6,404 B, `1,846` tok, SHA-256 `4a189a1919bb7c65…`)

### Step 2 — `paper` (Paper)
- **Notes:** Reference JSON, BibTeX, and README state scale 2.304 µm (2.304e-06 m) and age 0 Ma.
- **Artifact:** `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_reference.json` (5,752 B, `1,658` tok, SHA-256 `b4d3f1e6feedaad3…`)
- **Artifact:** `datasets/dictyostelium_discoideum_turing_wavelength_s0/paper_citations.bib` (3,894 B, `1,209` tok, SHA-256 `2c7ca5d088db26f9…`)
- **Artifact:** `datasets/dictyostelium_discoideum_turing_wavelength_s0/README.md` (3,683 B, `998` tok, SHA-256 `509146f7abd00675…`)

### Step 3 — `source` (Formal source)
- **Notes:** Vendored public-domain Wikimedia sorocarp by Bruno in Columbus (2008). SHA-256 e2d2b739211bda975bc175167debf554af4991ce21bc8b1106d1053e6bc95a0b.
- **Artifact:** `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_specimen.jpg` (209,184 B, `64` tok, SHA-256 `e2d2b739211bda97…`)

### Step 4 — `rules` (Pin the rules)
- **Notes:** Lean namespace MathOfBiology.Loop_3168 with four decide theorems and zero sorry. Kernel not installed.
- **Artifact:** `morphoeval/lean/loops/Loop_3168_dictyostelium_discoideum_turing_wavelength_s0.lean` (1,775 B, `64` tok, SHA-256 `aeec8fff1bfd7fd7…`)
- **Artifact:** `morphoeval/math_kg_loops/delphi10000-dictyostelium-discoideum-turing-wavelength-s0.yaml` (4,058 B, `1,127` tok, SHA-256 `749eb0f2efeb60a3…`)

### Step 5 — `script` (Script)
- **Notes:** Schnakenberg solver, --self-test passed: lambda 2.303721 µm, ablation sigma negative.
- **Artifact:** `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` (27,980 B, `7,874` tok, SHA-256 `fe2b43e59216af47…`)

### Step 6 — `output` (Output & compare)
- **Notes:** Full run. Pearson r=0.9999999973, RMSE=3.668e-05, KS=0.087, CV=83.2, GIF 12 frames 960x720.
- **Artifact:** `docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png` (292,818 B, `64` tok, SHA-256 `bac77ec104f165d2…`)
- **Artifact:** `docs/media/dictyostelium_discoideum_turing_wavelength_s0_growth.gif` (5,497,667 B, `64` tok, SHA-256 `b332e96863e53865…`)
- **Artifact:** `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_metrics.json` (7,403 B, `2,430` tok, SHA-256 `2df3e28ead71868e…`)

### Step 7 — `audit` (Dual-agent audit)
- **Notes:** Pass 1 overall 72.8. Remediation: sweep, ablation, specimen segmentation, goodness-of-fit, second assessor.
- **Artifact:** `morphoeval/audits/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit.yaml` (8,445 B, `2,166` tok, SHA-256 `a4f096b38aa31217…`)

### Step 8 — `measure` (Measure)
- **Notes:** measure_outputs quick_gate 100/100. Ten measurements, all value != reference.
- **Artifact:** `morphoeval/audits/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit.measurements.yaml` (6,341 B, `1,741` tok, SHA-256 `5e9eb7d7a79c02b0…`)
- **Artifact:** `morphoeval/audits/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2.measurements.yaml` (7,083 B, `1,953` tok, SHA-256 `a1d7eaa38accf333…`)

### Step 9 — `build` (Write as claims)
- **Notes:** build_morphoeval_kg --conform --allow-reintroduction. Pass 1 and Pass 2 conformance OK.
- **Artifact:** `morphoeval/kg/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit/REPORT.md` (8,108 B, `2,189` tok, SHA-256 `da631e7757c6ddeb…`)
- **Artifact:** `morphoeval/kg/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2/REPORT.md` (10,189 B, `2,771` tok, SHA-256 `31232f6af23bc39e…`)

### Step 10 — `publish` (Sign & verify)
- **Notes:** Signing unavailable: ~/.nanopub/profile.yml is absent. Nanopubs were not signed and signatures were not fabricated.

### Step 11 — `graph` (Public record)
- **Notes:** REPORT.md present for Pass 1 (overall 72.8) and Pass 2 (overall 86.5).
- **Artifact:** `morphoeval/kg/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2/REPORT.md` (10,189 B, `2,771` tok, SHA-256 `31232f6af23bc39e…`)

### Step 12 — `remed` (Remediate & re-score)
- **Notes:** Pass 2 closes sweep, ablation, specimen CV segmentation, and GOF. image_fitness=82.4, cv=83.2, frames=12, min_px=720, pearson_r=0.9999999973, audit=86.5. All Pass-2 gates pass. No Pass 3.
- **Artifact:** `morphoeval/audits/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2.yaml` (11,538 B, `2,981` tok, SHA-256 `1c76dae23fc11541…`)
- **Artifact:** `morphoeval/runs/2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-loop-image-fitness.json` (3,008 B, `919` tok, SHA-256 `1415f0aecc409244…`)

## Pass 2 gates

All gates passed. Pass 3 was not run.

| Gate | Result |
| --- | --- |
| image_fitness | 82.4 (>= 70) |
| cv_sim_similarity | 83.2 (>= 58) |
| GIF frames | 12 (>= 6) |
| min resolution | 720 px (>= 340) |
| non-circular specimen metrics | 10 |
| Pearson r (dispersion) | 0.9999999973 (> 0.98) |
| audit overall | 86.5 (>= 85) |

Pass 1 overall was 72.8. Pass 2 closed the diffusion-ratio sweep, the equal-diffusion ablation, the sorocarp border-contrast segmentation, and the dispersion goodness-of-fit.

Signing was not available: `~/.nanopub/profile.yml` is absent, so the nanopubs were conformance-checked and left unsigned. The Lean file has four `decide` theorems and no `sorry`; the Lean kernel was not installed in this environment.

The 2.304 µm wavelength is the slot coordinate for a cortical Schnakenberg mode. The vendored photograph is a public-domain sorocarp, not a cortical micrograph. Pixel identity between those two images is not claimed. The tracker token column counts artifact text, not a separate model-API bill.
