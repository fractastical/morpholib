# Loop Round: Turing Patterns for Cows — Murray (1981) vs. *Bos taurus* Coat Spotting

**Run ID:** `2026-10-26-bos-taurus-turing-wavelength-loop`
**Target simulation:** `murray-bos-taurus-turing-wavelength` (`models/1981-murray-bos-taurus-turing-wavelength.py`, new this round)
**Organism:** *Bos taurus*, Holstein-Friesian domestic cow
**Audit:** `morphoeval/audits/2026-10-26-bos-taurus-turing-wavelength-audit.yaml` — **57.6 / 100** (impl. 71.7 · bio. 35.0 · repro. 73.7), single assessor (builder-scored; second assessor pending, R1)
**Sources (Crossref-verified 2026-10-26):** Turing 1952 [doi:10.1098/rstb.1952.0012](https://doi.org/10.1098/rstb.1952.0012) · Murray 1981 [doi:10.1016/0022-5193(81)90334-9](https://doi.org/10.1016/0022-5193(81)90334-9) · MITF cattle spotting 2016 [doi:10.1007/s13258-016-0493-4](https://doi.org/10.1007/s13258-016-0493-4) (counter-evidence)

![Specimen flank, Otsu mask, and Turing simulation side by side](../../datasets/bos_taurus_turing_wavelength/bos_taurus_turing_wavelength_comparison.png)

![Animation: cow flank mask (static) vs. the Turing field self-organising from noise into spots (`--animate`, 41 frames)](../../datasets/bos_taurus_turing_wavelength/bos_taurus_turing_wavelength_animation.gif)

## Why this round

`Bos taurus` had no Turing loop: it existed only as a gallery entry whose image is airway histology (not coat), with no `turing_wavelength` dataset, script, audit or run. This round scouts the classic claim — Murray's (1981) reaction-diffusion pre-pattern for mammalian coat markings — against the most iconic "Turing cow" of all, the black-and-white Holstein, and runs the full cycle once: scout → formal source → script → measure → audit → remediation list. No empirical series was invented; every specimen number below is measured by the script from one pinned photograph.

## Steps 1–6: scout, source, script, output

- **Scout (1–2):** Murray (1981) coat-marking pre-pattern, tested on Holstein-Friesian flank patches; MITF/KIT melanoblast account carried as the competing hypothesis from the start.
- **Formal source (3):** `datasets/bos_taurus_turing_wavelength/` — public-domain Holstein photograph (Province of Ontario Picture Bureau, c.1920, via Wikimedia Commons), SHA-256 `74f82f3d…572555` pinned in the reference JSON and re-verified at runtime; two declared flank ROIs.
- **Pin the rules (4):** `morphoeval/lean/loops/Loop_bos_taurus_turing_wavelength.lean` — 4 kernel-checked theorems on the ×5-scaled Jacobian (tr < 0, det > 0, Turing inequality 5625 > 2000, equal-diffusion cross-term ≤ 0). `lean` exits 0, 0 `sorry`.
- **Script (5):** Schnakenberg RD with exact dispersion analysis, an equal-diffusion negative control, a γ ∈ {0.25, 1, 4} sweep, Otsu specimen segmentation, and identical patch morphometrics on both sides. `--self-test` passes (~1.5 s); full run ~7 s.
- **Output (6):** figure above + `bos_taurus_turing_wavelength_metrics.json` in the dataset dir.

## Steps 7–8: audit and measurement

Linear theory (γ=1): tr J = −0.20, det J = 1.0, Turing inequality 225 > 80, unstable band \(k \in [0.272, 0.822]\), \(k_c = 0.612\), \(\lambda_c = 10.26\) px, \(\sigma_{\max} = 0.263\); equal-diffusion control \(\sigma_{\max} = -0.10\) (no instability, as required). Simulated FFT wavelength 11.9 px vs \(\lambda_c\) 10.3 px.

| Statistic (flank ROI) | Cow specimen | Turing sim γ=0.25 | γ=1 | γ=4 |
|---|---:|---:|---:|---:|
| Big pigment patches | **3** | 21 | 142 | 434 |
| Pigment coverage | **0.77** | 0.42 | 0.36 | 0.37 |
| Largest patch share of pigment | **0.992** | 0.155 | 0.013 | 0.003 |
| Patch diameter CV | **1.25** | 0.46 | 0.13 | 0.05 |
| Median patch diameter / width | 0.032 (largest: 0.76) | 0.111 | 0.049 | 0.024 |

Sensitivity: the tighter inner-flank ROI gives the same picture (coverage 0.85, largest share 0.98).

**Verdict:** a single-wavelength Turing pre-pattern makes many quasi-uniform wavelength-scale patches; this cow's flank is essentially *one* body-scale patch. Lowering γ towards the cow's count (21 patches at γ=0.25) still misses by 7× while abandoning the wavelength selection that is the mechanism's point. **Single-wavelength Turing is not morphometrically supported for Holstein flank spotting** — the pattern is consistent with the melanoblast-migration (MITF/KIT) account, where patch scale is set by clone/migration territories, not by \(2\pi/k_c\). The audit's low `specimen-match` (20) records exactly this failure; `model-fidelity` (70) records that the Turing machinery itself is correctly implemented and controlled.

## Steps 9–12: record and remediation

Audit validates against rubric 0.1.0 (`build_morphoeval_kg.py --check`: *valid*; 3 warnings, all "file does not exist at checkpoint commit" because this round is uncommitted — commit the new files to pin it, then a KG build can follow). Nanopubs were **not** signed or published this round. Next round's list (from the audit): **R1** independent cross-provider second assessor · **R2** ≥10 Holstein flanks so patch statistics become a distribution · **R3** body-mask segmentation instead of rectangular ROIs · **R4** pin kinetics to Murray's own system or rename the claim · **R5** add a melanoblast-migration positive control that *can* make one body-scale patch.

## Remediation passes (2026-10-26, prompted by visual mismatch)

- **Pass 2** (`...-loop-pass2.md`, audit-v2, **57.6 → 66.0**): growing-domain RD (pattern laid down small, then grown) + free coat polarity + explicit morphometric fit objective. Distance 5.22 → 1.19; the animation now grows the pattern with the domain.
- **Pass 3** (`...-loop-pass3.md`, audit-v3, **66.0 → 68.6**): boundary-roughness morphometrics (cow enclaves 11.9 vs sim ~5 — the largest remaining gap, and a mechanistic fingerprint of clonal growth over RD) + two-scale γ heterogeneity (best: growing-het γ=0.25, het=0.8, q=0.75; coverage 0.75 vs cow 0.77, largest-patch share 0.997 vs 0.992). Both fits are on n=1 and are reported as fits, not validation; R1/R2/R3/R5 remain open.
- **Pass 4 — Fable round** (`...-loop-pass4-fable.md`, second-assessor audit): blinded **Fable** assessor scores the same working tree **54.4** (Δ −14.2 vs builder v3; panel in the pass-4 record). It verified the maths, DOIs and bit-level reproducibility (and scored repro *higher*, 80), but found the Pass-2/3 fit tautological on its heaviest terms, the leaderboard inside seed noise, ROI artefacts inflating the cow statistics, and three attribution errors (Schnakenberg-vs-Murray kinetics, MITF authors, roughness disc constant) — the factual ones fixed in the script/`.bib` afterwards. R1 is now closed; the melanoblast positive control (R5) is the best-motivated next remediation.
- **Pass 5 — positive control** (`...-loop-pass5.md`, audit-v4, builder **68.6 → 72.5**): melanoblast Eden colonisation added as a competing model, scored with a Fable-corrected distance (unfitted statistics only) over 5 seeds at matched coverage. Eden **2.03 ± 0.70** beats RD growing-het 2.75 ± 0.45 and RD static 4.95 on the discriminating enclave statistics (roughness 9.0 vs 5.3, cow 11.9; CV 0.93 vs 0.38, cow 0.99). Animation is now 3 panels: cow | RD coat | melanoblast colonisation. Next: R2 more specimens, R3 coat mask, R8 fresh independent re-score.
