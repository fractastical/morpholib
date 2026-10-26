# Loop Pass 3 (Remediation): Heterogeneity + Boundary Roughness

**Remediates:** v2/R6 — simulated enclaves are smooth ovals; the cow's white patches are jagged.
**Changes:** (1) `boundary_roughness_mean` (perimeter/√area per patch; a disc = 3.54) added to `patch_stats()` for both phases and to the fit objective; (2) two-scale spatially varying γ (tissue heterogeneity) in the growing simulation, amplitude grid-tested at 0 / 0.45 / 0.8.
**Result:** best = growing-het, γ=0.25, heterogeneity 0.8, q=0.75 — coverage 0.75 (cow 0.77), largest share 0.997 (cow 0.992), diameter CV 0.90 (cow 1.25), enclaves 15 vs 12 (coverage 0.25 vs 0.23). Distance **1.43 → 1.21** on the roughness-aware objective. Heterogeneity helps areas/counts but *not* boundary geometry: enclave roughness 5.0 vs cow 11.9 at every amplitude — diffusion smooths boundaries regardless of the γ-map. That residual is itself evidence for the competing clonal melanoblast-growth account (R5), whose patches are jagged by construction.
**Audit:** `2026-10-26-bos-taurus-turing-wavelength-audit-v3` — **66.0 → 68.6** (specimen-match 45 → 52, morphometric-validation 68 → 72).
**Animation:** regenerated from the fitted configuration (γ=0.25, het=0.8, q=0.75): embryonic pattern → two growth stages → adult coat, beside the cow mask.
**Still open:** R1 second assessor · R2 more specimens (breaks the fit-on-n=1 circle) · R3 body-mask ROI · R5 melanoblast positive control · R7 CLI/pins.
