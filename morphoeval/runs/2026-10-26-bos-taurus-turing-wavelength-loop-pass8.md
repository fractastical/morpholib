# Loop Pass 8: Coat Mask, Scale-Matched Statistics, Significance

**Remediates:** Fable round-2 findings (ROI contamination, resolution-dependent roughness, aspect bias, re-drawn heterogeneity, untested ranking).
**Changes:** `coat_segment()` — coat domain from HSV saturation (achromatic coat vs coloured background; dark-fur clause, largest component, hole fill), pigment by Otsu inside the coat; `scale_pattern()` — every mask (cow and sims) resampled to the 160×120 lattice before morphometrics; diameters normalised by √(domain area); γ-heterogeneity drawn once at final size and rescaled per stage (tissue-attached; no-dilution limitation stated); 12 seeds; paired Wilcoxon + bootstrap CI across specimens.

| Held-out (9 cows, coat-mask, scale-matched) | distance v2 |
|---|---:|
| RD static γ=1 | 6.46 |
| RD growing-het | **3.29** |
| Melanoblast Eden | 3.43 |

Eden wins **4/9** (was 7/9), mean diff **+0.14** (95% CI −0.80…1.14), Wilcoxon **p = 0.91**. Cow enclave roughness at lattice scale: 7.3 (cow 01) / median 8.2 — between RD (~5) and Eden (~9).

**Conclusion (revised):** at matched measurement scale the two mechanisms are **indistinguishable** on patch statistics; the Pass 5–6 Eden advantage was substantially the resolution artefact the Fable audit predicted. What survives every pass: static adult-scale Turing is wrong (6.5 vs ~3.3), and growing/clonal processes with body-scale territories are required. The full animation now uses coat masks and its closing chart states the null result.
**Audit:** v6, builder **73.8 → 74.6** (method quality up, favoured model gone — the rubric scores the package, not the hypothesis).
**Next:** R9 migration-wave Eden variant + citations · R10 push + nanopubs · R11 independent re-score before quoting any Eden-vs-RD claim.
