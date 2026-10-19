# Bressloff V1 Form Constants & Driven Neural-Field Fixtures

This directory holds reference reports and SHA-256-pinned fixtures vendored from [`MesmerPrism/bressloff-v1-form-constants`](https://github.com/MesmerPrism/bressloff-v1-form-constants) at commit `344863a1bd9e8a8e8c9cb20aced2897f3ffb2c4c` (MIT License; derived from [`karacsm/V1-sim`](https://github.com/karacsm/V1-sim) by Márton Karácsony, extended by Till Holzapfel).

![Bressloff V1 form constants: cortical planforms and retinal log-polar pullbacks across Klüver's four families](../../docs/media/bressloff_v1_form_constants.gif)

## Vendored Fixtures

| File | SHA-256 | Upstream Path | Description |
| :--- | :--- | :--- | :--- |
| `paper-calibration.json` | `af12b2c1aa3e38faf984166b3aa227a2aa064480a98279f14c004342ce2e7dd4` | `reports/paper-calibration.json` | Paper-preset calibration report (`bressloff-paper-calibration-v4`) covering Bressloff et al. (2001/2002) marginal-stability scans (`Fig 16` odd, `Fig 17` even), non-contoured single-map planforms (`Figs 29–30`), and contoured double-map visual-field planforms (`Figs 31–36`, `2002 Figs 5–7`) plus cubic amplitude-equation branch-selection diagnostics (`Fig 37–40`). |
| `rule-2011-floquet.json` | `5e349c7b80185de463e1ea8656a2fe91a83523c212875af4ce46d09e9dde2a8e` | `reports/rule-2011-floquet.json` | Rule, Stoffregen & Ermentrout (2011) homogeneous-orbit Floquet monodromy calibration surface (`rule-2011-floquet-calibration-v5`) with `1:1` (`+1`) stripe vs. `2:1` (`-1` period-doubling) hexagonal tongue boundary curves over forcing period $T$ (ms) and normalized wave number $\beta$. |
| `driven-neural-fields-registry.json` | `4f07b5175a48bcb50bad4998953505a3f92faf4c6f5514d2b4643c2a67c36b10` | `reports/driven-neural-fields-registry.json` | Public-safe implementation registry (`driven-neural-fields-registry-v1`) for driven neural-field examples (`mackay_localized_input`, `spatial_forcing_orthogonal_response`, `localized_time_periodic_input`) including the Gaussian-only negative control (`mackay_gaussian_kernel_negative_check`). |
| `horton_hoyt_1991_magnification.json` | `90d639ebb4601be59308699e128ac9b59574a9da6dbc47f0c7ad00be8d9e4a8f` | *(this repository; not from upstream)* | Empirical anchor: human V1 linear cortical magnification $M(E) = 17.3/(E + 0.75)$ mm/deg from Horton & Hoyt (1991). The script compares it with the Bressloff map's $dx/dr = \alpha/(w_0 + \epsilon r)$ over 1–40° (RMS relative error 6.8%; map-implied $A = \alpha/\epsilon = 18.72$ mm/deg, $E_2 = w_0/\epsilon = 1.71°$ vs. 17.3 and 0.75). |

## Fixture caveats surfaced by the script

* `paper-calibration.json` carries its own paper-expectation `checks`. 10 of 11 pass; `fig40_odd_hex_bifurcation/selected-pattern` expects `triangle` but the fixture (and this script, which reproduces it to $2\times10^{-16}$) gives `honeycomb`. On the odd branch $\eta_{hex} = 0$, so $H_0$ (honeycomb) and $H_\pi$ (triangle) are degenerate at cubic order; the quintic selection of Bressloff et al. (2001, §5) is not implemented. The script reports this in the CSV (`fixture_paper_checks` column), the figure ledger, and the self-test — it is not folded into the parity verdict.
* `rule-2011-floquet.json` has `fit_objective.acceptance_status = outside-threshold-diagnostic` and `calibration_claim_allowed = false` (domain-normalised RMS $\beta$ error 0.154 against a 0.10 threshold; 4 failed acceptance checks). The Rule (2011) Fig. 8 panel is therefore labelled a diagnostic overlay, not a reproduction.

## Primary Literature

1. **Bressloff PC, Cowan JD, Golubitsky M, Thomas PJ, Wiener MC (2001)**. *Geometric visual hallucinations, Euclidean symmetry and the functional architecture of striate cortex*. **Philosophical Transactions of the Royal Society of London B** 356(1407):299–330. [`doi:10.1098/rstb.2000.0769`](https://doi.org/10.1098/rstb.2000.0769)
2. **Bressloff PC, Cowan JD, Golubitsky M, Thomas PJ, Wiener MC (2002)**. *What geometric visual hallucinations tell us about the visual cortex*. **Neural Computation** 14(3):473–491. [`doi:10.1162/089976602317250861`](https://doi.org/10.1162/089976602317250861)
3. **Ermentrout GB, Cowan JD (1979)**. *A mathematical theory of visual hallucination patterns*. **Biological Cybernetics** 34(3):137–150. [`doi:10.1007/BF00336965`](https://doi.org/10.1007/BF00336965)
4. **Rule M, Stoffregen M, Ermentrout B (2011)**. *A model for the origin and properties of flicker-induced geometric phosphenes*. **PLoS Computational Biology** 7(9):e1002158. [`doi:10.1371/journal.pcbi.1002158`](https://doi.org/10.1371/journal.pcbi.1002158) (PMID 21980269)
5. **Horton JC, Hoyt WF (1991)**. *The representation of the visual field in human striate cortex: a revision of the classic Holmes map*. **Archives of Ophthalmology** 109(6):816–824. [`doi:10.1001/archopht.1991.01080060080030`](https://doi.org/10.1001/archopht.1991.01080060080030) (PMID 2043069)

`paper_citations.bib` holds the same five references.

## License & Notice

Upstream code and JSON reports from `MesmerPrism/bressloff-v1-form-constants` are distributed under the **MIT License**:

> Copyright (c) 2023 Márton Karácsony
> Copyright (c) 2026 Till Holzapfel
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
