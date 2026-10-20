# PIN1–Auxin–CUC2–RCO Feedback Loop & Leaf Margin Serration vs Compound Leaflet Dissection Dataset (`datasets/allometric_compound_leaf/`)

Reference dataset and empirical *Cardamine hirsuta* compound leaf botanical specimen for **MorphoEval Slot 037 (`przemyslaw-allometric-compound-leaf`)**, implemented in `2011-przemyslaw-allometric-compound-leaf.py` and formally verified in `morphoeval/lean/loops/Loop_037_allometric_compound_leaf.lean`.

## Primary Literature & DOIs

1. **Bilsborough, G. D., Runions, A., Barkoulas, M., Jenkins, H. W., Hasson, A., Galinha, C., Laufs, P., Hay, A., Prusinkiewicz, P., & Tsiantis, M. (2011).** Model for the regulation of *Arabidopsis thaliana* leaf margin development. *Proceedings of the National Academy of Sciences*, 108(8), 3424–3429. DOI: [`10.1073/pnas.1015162108`](https://doi.org/10.1073/pnas.1015162108)
2. **Barkoulas, M., Hay, A., Kougioumoutzi, E., & Tsiantis, M. (2008).** A developmental framework for dissected leaf formation in the *Arabidopsis* relative *Cardamine hirsuta*. *Nature Genetics*, 40(9), 1136–1141. DOI: [`10.1038/ng.189`](https://doi.org/10.1038/ng.189)
3. **Vlad, D., Kierzkowski, D., Rast, M. I., Vuolo, F., Dello Ioio, R., Galinha, C., ... & Tsiantis, M. (2014).** Leaf shape evolution through duplication, regulatory diversification, and loss of a homeobox gene. *Science*, 343(6172), 780–783. DOI: [`10.1126/science.1248384`](https://doi.org/10.1126/science.1248384)
4. **Kierzkowski, D., Runions, A., Vuolo, F., Strauss, S., Lymbouridou, R., Routier-Kierzkowska, A.-L., ... & Tsiantis, M. (2019).** A growth-based framework for leaf shape development and diversity. *Cell*, 177(6), 1405–1418.e17. DOI: [`10.1016/j.cell.2019.05.011`](https://doi.org/10.1016/j.cell.2019.05.011)

### Added in Pass 3 (2026-10-20) — cellular lamina mesh, vein canalization, microtubule anisotropy

5. **Mitchison, G. J. (1980).** A model for vein formation in higher plants. *Proceedings of the Royal Society of London B*, 207(1166), 79–109. DOI: [`10.1098/rspb.1980.0015`](https://doi.org/10.1098/rspb.1980.0015) — flux-feedback conductance update `D' = D + Δt(k q² − μ D)` used to differentiate lateral veins (Lean theorem `mitchison_1980_canalization_conductance_feedback`).
6. **Scarpella, E., Marcos, D., Friml, J., & Berleth, T. (2006).** Control of leaf vascular patterning by polar auxin transport. *Genes & Development*, 20(8), 1015–1027. DOI: [`10.1101/gad.1402406`](https://doi.org/10.1101/gad.1402406) — marginal auxin convergence points initiate vein strands that connect to the midvein (reference for the "all lateral veins drained to midvein" measurement).
7. **Hamant, O., Heisler, M. G., Jönsson, H., et al. (2008).** Developmental patterning by mechanical signals in *Arabidopsis*. *Science*, 322(5908), 1650–1655. DOI: [`10.1126/science.1165594`](https://doi.org/10.1126/science.1165594) — cortical microtubule alignment sets anisotropic cell-wall reinforcement (Lean theorem `hamant_2008_microtubule_anisotropic_growth_partition`).
8. **Sampathkumar, A., Krupinski, P., Wightman, R., et al. (2014).** Subcellular and supracellular mechanical stress prescribes cytoskeleton behavior in *Arabidopsis* cotyledon pavement cells. *eLife*, 3, e01967. DOI: [`10.7554/eLife.01967`](https://doi.org/10.7554/eLife.01967) — stress-feedback microtubule reorientation (implemented as the lamina stress tensor low-pass, `a_lamina`, disabled by default in Pass 3 for stability).

All eight DOIs were resolved against the Crossref API on 2026-10-20 (title/journal/volume/pages matched).

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 037 (`46ea1539febdb312ea029bbf5dd4c2b54d1689a2a88695b475a752189c293a7a`).
- `allometric_compound_leaf_reference.json`: Reference parameters and quantitative benchmarks for the 1D/2D growing leaf margin Auxin–PIN1–CUC2–RCO feedback loop, intercalary tooth/leaflet insertion at critical wavelength $\lambda_c$, 6-genotype morphospace sweep (`cuc2-1`, `cuc2-1D`, *Arabidopsis* WT Col-0, `mir164a-4`, *Cardamine* `rco`, and *Cardamine hirsuta* WT compound leaf), and specimen morphometrics (`fa4f9b265dd83379ff238237f84d4600147d010812ccb6f9deff72a0bf9b047c`).
- `paper_citations.bib`: BibTeX citations with verified DOIs (`8323f1f35f2613d6de6004d9f600b9f7c0f96b4edf8d807b579418db7c861c1d`).
- `cardamine_hirsuta_compound_leaf_specimen.jpg`: Flat-mounted studio photograph of *Cardamine hirsuta* (`File:Cardamine hirsuta.jpg`, Wikimedia Commons, Public domain, 617×992 px, SHA-256: `1c268fecb3660b310d99f688a13a679c347aabc4e2522eace472f8c985bf8510`).

## Pass 3 note on the reference benchmarks

The `dissection_index_ID_ref` values in `allometric_compound_leaf_reference.json` are unchanged. In Pass 2 the simulation *reproduced* them by construction (hard-coded mutant values and a fitted closed-form target), which made the reported 6-genotype Pearson r = 0.998 circular. In Pass 3 all simulated `I_D` values are emergent from the integrated margin contour (`integrate_margin_outgrowth` + `dissection_index_radial`) or from the growing cellular spring-lattice lamina (`GrowingLaminaSpringMesh`); one gain (`kappa_n = 0.50`) is calibrated on the *Arabidopsis* WT datum (0.228). The *Cardamine hirsuta* leaflet-level value (0.785) is **not** reproduced by the equal-growth-duration model (emergent 0.254; sweep ceiling 0.427) — this is recorded as an open remediation item rather than fitted away.
