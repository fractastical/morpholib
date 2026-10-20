# Shoot Apical Meristem Shell Stress Anisotropy, Stress-vs-Strain Cue Discrimination & Katanin Feedback Dataset (`datasets/microtubule_stress_meristem/`)

Reference dataset and empirical *Arabidopsis thaliana* shoot apical meristem longitudinal section micrograph for **MorphoEval Slot 036 (`hamant-microtubule-stress-meristem`)**, implemented in `2008-hamant-microtubule-stress-meristem.py` and formally verified in `morphoeval/lean/loops/Loop_036_microtubule_stress_meristem.lean`.

## Primary Literature & DOIs

1. **Bozorg, B., Krupinski, P., & Jönsson, H. (2014).** Stress and Strain Provide Positional and Directional Cues in Development. *PLoS Computational Biology*, 10(1), e1003410. DOI: [`10.1371/journal.pcbi.1003410`](https://doi.org/10.1371/journal.pcbi.1003410)
2. **Sampathkumar, A., Krupinski, P., Wightman, R., Milani, P., Berquand, A., Boudaoud, A., Hamant, O., Jönsson, H., & Meyerowitz, E. M. (2014).** Subcellular and supracellular mechanical stress prescribes cytoskeleton behavior in *Arabidopsis* cotyledon pavement cells. *eLife*, 3, e01967. DOI: [`10.7554/eLife.01967`](https://doi.org/10.7554/eLife.01967)
3. **Hamant, O., Heisler, M. G., Jönsson, H., Krupinski, P., Uyttewaal, M., Boudaoud, A., Meyerowitz, E. M., & Traas, J. (2008).** Developmental Patterning by Mechanical Signals in *Arabidopsis*. *Science*, 322(5908), 1650–1655. DOI: [`10.1126/science.1165594`](https://doi.org/10.1126/science.1165594)
4. **Uyttewaal, M., Burian, A., Alim, K., Landrein, B., Borowska-Wykręt, D., Dedieu, A., Peaucelle, A., Ludynia, M., Traas, J., Boudaoud, A., Kwiatkowska, D., & Hamant, O. (2012).** Mechanical Stress Acts via Katanin to Amplify Differences in Growth Rate between Adjacent Cells in *Arabidopsis*. *Cell*, 149(2), 439–451. DOI: [`10.1016/j.cell.2012.02.048`](https://doi.org/10.1016/j.cell.2012.02.048)

### Added in Pass 3 (audit `2026-10-19-microtubule-stress-meristem-audit-v3`)

5. **Heisler, M. G., Hamant, O., Krupinski, P., Uyttewaal, M., Ohno, C., Jönsson, H., Traas, J., & Meyerowitz, E. M. (2010).** Alignment between PIN1 Polarity and Microtubule Orientation in the Shoot Apical Meristem Reveals a Tight Coupling between Morphogenesis and Auxin Transport. *PLoS Biology*, 8(10), e1000516. DOI: [`10.1371/journal.pbio.1000516`](https://doi.org/10.1371/journal.pbio.1000516)
6. **Louveaux, M., Julien, J.-D., Mirabet, V., Boudaoud, A., & Hamant, O. (2016).** Cell division plane orientation based on tensile stress in *Arabidopsis thaliana*. *PNAS*, 113(30), E4294–E4303. DOI: [`10.1073/pnas.1600677113`](https://doi.org/10.1073/pnas.1600677113)
7. **Hamant, O., & Traas, J. (2010).** The mechanics behind plant development. *New Phytologist*, 185(2), 369–385. DOI: [`10.1111/j.1469-8137.2009.03100.x`](https://doi.org/10.1111/j.1469-8137.2009.03100.x) — cited for the absence of T1 neighbour exchange in cemented plant tissue.

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 036 (`e3cd62cc5263f317c433e043c2100f6f7f237741722d19c17d4a9ff4c2fbe999`).
- `microtubule_stress_meristem_reference.json`: Reference parameters and quantitative benchmarks for the 4 canonical SAM zones (Central Dome $K > 0$, Peripheral Flank, Organ-Meristem Saddle Boundary $K < 0$, and Cylindrical Stem $\sigma_\theta / \sigma_z = 2$), Bozorg et al. (2014) stress-vs-strain cue comparison, laser ablation ring reorientation, lateral compression clamp, and `bot1`/`ktn1` + Oryzalin perturbations (`3e04a6dded52881911fd9eedb0387c4a184753d1f78fb51754168201712b5f5e`).
- `paper_citations.bib`: BibTeX citations with verified DOIs; Pass 3 appended Heisler 2010, Louveaux 2016 and Hamant & Traas 2010 (SHA-256 now `9009d0cffd73437a3aadb9bea656ef6b5bb5a08ea2de78bf54420a2e58c80d19`; the Pass 2 hash was `9355fa7c…`).
- `arabidopsis_meristem_boundary_specimen.jpg`: Longitudinal histology section of *Arabidopsis thaliana* shoot apical meristem (`File:Arabidopsis round SAM.jpg`, Wikimedia Commons, CC BY-SA 4.0, 1200×800 px, SHA-256: `3a8b54b71f7c780cf7f68a2ff383f346b511ff2aaea015aebff4d1150328c164`). Unchanged since Pass 1.
- `pin1_cmt_coalignment_reference.json` (**Pass 3**): Heisler (2010) / Louveaux (2016) reference thresholds and the full parameter table of the L1 cellular vertex-mesh model (`4d5ce4132c9e546dd3510dee6ea53b925956b573e893c89818dca639cd4fd2a1`). The `reference_value`s flagged `ordinal_threshold_from_qualitative_report` (0.75 = alignment within 30°) are derived from the papers' qualitative statements, **not** digitised figure values; `random_control_value` gives the null (0.5) for each alignment index.

## Pass 3 vertex-mesh scope (honest statement)

The vertex mesh in `2008-hamant-microtubule-stress-meristem.py` §3b is a **2D apical-view patch** of the L1 layer (61 cells, 4 hexagonal rings, Voronoi-jittered). The zone shell stress tensor $(\sigma_1, \sigma_2)$ enters only as an affine boundary pre-strain $\epsilon_i = \sigma_i / E_\text{shell}$; curvature is not represented on a 3D surface. T1 flips are implemented and unit-tested but disabled by default (plant walls are cemented). The Bozorg stress-vs-strain discrimination is *not* reproduced by the mesh (both cues give circumferential CMTs around a wound in this tension-dominated sheet); that result is reported, not hidden.
