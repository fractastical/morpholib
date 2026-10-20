# Shoot Apical Meristem Shell Stress Anisotropy, Stress-vs-Strain Cue Discrimination & Katanin Feedback Dataset (`datasets/microtubule_stress_meristem/`)

Reference dataset and empirical *Arabidopsis thaliana* shoot apical meristem longitudinal section micrograph for **MorphoEval Slot 036 (`hamant-microtubule-stress-meristem`)**, implemented in `2008-hamant-microtubule-stress-meristem.py` and formally verified in `morphoeval/lean/loops/Loop_036_microtubule_stress_meristem.lean`.

## Primary Literature & DOIs

1. **Bozorg, B., Krupinski, P., & Jönsson, H. (2014).** Stress and Strain Provide Positional and Directional Cues in Development. *PLoS Computational Biology*, 10(1), e1003410. DOI: [`10.1371/journal.pcbi.1003410`](https://doi.org/10.1371/journal.pcbi.1003410)
2. **Sampathkumar, A., Krupinski, P., Wightman, R., Milani, P., Berquand, A., Boudaoud, A., Hamant, O., Jönsson, H., & Meyerowitz, E. M. (2014).** Subcellular and supracellular mechanical stress prescribes cytoskeleton behavior in *Arabidopsis* cotyledon pavement cells. *eLife*, 3, e01967. DOI: [`10.7554/eLife.01967`](https://doi.org/10.7554/eLife.01967)
3. **Hamant, O., Heisler, M. G., Jönsson, H., Krupinski, P., Uyttewaal, M., Boudaoud, A., Meyerowitz, E. M., & Traas, J. (2008).** Developmental Patterning by Mechanical Signals in *Arabidopsis*. *Science*, 322(5908), 1650–1655. DOI: [`10.1126/science.1165594`](https://doi.org/10.1126/science.1165594)
4. **Uyttewaal, M., Burian, A., Alim, K., Landrein, B., Borowska-Wykręt, D., Dedieu, A., Peaucelle, A., Ludynia, M., Traas, J., Boudaoud, A., Kwiatkowska, D., & Hamant, O. (2012).** Mechanical Stress Acts via Katanin to Amplify Differences in Growth Rate between Adjacent Cells in *Arabidopsis*. *Cell*, 149(2), 439–451. DOI: [`10.1016/j.cell.2012.02.048`](https://doi.org/10.1016/j.cell.2012.02.048)

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 036 (`e3cd62cc5263f317c433e043c2100f6f7f237741722d19c17d4a9ff4c2fbe999`).
- `microtubule_stress_meristem_reference.json`: Reference parameters and quantitative benchmarks for the 4 canonical SAM zones (Central Dome $K > 0$, Peripheral Flank, Organ-Meristem Saddle Boundary $K < 0$, and Cylindrical Stem $\sigma_\theta / \sigma_z = 2$), Bozorg et al. (2014) stress-vs-strain cue comparison, laser ablation ring reorientation, lateral compression clamp, and `bot1`/`ktn1` + Oryzalin perturbations (`3e04a6dded52881911fd9eedb0387c4a184753d1f78fb51754168201712b5f5e`).
- `paper_citations.bib`: BibTeX citations with verified DOIs (`9355fa7c7b784918dc3e6606119796633d58b5ffb67f252c129e1568d173e28e`).
- `arabidopsis_meristem_boundary_specimen.jpg`: Longitudinal histology section of *Arabidopsis thaliana* shoot apical meristem (`File:Arabidopsis round SAM.jpg`, Wikimedia Commons, CC BY-SA 4.0, 1200×800 px, SHA-256: `3a8b54b71f7c780cf7f68a2ff383f346b511ff2aaea015aebff4d1150328c164`).
