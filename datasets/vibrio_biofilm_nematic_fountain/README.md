# Dataset: 3D Rod-Cell Nematic Verticalization & Fountain Flow in Growing *Vibrio cholerae* Biofilms (`vibrio_biofilm_nematic_fountain`)

## Primary Literature & DOIs

1. **Beroz F, Yan J, Meir Y, Sabass B, Stone HA, Bassler BL, Wingreen NS (2018)**.
   *Verticalization of bacterial biofilms.* **Nature Physics** 14(9):954–960.
   DOI: [`10.1038/s41567-018-0170-4`](https://doi.org/10.1038/s41567-018-0170-4)
2. **Hartmann R, Singh PK, Pearce P, Mok R, Song B, Díaz-Pascual F, Dunkel J, Drescher K (2019)**.
   *Emergence of three-dimensional order and structure in growing biofilms.* **Nature Physics** 15(3):251–256.
   DOI: [`10.1038/s41567-018-0356-9`](https://doi.org/10.1038/s41567-018-0356-9)
3. **Yan J, Sharo AG, Stone HA, Wingreen NS, Bassler BL (2016)**.
   *Vibrio cholerae biofilm growth program and architecture revealed by single-cell live imaging.* **Proceedings of the National Academy of Sciences** 113(36):E5337–E5343.
   DOI: [`10.1073/pnas.1611494113`](https://doi.org/10.1073/pnas.1611494113)
4. **Drescher K, Dunkel J, Nadell CD, van Teeffelen S, Grin I, Wingreen NS, Stone HA, Bassler BL (2016)**.
   *Architectural transitions in Vibrio cholerae biofilms at single-cell resolution.* **Proceedings of the National Academy of Sciences** 113(14):E2066–E2072.
   DOI: [`10.1073/pnas.1601702113`](https://doi.org/10.1073/pnas.1601702113)

## Governing Mathematical & Biomechanical Framework

During early *Vibrio cholerae* biofilm expansion on a solid surface, rod-shaped cells of length $L$ and diameter $d$ proliferate at growth rate $g$ against surface adhesion and RbmA/Bap1/RbmC matrix friction $\zeta$.

1. **Parabolic Radial Compression Stress Profile (`Beroz et al. 2018 Nature Physics Eq. 1`):**
   $$\sigma_{rr}(r) = \frac{\zeta g}{4 h}(R^2 - r^2) = \kappa (R^2 - r^2), \qquad \kappa = \frac{\zeta g}{4 h},$$
   which reaches its global maximum $\sigma_{rr}(0) = \kappa R^2$ at the colony center ($r = 0$) and vanishes at the expanding margin ($r = R$).
2. **Hertzian Rod-on-Substrate Peeling Threshold & Inverse Cell-Length Scaling (`Beroz et al. 2018 Figs. 2–3`):**
   $$\sigma_c(L) = \sigma_{c,0}\left(\frac{L_0}{L}\right), \qquad R_c(L) = \sqrt{\frac{\sigma_c(L)}{\kappa}} = R_{c,0}\sqrt{\frac{L_0}{L}}.$$
   Longer cells (e.g. cephalexin-treated filaments) verticalize at smaller critical stress $\sigma_c(L)$ and smaller colony radius $R_c(L)$, whereas shorter cells (e.g. A22-treated rods) require larger colony radii to verticalize.
3. **Propagating Verticalization Front & 3D Nematic Fountain Flow (`Beroz et al. 2018 Fig. 4; Hartmann et al. 2019`):**
   For $R(t) > R_c(L)$, cells inside the central core $r \le r_{\text{vert}}(R) = \sqrt{R^2 - R_c(L)^2}$ buckle out of the substrate plane ($\theta \to 0$, vertical nematic order parameter $S_z = \langle \frac{3\cos^2\theta - 1}{2}\rangle \to +1$), extruding daughter cells upward into a 3D fountain-like dome surrounded by a peripheral horizontal monolayer rim ($S_z = -0.5$). Loss of RbmA cohesion ($\Delta rbmA$) or Bap1/RbmC surface adhesion ($\Delta bap1 \Delta rbmC$) lowers $\kappa = \zeta g / (4h)$ below threshold, leaving the colony as a flat unverticalized monolayer.

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 095.
- `paper_citations.bib` — BibTeX citations with DOIs for Beroz et al. (2018), Hartmann et al. (2019), Yan et al. (2016), and Drescher et al. (2016).
- `vibrio_biofilm_nematic_fountain_reference.json` — Quantitative reference tables (7-point A22/WT/cephalexin cell-length sweep $L \in [1.4, 4.5]\,\mu\text{m}$, 8-point radial expansion trajectory $R \in [6.0, 25.0]\,\mu\text{m}$, 5 genetic/pharmacological controls including $\Delta rbmA$ and $\Delta bap1\Delta rbmC$, and SHA-256 digests).
- `vibrio_cholerae_biofilm_specimen.jpg` — Public Domain (CDC PHIL / Wikimedia Commons) micrograph of surface-adhered rod-shaped *Vibrio cholerae* bacteria (`File:Vibrio cholerae 01.jpg`, 1580×1064 px).
