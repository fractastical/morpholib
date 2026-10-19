# Leaf Venation Morphogenesis Reference Dataset (`datasets/venation/`)

Primary mathematical specifications, empirical leaf-network morphometric tables, and specimen photographs for `2005-runions-leaf-venation.py`.

## Primary Sources

1. **Runions, Fuhrer, Lane, Federl, Rolland-Lagan & Prusinkiewicz (2005)**, *"Modeling and visualization of leaf venation patterns"*, ACM Transactions on Graphics 24(3):702–711 (`10.1145/1073204.1073251`).
2. **Murray (1926)**, *"The physiological principle of minimum work. I. The vascular system and the cost of blood volume"*, PNAS 12(3):207–214 (`10.1073/pnas.12.3.207`).
3. **Price, Knox & Brodribb (2013)**, *"The Influence of Branch Order on Optimal Leaf Vein Geometries: Murray's Law and Area Preserving Branching"*, PLoS ONE 8(12):e85420 (`10.1371/journal.pone.0085420`) — 349 cleared leaf vein networks comprising 1,514,771 vein junctions across Horton-Strahler orders 1–8.
4. **Katifori & Magnasco (2012)**, *"Quantifying Loopy Network Architectures"*, PLoS ONE 7(6):e37994 (`10.1371/journal.pone.0037994`).
5. **Ronellenfitsch, Lasser, Daly & Katifori (2015)**, *"Topological Phenotypes Constitute a New Dimension in the Phenotypic Space of Leaf Venation Networks"*, PLoS Computational Biology 11(12):e1004680 (`10.1371/journal.pcbi.1004680`).

## Files & SHA-256 Provenance

- `venation_reference.json` — Complete Table 1 parameter registry from Runions et al. (2005), Horton-Strahler order-by-order empirical scaling data from Price et al. (2013), and topological loop nesting benchmarks from Katifori & Magnasco (2012) / Ronellenfitsch et al. (2015).
- `ginkgo_open_venation_specimen.jpg` — *Ginkgo biloba* leaf specimen photograph (`File:Ginkgo biloba 01 SR 87-36-02 A.jpg`, CC BY-SA 3.0, SHA-256 `5a620943881fe69889ff21c48f174c4b9a966369edd575daeb93dc8f82085d4b`) showing open dichotomous venation without anastomoses ($L = E - V + 1 = 0$).
- `reticulate_vein_skeleton_specimen.jpg` — Dicotyledonous angiosperm leaf vein skeleton (`File:Vein skeleton of a leaf (de-ghosted).jpg`, CC BY-SA 3.0, SHA-256 `6d06d3e545bd585bad4530aaa1034353e4cf7ec24815f054df903bb576e55aa0`) showing hierarchically nested primary, secondary, and tertiary reticulate areole loops ($L = E - V + 1 > 0$).
- `paper_citations.bib` — BibTeX citations with DOIs.
