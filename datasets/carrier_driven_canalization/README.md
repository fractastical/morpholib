# PIN1 Carrier-Driven Canalization & Hexagonal Leaf Venation Loops Dataset (`datasets/carrier_driven_canalization/`)

Reference dataset and empirical cleared leaf tip venation skeleton specimen for **MorphoEval Slot 035 (`feugier-carrier-driven-canalization`)**, implemented in `2005-feugier-carrier-driven-canalization.py` and formally verified in `morphoeval/lean/loops/Loop_035_carrier_driven_canalization.lean`.

## Primary Literature & DOIs

1. **Feugier, F. G., Mochizuki, A., & Iwasa, Y. (2005).** Self-organization of vascular systems in plant leaves: Inter-dependent dynamics of auxin flux and carrier proteins. *Journal of Theoretical Biology*, 236(3), 313–325. DOI: [`10.1016/j.jtbi.2005.03.017`](https://doi.org/10.1016/j.jtbi.2005.03.017)
2. **Scarpella, E., Marcos, D., Friml, J., & Berleth, T. (2006).** Control of leaf vascular patterning by polar auxin transport. *Genes & Development*, 20(8), 1015–1027. DOI: [`10.1101/gad.1402406`](https://doi.org/10.1101/gad.1402406)
3. **Rolland-Lagan, A.-G., & Prusinkiewicz, P. (2005).** Reviewing models of auxin canalization in the context of leaf vein pattern formation in *Arabidopsis*. *The Plant Journal*, 44(5), 854–865. DOI: [`10.1111/j.1365-313X.2005.02581.x`](https://doi.org/10.1111/j.1365-313X.2005.02581.x)
4. **Stoma, S., Lucas, M., Chopard, J., Schaedel, M., Traas, J., & Godin, C. (2008).** Flux-based transport enhancement as a plausible unifying mechanism for auxin transport in meristem development. *PLoS Computational Biology*, 4(10), e1000207. DOI: [`10.1371/journal.pcbi.1000207`](https://doi.org/10.1371/journal.pcbi.1000207)

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 035 (`1de2b4f4c53ac2348e9816c4ac584be15a268262e63d2b825874d1342e88ea78`).
- `carrier_driven_canalization_reference.json`: Reference parameters and quantitative benchmarks for hexagonal cell-level auxin dynamics, conserved intracellular PIN1 carrier pool dynamics ($P_{i,\text{cyt}} + \sum_{j=1}^6 P_{ij} = P_{\text{total}}$), winner-take-all open dendritic trees ($\beta_1 = 0$) vs saturating Hill allocation + Scarpella epidermal convergence closed reticulate loops ($\beta_1 > 0$) with bipolar bridge cells, and `gnom`/`pin1` exocytosis ablation control.
- `paper_citations.bib`: BibTeX citations with verified DOIs.
- `leaf_tip_venation_loops_specimen.jpg`: Cleared dicot leaf tip venation skeleton (`File:Leaf tip skeleton, detail from file by Stefanie Leuker.jpg`, Wikimedia Commons, CC BY-SA 4.0, 539×351 px, SHA-256: `e54c8add6cf7c9166f34e9efee7aefae8ef2ff94d1657a25b100636290b42653`).
