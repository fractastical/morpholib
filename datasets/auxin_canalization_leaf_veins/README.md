# Auxin Canalization & Leaf Venation Networks Dataset (`datasets/auxin_canalization_leaf_veins/`)

Reference dataset and empirical cleared dicot leaf venation specimen for **MorphoEval Slot 034 (`mitchison-canalization-auxin-flux`)**, implemented in `1997-mitchison-canalization-auxin-flux.py` and formally verified in `morphoeval/lean/loops/Loop_034_auxin_canalization_leaf_veins.lean`.

## Primary Literature & DOIs

1. **Mitchison, G. J. (1980).** A model for vein formation in higher plants. *Proceedings of the Royal Society of London. Series B. Biological Sciences*, 207(1166), 79–109. DOI: [`10.1098/rspb.1980.0082`](https://doi.org/10.1098/rspb.1980.0082)
2. **Mitchison, G. J. (1981).** The polar transport of auxin and vein pattern in plants. *Philosophical Transactions of the Royal Society of London. B, Biological Sciences*, 295(1078), 461–471. DOI: [`10.1098/rstb.1981.0081`](https://doi.org/10.1098/rstb.1981.0081)
3. **Feugier, F. G., Mochizuki, A., & Iwasa, Y. (2005).** Self-organization of vascular systems in plant leaves: Inter-dependent dynamics of auxin flux and carrier proteins. *Journal of Theoretical Biology*, 236(3), 366–375. DOI: [`10.1016/j.jtbi.2005.03.017`](https://doi.org/10.1016/j.jtbi.2005.03.017)
4. **Sachs, T. (1981).** The Control of the Patterned Differentiation of Vascular Tissues. *Advances in Botanical Research*, 9, 151–262. DOI: [`10.1016/S0065-2296(08)60351-1`](https://doi.org/10.1016/S0065-2296(08)60351-1)
5. **Rolland-Lagan, A.-G., & Prusinkiewicz, P. (2005).** Leaf venation: structure and mechanisms. *The Plant Journal*, 44(5), 854–865. DOI: [`10.1111/j.1365-313X.2005.02581.x`](https://doi.org/10.1111/j.1365-313X.2005.02581.x)

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 034.
- `auxin_canalization_leaf_veins_reference.json`: Reference parameters and quantitative benchmarks for feedback exponent $p \in [0.5, 2.5]$ symmetry breaking, canalized vein concentration drainage ($S_{\text{vein}} < S_{\text{parenchyma}}$), Sachs transverse stem incision bypass regeneration, and Feugier et al. (2005) open-tree vs closed-areole venation topology.
- `paper_citations.bib`: BibTeX citations with verified DOIs.
- `leaf_venation_canalization_specimen.jpg`: Cleared dicot leaf venation skeleton (`File:Vein skeleton of a leaf (de-ghosted).jpg`, Wikimedia Commons, CC BY-SA 3.0, 960×559 px, SHA-256: `7f67774708f8e952fe3845452f8288b7fdcb34cfc29e0fcb08be77e66d08492b`).
