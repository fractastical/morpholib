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

## Pass 3 additions (growing-lamina canalization solver)

- `growing_lamina_reference.json` (SHA-256 `a51f3d788874ecb60eac26774976b3b081ebf6205887920fda1460901de23f2e`): solver parameters and reference benchmarks for the Pass 3 `simulate_growing_lamina_venation()` / `simulate_sachs_wound_bypass_dynamic()` solvers — Mitchison (1980) facilitated-diffusion canalization with soft-saturating feedback $\dot D = \alpha|\phi|^2/(1+\phi^2/\phi_s^2) + \beta - \gamma D$ on a hexagonal cell lattice that divides $12\times11 \to 24\times22 \to 48\times44$ (75 → 312 → 1281 lamina cells). Every benchmark states whether it is literature-derived, specimen-derived, or a model-internal check; the file also lists the known limitations (hex lattice not FEM; facilitated diffusion not polar PIN1; **no loop closure, $\beta_1 = 0$**, so areole subdivision is not reproduced; lattice-quantised strand angles).
- `paper_citations_pass3.bib` (SHA-256 `b3a22b5ca5439009362b1dca63112b3a28dba062a851e1c6e0baa09767dec635`): five additional references used by the Pass 3 measurements:
  6. **Feugier, F. G., & Iwasa, Y. (2006).** How canalization can make loops: A new model of reticulated leaf vascular pattern formation. *J. Theor. Biol.*, 243(2), 235–244. DOI: [`10.1016/j.jtbi.2006.05.022`](https://doi.org/10.1016/j.jtbi.2006.05.022)
  7. **Scarpella, E., Marcos, D., Friml, J., & Berleth, T. (2006).** Control of leaf vascular patterning by polar auxin transport. *Genes & Development*, 20(8), 1015–1027. DOI: [`10.1101/gad.1402406`](https://doi.org/10.1101/gad.1402406)
  8. **Kang, J., & Dengler, N. (2004).** Vein pattern development in adult leaves of *Arabidopsis thaliana*. *Int. J. Plant Sci.*, 165(2), 231–242. DOI: [`10.1086/382794`](https://doi.org/10.1086/382794)
  9. **Runions, A., Fuhrer, M., Lane, B., Federl, P., Rolland-Lagan, A.-G., & Prusinkiewicz, P. (2005).** Modeling and visualization of leaf venation patterns. *ACM Trans. Graph.*, 24(3), 702–711. DOI: [`10.1145/1073204.1073251`](https://doi.org/10.1145/1073204.1073251)
  10. **Sack, L., & Scoffoni, C. (2013).** Leaf venation: structure, function, development, evolution, ecology and applications in the past, present and future. *New Phytologist*, 198(4), 983–1000. DOI: [`10.1111/nph.12253`](https://doi.org/10.1111/nph.12253)

The Pass 1–2 files (`paper_citations.bib`, `auxin_canalization_leaf_veins_reference.json`, `scout_manifest.json`, specimen) are byte-identical to their pinned SHA-256 values; all six files are verified at runtime by `verify_dataset_integrity()`.
