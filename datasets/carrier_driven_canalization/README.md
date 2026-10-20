# PIN1 Carrier-Driven Canalization & Hexagonal Leaf Venation Loops Dataset (`datasets/carrier_driven_canalization/`)

Reference dataset and empirical cleared leaf tip venation skeleton specimen for **MorphoEval Slot 035 (`feugier-carrier-driven-canalization`)**, implemented in `2005-feugier-carrier-driven-canalization.py` and formally verified in `morphoeval/lean/loops/Loop_035_carrier_driven_canalization.lean`.

## Pass 3 correction of record (2026-10-20)

Passes 1–2 cited Feugier, Mochizuki & Iwasa (2005) as *J. Theor. Biol.* **236(3)**:313–325. The DOI `10.1016/j.jtbi.2005.03.017` resolves (Crossref + PubMed `15899502`) to *J. Theor. Biol.* **236(4)**:366–375, title "Self-organization of **the** vascular system in plant leaves". All vendored files are corrected and the script's SHA-256 pins updated. The paper's abstract also states that *linear or decelerating* carrier-response functions form **no** branching pattern (only accelerating ones do); the Pass 2 claim that a saturating Hill response is the paper's loop-forming "Model B" is therefore withdrawn, and loop closure is attributed to Scarpella et al. (2004, 2006) procambial commitment / bipolar convergence-point cells instead.

## Primary Literature & DOIs

1. **Feugier, F. G., Mochizuki, A., & Iwasa, Y. (2005).** Self-organization of the vascular system in plant leaves: Inter-dependent dynamics of auxin flux and carrier proteins. *Journal of Theoretical Biology*, 236(4), 366–375. DOI: [`10.1016/j.jtbi.2005.03.017`](https://doi.org/10.1016/j.jtbi.2005.03.017), PMID 15899502.
2. **Scarpella, E., Marcos, D., Friml, J., & Berleth, T. (2006).** Control of leaf vascular patterning by polar auxin transport. *Genes & Development*, 20(8), 1015–1027. DOI: [`10.1101/gad.1402406`](https://doi.org/10.1101/gad.1402406)
3. **Scarpella, E., Francis, P., & Berleth, T. (2004).** Stage-specific markers define early steps of procambium development in *Arabidopsis* leaves and correlate termination of vein formation with mesophyll differentiation. *Development*, 131(14), 3445–3455. DOI: [`10.1242/dev.01182`](https://doi.org/10.1242/dev.01182), PMID 15226260. *(added Pass 3)*
4. **Rolland-Lagan, A.-G., & Prusinkiewicz, P. (2005).** Reviewing models of auxin canalization in the context of leaf vein pattern formation in *Arabidopsis*. *The Plant Journal*, 44(5), 854–865. DOI: [`10.1111/j.1365-313X.2005.02581.x`](https://doi.org/10.1111/j.1365-313X.2005.02581.x)
5. **Stoma, S., Lucas, M., Chopard, J., Schaedel, M., Traas, J., & Godin, C. (2008).** Flux-based transport enhancement as a plausible unifying mechanism for auxin transport in meristem development. *PLoS Computational Biology*, 4(10), e1000207. DOI: [`10.1371/journal.pcbi.1000207`](https://doi.org/10.1371/journal.pcbi.1000207)
6. **Kang, J., & Dengler, N. (2004).** Vein pattern development in adult leaves of *Arabidopsis thaliana*. *International Journal of Plant Sciences*, 165(2), 231–242. DOI: [`10.1086/382794`](https://doi.org/10.1086/382794) *(added Pass 3)*
7. **Lockhart, J. A. (1965).** An analysis of irreversible plant cell elongation. *Journal of Theoretical Biology*, 8(2), 264–275. DOI: [`10.1016/0022-5193(65)90077-9`](https://doi.org/10.1016/0022-5193(65)90077-9) *(added Pass 3; biomechanical blade growth law)*
8. **Kuchen, E. E., et al. (2012).** Generation of leaf shape through early patterns of growth and tissue polarity. *Science*, 335(6072), 1092–1096. DOI: [`10.1126/science.1214678`](https://doi.org/10.1126/science.1214678) *(added Pass 3; early narrow-to-broad lamina allometry)*

## Files & Integrity

SHA-256 digests are pinned in `EXPECTED_SHA256` of `2005-feugier-carrier-driven-canalization.py` and verified fail-closed at every run (`verify_dataset_integrity`).

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 035 (unchanged since Pass 1).
- `carrier_driven_canalization_reference.json`: Pass 1–2 reference parameters and quantitative benchmarks for hexagonal cell-level auxin dynamics, conserved intracellular PIN1 carrier pool dynamics ($P_{i,\text{cyt}} + \sum_{j=1}^6 P_{ij} = P_{\text{total}}$), the 6-regime single-cell allocation sweep, the templated-scaffold lattice benchmarks, and the `gnom`/`pin1` exocytosis ablation control. Pass 3 corrected the Feugier (2005) citation string only.
- `growing_blade_reference.json` *(new, Pass 3)*: Lockhart (1965) blade-growth law and parameters, the four-stage developmental schedule (2.0, 3.5, 5.0, 7.0 DAG), emergent canalization parameters for the growing hexagonal blade mesh, and the expected outcomes per regime (open tree β₁ = 0; committed reticulate β₁ ≥ 1 decreasing with τ_commit; decelerating Φ → no branching, vein fraction → 1; k₁ = 0 ablation → nothing), with the bibliographic correction record.
- `paper_citations.bib`: BibTeX citations with verified DOIs (8 entries; Feugier 2005 corrected).
- `leaf_tip_venation_loops_specimen.jpg`: Cleared dicot leaf tip venation skeleton (`File:Leaf tip skeleton, detail from file by Stefanie Leuker.jpg`, Wikimedia Commons, CC BY-SA 4.0, 539×351 px, SHA-256: `e54c8add6cf7c9166f34e9efee7aefae8ef2ff94d1657a25b100636290b42653`). Unchanged; never to be removed.
