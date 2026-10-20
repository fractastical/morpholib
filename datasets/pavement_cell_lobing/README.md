# Arabidopsis Puzzle-Shaped Pavement Cell Lobing & Largest Empty Circle (LEC) Stress Reduction Dataset

Reference parameters, empirical scaling benchmarks, BibTeX citations, and biological specimen micrograph for **Slot 039 (`slot-039-pavement-cell-lobing` / `runions-pavement-cell-lobing`)**:

- **Primary Script:** `2017-runions-pavement-cell-lobing.py`
- **Lean 4 Module:** `morphoeval/lean/loops/Loop_039_pavement_cell_lobing.lean`
- **Math KG Spec:** `morphoeval/math_kg_loops/runions-pavement-cell-lobing.yaml`

## Primary Literature

1. **Sapala, A., Runions, A., Routier-Kierzkowska, A.-L., Das Gupta, M., Hong, L., Hofhuis, H., Verger, S., Mosca, G., Li, C.-B., Hay, A., Hamant, O., Roeder, A. H. K., Tsiantis, M., Prusinkiewicz, P., & Smith, R. S. (2018).** Why plants make puzzle cells, and how their shape emerges. *eLife*, 7, e32794. `doi:10.7554/eLife.32794`
2. **Fu, Y., Gu, Y., Zheng, Z.-L., Wasteneys, G., & Yang, Z. (2005).** *Arabidopsis* interdigitating cell growth requires two antagonistic pathways with opposing action on cell morphogenesis. *Cell*, 120(5), 687–700. `doi:10.1016/j.cell.2004.12.026`
3. **Bidhendi, A. J., Altartouri, B., Gosselin, F. P., & Geitmann, A. (2019).** Mechanical stress initiates and sustains the morphogenesis of wavy leaf epidermal cells. *Nature Plants*, 5(9), 981–992. `doi:10.1038/s41477-019-0500-2`
4. **Lin, D., Cao, L., Zhou, Z., Zhu, L., Ehrhardt, D., Yang, Z., & Fu, Y. (2013).** Rho GTPase signaling activates microtubule severing to promote lobe outgrowth in *Arabidopsis* pavement cells. *Current Biology*, 23(4), 290–297. `doi:10.1016/j.cub.2013.01.022`
5. **Sampathkumar, A., Krupinski, P., Wightman, R., Milani, P., Berquand, A., Boudaoud, A., Hamant, O., Jönsson, H., & Meyerowitz, E. M. (2014).** Subcellular and supracellular mechanical stress prescribes cytoskeleton behavior in *Arabidopsis* cotyledon pavement cells. *eLife*, 3, e01967. `doi:10.7554/eLife.01967`

## Vendored Files

- `pavement_cell_lobing_reference.json` — Largest Empty Circle ($R_{\text{LEC}}$) scaling across cell area $A \in [250, 4500]\ \mu\text{m}^2$, periclinal wall tensile stress $\sigma_{\max} = P R_{\text{LEC}} / (2h)$, ROP2–ROP6 cortical reaction-diffusion parameters, and wild-type vs *rop2 rop4* RNAi / *ktn1* / *ric1* mutant lobeyness ($L = P_{\text{perim}}/P_{\text{hull}}$) and solidity ($S = A/A_{\text{hull}}$) benchmarks.
- `arabidopsis_pavement_cell_specimen.jpg` — Surface micrograph of *Arabidopsis thaliana* leaf epidermis showing interlocking puzzle-shaped pavement cells and stomata (`File:Arabidopsis-epiderm-stomata.jpg`, Wikimedia Commons, CC BY-SA 3.0, SHA-256: `4b7c1062f6ebfd6e792fac017e39d7c011eaf6f8db2c9e492b6a82519088d7b8`).
- `paper_citations.bib` — BibTeX entries with verified DOIs.
- `scout_manifest.json` — Step 1 (`01_scout`) OpenAlex + Wikimedia Commons discovery dossier.
