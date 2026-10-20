# `datasets/neural_crest_chase_and_run/` — Carmona-Fontaine et al. (2008, 2011) & Theveneau et al. (2010, 2013) Neural Crest Chase-and-Run Reference Package

Formal SHA-256-locked dataset and biological specimen reconstruction for `2012-carmona-neural-crest-chase-and-run.py` (Slot 091: Contact Inhibition of Locomotion + Co-Attraction + Sdf1/Placode Chase-and-Run Collective Migration).

## Contents

1. **`neural_crest_chase_and_run_reference.json`**
   - **Carmona-Fontaine et al. (2008) *Nature* 456(7224):957–961 (`doi:10.1038/nature07441`)**:
     - Contact Inhibition of Locomotion (CIL) mediated by N-cadherin and Wnt/PCP (Frizzled7/Dishevelled $\to$ RhoA activation and Rac1 collapse at cell-cell contacts), causing colliding neural crest (NC) cells to repolarize away from contact vectors within $d < R_{\text{CIL}}$.
   - **Carmona-Fontaine et al. (2011) *Dev. Cell* 21(6):1026–1037 (`doi:10.1016/j.devcel.2011.10.012`) & Woods et al. (2014) *PLOS ONE* 9(9):e104969 (`doi:10.1371/journal.pone.0104969`)**:
     - Autocrine complement fragment C3a secretion and C3aR receptor signaling generating medium-range exponential Co-Attraction (COA) $F_{\text{COA}}(d) = k_{\text{COA}} \exp(-d / \ell_{\text{COA}})$ that balances short-range CIL dispersion to preserve cohesive mesenchymal streams.
   - **Theveneau et al. (2010) *Dev. Cell* 19(1):39–53 (`doi:10.1016/j.devcel.2010.06.012`) & Theveneau et al. (2013) *Nature Cell Biology* 15(7):763–772 (`doi:10.1038/ncb2772`)**:
     - Coupled **Chase-and-Run** dynamics between Cxcr4+ cranial neural crest cells ("chasers") and Sdf1/Cxcl12-secreting epibranchial placode cells ("runners"):
       $$\mathbf{F}_i^{\text{NC}} = \mathbf{F}_{\text{CIL}, i}^{\text{NC-NC}} + \mathbf{F}_{\text{COA}, i}^{\text{C3a}} + \chi_{\text{Sdf1}} \nabla S_{\text{Sdf1}}(\mathbf{x}_i) + \mathbf{F}_{\text{het}, i}^{\text{NC-PL}}, \qquad \mathbf{F}_m^{\text{PL}} = \mathbf{F}_{\text{run}, m}^{\text{PL-NC}} + \mathbf{F}_{\text{adh}, m}^{\text{PL-PL}}$$
     - Five experimental perturbation regimes (Wild-Type Chase-and-Run, Placode Removal / Sdf1 MO, C3aR Morpholino / COA Ablation, N-cadherin / DshDep+ CIL Blockade, and Stationary Sdf1 Bead Control).

2. **`neural_crest_specimen.jpg`**
   - 3D whole-embryo surface reconstruction of cranial neural crest streams (`r4`, `r6`, `r7`, blue) migrating dorsoventrally toward epibranchial placodes (`e1`, `e2`, `e3`, orange) separated by pharyngeal pouches (`p1`, `p2`, `p3`, green) in control embryos (Panel A) vs FGFR-inhibited (Panel B) and BMPR-inhibited (Panel C) embryos.
   - Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Assessment_of_potential_FGF_and-or_BMP_dependencies_of_neural_crest_streams_emerging_segmentally_from_the_rhombomeres_4,_6,_and_7._9%E2%80%9314_somite_mouse_embryos,_cultured_for_24_h.jpg) (CC BY 4.0).
   - SHA-256: `291091fcf1198a71a62aa3d3ba3599e91cbc8fca709684c6d99c600d03b39c33`.

3. **`paper_citations.bib`** — BibTeX entries for Theveneau et al. (2013), Carmona-Fontaine et al. (2008, 2011), Theveneau et al. (2010), and Woods et al. (2014).

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.
