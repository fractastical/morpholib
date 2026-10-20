# `datasets/differential_adhesion_sorting/` — Steinberg (1963, 1970), Graner & Glazier (1992) & Foty et al. (1996) Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `1964-steinberg-differential-adhesion-sorting.py` (Slot 041: Differential Adhesion Potts Sorting).

## Contents

1. **`differential_adhesion_sorting_reference.json`**
   - **Steinberg (1963) *Science* 141(3579):401–408 (`doi:10.1126/science.141.3579.401`) & Steinberg (1970) *J. Exp. Zool.* 173(4):395–433 (`doi:10.1002/jez.1401730406`)**:
     - Differential Adhesion Hypothesis (DAH) interfacial tension definitions:
       $$\gamma_{AB} = J_{AB} - \frac{J_{AA} + J_{BB}}{2}, \qquad \gamma_{AM} = J_{AM} - \frac{J_{AA}}{2}, \qquad \gamma_{BM} = J_{BM} - \frac{J_{BB}}{2}$$
     - Four thermodynamic equilibrium regimes:
       - Checkerboard / cell mixing: $\gamma_{AB} < 0$ ($2 J_{AB} < J_{AA} + J_{BB}$)
       - Complete engulfment / concentric cell sorting: $0 < \gamma_{AB} < \gamma_{BM} - \gamma_{AM}$
       - Partial engulfment / doublet lens: $\gamma_{AB} > \gamma_{BM} - \gamma_{AM} > 0$
       - Cell dissociation into medium: $\gamma_{AM} < 0$
   - **Graner & Glazier (1992) *Phys. Rev. Lett.* 69(13):2013–2016 (`doi:10.1103/PhysRevLett.69.2013`) & Glazier & Graner (1993) *Phys. Rev. E* 47(3):2128–2154 (`doi:10.1103/PhysRevE.47.2128`)**:
     - Extended Cellular Potts Hamiltonian $H = \sum J(\tau(\sigma), \tau(\sigma'))(1 - \delta_{\sigma,\sigma'}) + \lambda \sum_{\sigma > 0} (a(\sigma) - A_0)^2$ and canonical cell-sorting bond energies ($J_{AA}=14, J_{BB}=2, J_{AB}=11, J_{AM}=J_{BM}=16, J_{MM}=0$).
   - **Foty, Pfleger, Forgacs & Steinberg (1996) *Development* 122(5):1611–1620 (`doi:10.1242/dev.122.5.1611`)**:
     - Empirical *Gallus gallus* embryonic tissue surface tensions $\sigma \in \{20.1, 12.6, 8.5, 4.6, 1.6\}\text{ dyn/cm}$ across limb bud precartilage, pigmented epithelium, heart ventricle, embryonic liver, and neural retina.

2. **`cell_sorting_specimen.jpg`**
   - Confocal fluorescence microscopy image of a multicellular embryoid body spheroid showing a cohesive inner GFP-labeled cell population sorted inside an enveloping red-labeled outer embryonic tissue layer.
   - Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Constru%E2%80%A1%C3%86o_de_Uma_Imagem_com_Microscopia_Confocal_-_Embryoid_Bodies_(C%E2%80%9Alulas_Estaminais).jpg) (CC BY 3.0).
   - SHA-256: `9b31f7b855d90faa4c561311702d1dfe81c552ece7d3cc8215a0ff82ddd6412c`.

3. **`paper_citations.bib`** — BibTeX entries for Steinberg (1963, 1970), Graner & Glazier (1992), Glazier & Graner (1993), and Foty et al. (1996).

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.
