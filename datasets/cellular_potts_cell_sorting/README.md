# `datasets/cellular_potts_cell_sorting/` — Graner & Glazier (1992) PRL / Glazier & Graner (1993) PRE Cellular Potts Model Reference Package

Formal SHA-256-locked dataset and biological multicellular aggregate specimen photograph for `1992-graner-cellular-potts-cell-sorting.py` (Slot 042: Cellular Potts Cell Sorting).

## Contents

1. **`cellular_potts_cell_sorting_reference.json`**
   - **Graner & Glazier (1992) *Phys. Rev. Lett.* 69(13):2013–2016 (`doi:10.1103/PhysRevLett.69.2013`) & Glazier & Graner (1993) *Phys. Rev. E* 47(3):2128–2154 (`doi:10.1103/PhysRevE.47.2128`)**:
     - Extended large-$Q$ Cellular Potts Model (CPM / GGH) Hamiltonian:
       $$H = \sum_{\langle i,j\rangle} J\bigl(\tau(\sigma_i), \tau(\sigma_j)\bigr)\bigl(1 - \delta_{\sigma_i, \sigma_j}\bigr) + \lambda \sum_{\sigma > 0} \bigl(v_\sigma - V_t(\tau(\sigma))\bigr)^2$$
     - Bond-energy surface tension relations (with $J_{\text{MM}} = 0$):
       $$\gamma_{\text{LD}} = J_{\text{LD}} - \frac{J_{\text{LL}} + J_{\text{DD}}}{2}, \qquad \gamma_{\text{LM}} = J_{\text{LM}} - \frac{J_{\text{LL}}}{2}, \qquad \gamma_{\text{DM}} = J_{\text{DM}} - \frac{J_{\text{DD}}}{2}$$
     - Four canonical phase regimes:
       1. **Complete sorting / engulfment**: $0 < \gamma_{\text{LD}} < \gamma_{\text{DM}} - \gamma_{\text{LM}}$ ($J_{\text{LL}}=14, J_{\text{DD}}=2, J_{\text{LD}}=11, J_{\text{LM}}=J_{\text{DM}}=16 \implies \gamma_{\text{LD}}=3 < 6$)
       2. **Partial sorting / lens formation**: $\gamma_{\text{LD}} > |\gamma_{\text{LM}} - \gamma_{\text{DM}}| > 0$ ($J_{\text{LD}}=16 \implies \gamma_{\text{LD}}=8 > 6$)
       3. **Checkerboard cell mixing**: $\gamma_{\text{LD}} < 0$ ($J_{\text{LL}}=J_{\text{DD}}=14, J_{\text{LD}}=4 \implies \gamma_{\text{LD}}=-10 < 0$)
       4. **Cell dissociation into medium**: $\gamma_{\text{LM}} < 0$ ($J_{\text{LM}}=2 \implies \gamma_{\text{LM}}=-5 < 0$)
     - Logarithmic heterotypic boundary length decay $L_{\text{het}}(t) \sim A - B \ln t$ (Graner & Glazier 1992 Fig. 3) and zero-temperature $T = 0$ pinning arrest vs. optimal $T = 10$ sorting and high-$T$ melting controls (Glazier & Graner 1993 Section III.B).
   - **Pass 3 extension — Ouchi, Glazier, Rieu, Upadhyaya & Sawada (2003) *Physica A* 329(3–4):451–458 (`doi:10.1016/S0378-4371(03)00574-0`, Eq. 2)** (`perimeter_constraint_extension_ouchi_2003` block):
     - Perimeter / cortical tension elasticity term added to the Hamiltonian:
       $$H_P = \lambda_P \sum_{\sigma > 0} \bigl(p_\sigma - P_t\bigr)^2$$
       with $p_\sigma$ the 8-neighbour bond-count perimeter of cell $\sigma$. For an $s \times s$ square block $p = 12 s - 4$, so the canonical $V_t = 16$ ($4 \times 4$) cell has $P_t = 44$ (machine-checked in Lean `cpm_square_block_target_perimeter`).
     - Single-copy perimeter change laws $\Delta p_{\text{old}} = 2 n_{\text{same}} - 8$, $\Delta p_{\text{new}} = 8 - 2 n_{\text{same}}$ (always even, $|\Delta p| \le 8$) and the energy change $\Delta E_P = \lambda_P \Delta p\,(2(p - P_t) + \Delta p)$ (Lean `cpm_perimeter_elasticity_restoring_identity`).
     - Sweep tables: $\lambda_P \in \{0, 0.25, 0.5, 1, 2, 4\}$ (ablation; $\lambda_P = 0$ is the exact 1992 Hamiltonian) and $P_t \in \{32, 38, 44, 50, 56\}$ ($3\times3$ … $5\times5$ squares).
     - The two Pass 3 DOIs (Ouchi 2003, Farhadifar 2007) were recorded from memory during the offline loop and are flagged `doi_verification` in the JSON; re-resolve them before publishing.

2. **`embryonic_cell_aggregate_specimen.jpg`**
   - Confocal fluorescence microscopy specimen of a multicellular embryonic cell aggregate / embryoid body spheroid showing a cohesive inner GFP-labeled dark-type cell bolus (`D`) enveloped inside a polygonal RFP-outlined light-type outer tissue matrix (`L`).
   - Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Constru%E2%80%A1%C3%86o_de_Uma_Imagem_com_Microscopia_Confocal_-_Embryoid_Bodies_(C%E2%80%9Alulas_Estaminais).jpg) (CC BY 3.0).
   - SHA-256: `9b31f7b855d90faa4c561311702d1dfe81c552ece7d3cc8215a0ff82ddd6412c`.

3. **`paper_citations.bib`** — BibTeX entries for Graner & Glazier (1992), Glazier & Graner (1993), Steinberg (1963), Gierer et al. (1972), Foty et al. (1996), and (Pass 3) Ouchi et al. (2003) and Farhadifar et al. (2007).

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.
