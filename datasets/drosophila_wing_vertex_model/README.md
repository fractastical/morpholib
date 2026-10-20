# `datasets/drosophila_wing_vertex_model/` — Farhadifar et al. (2007) *Drosophila* Wing Epithelial Vertex Model Reference Package

Formal SHA-256-locked dataset and biological specimen micrograph for `2007-farhadifar-drosophila-wing-vertex-model.py` (Slot 043: *Drosophila* Wing Disc Apical Vertex Mechanics).

## Contents

1. **`drosophila_wing_vertex_model_reference.json`**
   - **Farhadifar, Röper, Aigouy, Eaton & Jülicher (2007) *Current Biology* 17(24):2095–2104 (`doi:10.1016/j.cub.2007.11.049`)**:
     - Apical epithelial vertex energy functional (Eq. 1):
       $$E(\{\mathbf{r}_i\}) = \sum_\alpha \frac{K_\alpha}{2}\left(A_\alpha - A_\alpha^{(0)}\right)^2 + \sum_{\langle i,j\rangle} \Lambda_{ij} \ell_{ij} + \sum_\alpha \frac{\Gamma_\alpha}{2} L_\alpha^2$$
     - Dimensionless normalized parameters (Eq. 2):
       $$\bar{\Lambda} = \frac{\Lambda}{K (A^{(0)})^{3/2}}, \qquad \bar{\Gamma} = \frac{\Gamma}{K A^{(0)}}$$
     - Canonical *Drosophila* 3rd-instar larval wing disc fit: $(\bar{\Lambda}, \bar{\Gamma}) = (0.12, 0.04)$ (Case I ground state).
     - Phase diagram regimes: Case I (rigid hexagonal ground state, $\Delta E_{T1} > 0$), Case II (soft fluid-like network, $\bar{\Lambda} < \bar{\Lambda}_c(\bar{\Gamma})$), and Case III (cell collapse $A^* \to 0$).
   - **Gibson et al. (2006) *Nature* 442:1038–1041 (`doi:10.1038/nature05014`) & Classen et al. (2005) *Dev. Cell* 9:805–817 (`doi:10.1016/j.devcel.2005.10.016`)**:
     - Polygon-class distributions $P(n)$ for $n \in \{4,5,6,7,8,9\}$ in proliferating larval wing disc ($P(6) = 0.455$, $P(5) = 0.280$, $P(7) = 0.200$) vs. post-mitotic pupal wing after PCP-driven junctional remodeling ($P(6) = 0.780$).
   - **Lewis (1928) *Anat. Rec.* 38:341–376 (`doi:10.1002/ar.1090380305`)**:
     - Linear area–topology law $\langle A_n\rangle / \langle A\rangle = (n - 2)/4$ for $n = 4..9$.

2. **`drosophila_wing_epithelium_specimen.jpg`**
   - Fluorescence microscopy micrograph of an apical junction-stained polygonal epithelial monolayer (green tight/adherens junction network with central nuclei showing contiguous 5-, 6-, and 7-sided epithelial cells and tri-cellular vertices).
   - Source: [Wikimedia Commons (`File:Honeycomb_cells.jpg`)](https://commons.wikimedia.org/wiki/File:Honeycomb_cells.jpg) (CC BY-SA 4.0).
   - SHA-256: `df24205361561b79fc3770d3c4627733b82ebff90957794958d16102ba5745ff`.

3. **`paper_citations.bib`** — BibTeX entries for Farhadifar et al. (2007), Gibson et al. (2006), Classen et al. (2005), Nagai & Honda (2001), Lewis (1928), and (Pass 3) Aigouy et al. (2010), Etournay et al. (2015), Sugimura & Ishihara (2013).

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.

## Pass 3 (v3) additions — PCP Frizzled/Strabismus bond-asymmetry tensor & hinge contraction

The reference JSON gained a `pcp_hinge_contraction_reference` block and three sources used by Section 2b of the script (an explicit tri-cellular vertex model with analytic forces, T1 exchanges, per-bond PCP-modulated line tension $\Lambda_{ab} = \bar\Lambda\,(1 + \varepsilon_{\mathrm{PCP}}\, \mathbf{n}\cdot Q_{ab}\cdot\mathbf{n})$, Fz–Stbm feedback polarity dynamics and an area-preserving hinge-contraction boundary strain $F = \mathrm{diag}(\lambda, 1/\lambda)$):

| id | citation | DOI | status |
|----|----------|-----|--------|
| `aigouy2010` | Aigouy B, Farhadifar R, Staple DB, Sagner A, Röper JC, Jülicher F, Eaton S (2010) *Cell* 142(5):773–786 | `10.1016/j.cell.2010.07.042` | recorded offline, **not** re-verified online in Pass 3 |
| `etournay2015` | Etournay R, Popović M, Merkel M, Nandi A, Blasse C, Aigouy B, Brandl H, Myers G, Salbreux G, Jülicher F, Eaton S (2015) *eLife* 4:e07090 | `10.7554/eLife.07090` | recorded offline, **not** re-verified online in Pass 3 |
| `sugimura2013` | Sugimura K, Ishihara S (2013) *Development* 140(19):4091–4101 | `10.1242/dev.094060` | recorded offline, **not** re-verified online in Pass 3 |

Numeric reference values in `pcp_hinge_contraction_reference` (cell-elongation peak $|Q| \approx 0.25$, rearrangement share of blade shear $\approx 0.5$ from Etournay et al. 2015 Fig. 3; wild-type vs PCP-mutant pupal $P(6) \approx 0.78$ vs $0.60$ from Classen et al. 2005) are **approximate figure reads recorded offline** and flagged as such in the JSON (`provenance` fields). They should be digitised and pinned in Pass 4 before being treated as ground truth. The specimen micrograph and its SHA-256 are unchanged.
