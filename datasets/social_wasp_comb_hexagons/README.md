# Dataset: Stigmergic Hexagonal Lattice Wasp Nest Architecture & Hales' Honeycomb Isoperimetric Optimality (`social_wasp_comb_hexagons`)

## Primary Literature & DOIs

1. **Karsai I, Pénzes Z (1993)**.
   *Comb building in social wasps: self-organization and stigmergic script.* **Journal of Theoretical Biology** 161(4):505–525.
   DOI: [`10.1006/jtbi.1993.1072`](https://doi.org/10.1006/jtbi.1993.1072)
2. **Theraulaz G, Bonabeau E (1995)**.
   *Coordination in distributed building.* **Science** 269(5224):686–688.
   DOI: [`10.1126/science.269.5224.686`](https://doi.org/10.1126/science.269.5224.686)
3. **Karsai I, Pénzes Z (2000)**.
   *Optimality of cell arrangement and rules of thumb of cell initiation in Polistes dominulus: a modeling approach.* **Behavioral Ecology** 11(4):387–395.
   DOI: [`10.1093/beheco/11.4.387`](https://doi.org/10.1093/beheco/11.4.387)
4. **Smith ML, Napp KH, Gravish N, Seeley TD, Petersen KH (2021)**.
   *Imperfect comb construction reveals the architectural abilities of honeybees.* **Proceedings of the National Academy of Sciences** 118(31):e2103605118.
   DOI: [`10.1073/pnas.2103605118`](https://doi.org/10.1073/pnas.2103605118)
5. **Hales TC (2001)**.
   *The Honeycomb Conjecture.* **Discrete & Computational Geometry** 25(1):1–22.
   DOI: [`10.1007/s004540010071`](https://doi.org/10.1007/s004540010071)

## Governing Mathematical & Architectural Framework

Social paper wasps (*Polistes dominula*, *Vespula vulgaris*) construct compact, planar and domed hexagonal brood combs without a global blueprint via local stigmergic cues encoded in the number of existing adjacent walls $k \in \{1, 2, 3\}$ at candidate perimeter pockets:

1. **Boltzmann Stigmergic Wall-Sharing Rule (`Karsai & Pénzes 1993; Theraulaz & Bonabeau 1995; Karsai & Pénzes 2000`):**
   $$P_{\text{init}}(j) = \frac{\exp(\beta k_j)}{\sum_{m \in \partial\Omega} \exp(\beta k_m)}, \qquad k_j \in \{1, 2, 3\},$$
   where $\beta > 0$ is the stigmergic wall-sharing selectivity. When $\beta \gg 1$ ($w_3 \gg w_2 \gg w_1 \approx 0$), wasps prioritize 3-wall pockets ($k=3$) along the sides of an incomplete ring before initiating a new ring at a 2-wall corner pocket ($k=2$), while suppressing 1-wall dendritic protrusions ($k=1$).
2. **Centered Hexagonal Ring Progression & Shared-Wall Pulp Economy (`Karsai & Pénzes 1993; Hales 2001`):**
   A compact concentric hexagonal comb of $R$ rings around a central founder cell contains
   $$N(R) = 1 + 3R(R+1) \quad\text{cells}, \qquad \Delta N(R) = N(R) - N(R-1) = 6R \quad\text{cells in ring } R,$$
   and requires a total of
   $$W(R) = 3 N(R) + 6R + 3 = 9R^2 + 15R + 6 \quad\text{distinct wall segments},$$
   so the wall-to-cell ratio $W(R)/N(R) = 3 + \frac{6R+3}{1 + 3R(R+1)}$ converges monotonically to $3.0$ walls/cell as $R \to \infty$, saving $S_{\text{saved}}(R) = 6N(R) - W(R) = 3R(3R+1)$ paper pulp walls ($50\%$ asymptotic pulp savings compared to $6$ walls per isolated cell).
3. **Topological 5–7 Disclination Pairs at Worker-to-Reproductive Cell Size Transitions (`Smith et al. 2021 PNAS`):**
   When builders transition from smaller worker cells ($d_w = 4.6\,\text{mm}$) to larger reproductive cells ($d_q = \lambda d_w$, $\lambda > 1$), lattice incommensurability is resolved by inserting paired pentagon–heptagon ($z_5 = 5, z_7 = 7$) topological dislocations whose Euler charges cancel identically:
   $$q_5 + q_7 = (6 - 5) + (6 - 7) = (+1) + (-1) = 0, \qquad \sum_{i} (6 - z_i) = 0.$$

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 096.
- `paper_citations.bib` — BibTeX citations with DOIs for Karsai & Pénzes (1993, 2000), Theraulaz & Bonabeau (1995), Smith et al. (2021), and Hales (2001).
- `social_wasp_comb_hexagons_reference.json` — Quantitative reference tables (8-ring concentric comb economy trajectory $R \in [0, 7]$, 7-point stigmergic selectivity sweep $\beta \in [0.0, 5.5]$, 5 topological/ablation controls, and SHA-256 digests).
- `polistes_wasp_comb_specimen.jpg` — CC BY 2.0 photograph of a *Polistes* paper wasp hexagonal brood comb (`File:Flickr - ggallice - Paper wasp nest.jpg`, 900×600 px).
