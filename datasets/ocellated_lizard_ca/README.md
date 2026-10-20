# `datasets/ocellated_lizard_ca/` — Manukyan et al. (2017) Ocellated Lizard (*Timon lepidus*) Cellular Automaton Reference Package

Formal SHA-256-locked dataset and biological specimen photograph for `2015-manukyan-ocellated-lizard-ca.py` (Slot 009 of the 100-Loop Infinite Morphospace Campaign).

## Contents

1. **`ocellated_lizard_ca_reference.json`**
   - **Manukyan, Montandon, Fofonjka, Smirnov & Milinkovitch (2017) *Nature* 544(7649):173–179 (`doi:10.1038/nature22031`)**:
     - Continuous Kondo-Asai (`doi:10.1038/376765a0`) reaction-diffusion PDE on a 2D hexagonal skin-scale geometry where skin thickness is reduced by factor $P = h_{\text{border}} / h_{\text{interior}} \approx 0.17$ at inter-scale borders:
       $$\frac{\partial u}{\partial t} = F(u, v) - \gamma_u u + \nabla \cdot \big(D_u(\mathbf{x}) \nabla u\big), \qquad \frac{\partial v}{\partial t} = G(u, v) - \gamma_v v + \nabla \cdot \big(D_v(\mathbf{x}) \nabla v\big)$$
     - Because intra-scale diffusion is much faster than inter-scale border diffusion ($D_{\text{border}} = P \cdot D_{\text{interior}} \ll D_{\text{interior}}$), morphogen concentrations homogenize within each hexagonal scale (producing monochromatic green vs black scales) and the continuous PDE reduces mathematically to a discrete 2D **hexagonal outer-totalistic cellular automaton (CA)**.
   - **Hexagonal Outer-Totalistic CA Transition Probabilities & Adult Neighbor Histograms (Manukyan et al. 2017 Figs. 3–4)**:
     - Juvenile-to-adult iterative scale color flipping $S_i \in \{0\text{ (black)}, 1\text{ (green)}\}$ governed by first-nearest-neighbor green count $n_g(i) \in \{0, \dots, 6\}$.
     - Steady-state adult nearest-neighbor histogram peaking at $n_g = 3$ green neighbors (and $n_b = 3$ black neighbors), forming 1–2-scale-wide labyrinthine chains.
   - **Border Thickness Reduction Factor $P$ Morphospace Sweep**:
     - Five characteristic regimes spanning $P = 0.0$ (decoupled insulated scales), $P = 0.08$ (sharp discrete CA), $P = 0.17$ (physiological *Timon lepidus* scale automaton), $P = 0.45$ (partially discretized Turing labyrinth), and $P = 1.0$ (smooth continuum Turing waves cutting across scale boundaries).

2. **`timon_lepidus_specimen.jpg`**
   - Adult ocellated / jeweled lizard (*Timon lepidus*) displaying the characteristic dorsal mosaic of hexagonal green and black skin scales arranged in labyrinthine chains.
   - Photograph by Danelu ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Timon_Lepidus_danelu.jpg), CC BY-SA 4.0).
   - SHA-256: `bc8e149849e6729d3eb9d4b6507daf6ad440ac50d97546ad4ef29db79c9b91cf`.

3. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons literature & specimen discovery dossier for Slot 009.
   - SHA-256: `e5bca2d8766b49e70b149cad4bf068d8b93192157b57de770973193cc36cab2c`.

4. **`paper_citations.bib`** — BibTeX entries for Manukyan et al. (2017), Kondo & Asai (1995), Fofonjka & Milinkovitch (2021), and Zakany & Milinkovitch (2023).
