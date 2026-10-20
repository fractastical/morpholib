# `datasets/magnetic_drop_phyllotaxis/` — Douady & Couder (1992, 1996) Magnetic-Drop Phyllotaxis Reference Package

Formal SHA-256-locked reference dataset and biological conifer cone specimen photograph for `1992-douady-magnetic-drop-phyllotaxis.py` (Slot 031 of the 100-Loop Infinite Morphospace & MorphoEval Campaign).

## Contents

1. **`magnetic_drop_phyllotaxis_reference.json`**
   - **Douady & Couder (1992) *Phys. Rev. Lett.* 68(13):2098–2101 (`doi:10.1103/PhysRevLett.68.2098`)** & **Douady & Couder (1996) *J. Theor. Biol.* 178(3):255–274 (`doi:10.1006/jtbi.1996.0024`)**:
     - Iterative magnetic dipole repulsion energy $E(\phi) = \sum_{k \ge 1} d_k(\phi, \phi_{n-k})^{-\gamma}$ ($\gamma = 3$) on a circular meristem rim $r = R_0$ with radial advection $r_k = R_0(1 + k G)$ (linear dish advection) and $r_k = R_0 \exp(k G)$ (exponential meristem advection), controlled by dimensionless plastochron ratio $G = V T / R_0$.
     - Analytical 2-drop and full-chain pitchfork bifurcation thresholds $G_{c,1}$ separating alternating opposition $(1,1)$ ($\Delta\phi = 180^\circ$) from chiral spiral $(1,2)$ modes ($G_{c,1}^{(2,\text{exp})} = 0.77348399$, $G_{c,1}^{(\infty,\text{exp})} = 0.77253817$, $G_{c,1}^{(2,\text{lin})} = 2.51456033$).
     - Regular-spiral equilibrium zero-force curves $F(G, \Delta\phi) = \left.\frac{\partial E}{\partial \phi}\right|_{\phi=0} = 0$ and imperfect pitchfork cascades along the main Fibonacci branch $(1,1) \to (1,2) \to (2,3) \to (3,5) \to (5,8) \to (8,13)$ ($\Delta\phi \to 137.507764^\circ$) vs. the disconnected Lucas saddle-node branch $(1,3) \to (3,4) \to (4,7) \to (7,11)$ ($\Delta\phi \to 99.501553^\circ$).

2. **`conifer_cone_specimen.jpg`**
   - Apical view of a woody conifer seed cone (*Pinus* sp.) exhibiting interlocking Fibonacci $(5, 8)$ opposing contact parastichies.
   - Photograph by Bubba73 ([Wikimedia Commons `File:Top of a pine cone.jpg`](https://commons.wikimedia.org/wiki/File:Top_of_a_pine_cone.jpg), CC BY-SA 4.0).
   - Dimensions: `637 x 640` px; SHA-256: `7bcb68e856d4db535e622eb9674c6d7f6699047aa008ef429cbeefbfa0bd1fcb`.

3. **`paper_citations.bib`** — BibTeX citations for Douady & Couder (1992, 1996a, 1996b) and Levitov (1991).

4. **`scout_manifest.json`** — Step 1 (`01_scout`) literature and specimen discovery manifest.
