# `datasets/limit_cycle_turing/` — Schnakenberg (1979) Limit-Cycle & Turing Reaction-Diffusion Reference Package

Formal SHA-256-locked reference dataset and biological specimen image for `1979-schnakenberg-limit-cycle-turing.py` (Slot 002: `slot-002-limit-cycle-turing`, simulation ID `schnakenberg-limit-cycle-turing`).

## Contents

1. **`scout_manifest.json`**
   - Step 1 (`01_scout`) OpenAlex + Wikimedia Commons literature & specimen discovery dossier for Slot 002.
   - SHA-256: `e71deffb9f317a88709c97e4e7c93507dbf59b6fbcc9b8702527f0bf56202e7f`.

2. **`limit_cycle_turing_reference.json`**
   - **Schnakenberg (1979) *J. Theor. Biol.* 81(3):389–400 (`doi:10.1016/0022-5193(79)90042-0`)**:
     - Tri-molecular cubic autocatalytic reaction-diffusion PDE:
       $$\frac{\partial u}{\partial t} = \gamma(a - u + u^2 v) + \nabla^2 u, \qquad \frac{\partial v}{\partial t} = \gamma(b - u^2 v) + d\nabla^2 v$$
     - Unique positive homogeneous equilibrium $(u_*, v_*) = \left(a + b,\; \frac{b}{(a+b)^2}\right)$ with invariant autocatalytic flux $u_*^2 v_* = b$.
     - Reaction Jacobian $J_0$ trace $\operatorname{tr}(J_0) = \frac{b - a - (a+b)^3}{a+b}$, strictly positive determinant $\det(J_0) = (a+b)^2 > 0$, exact Hopf bifurcation curve $b - a = (a+b)^3$, and exact Turing threshold $d_c(a,b) = \left(\frac{(a+b)^2 + (a+b)\sqrt{2b(a+b)}}{b - a}\right)^2$.
   - **Goldbeter (1975) *Nature* 253:540–542 (`doi:10.1038/253540a0`) & Grace & Hütt (2015) *PLoS Comput. Biol.* (`doi:10.1371/journal.pcbi.1004367`)**:
     - Pulsatile cAMP / glycolytic limit-cycle oscillations and spatial aggregation territories in *Dictyostelium discoideum*.

3. **`dictyostelium_specimen.jpg`**
   - Macro photograph of *Dictyostelium discoideum* fruiting body sorocarps (spore heads/sori, stalks, and basal aggregation mounds on agar).
   - Source: [Wikimedia Commons `File:Dictyostelium Fruiting Bodies.JPG`](https://commons.wikimedia.org/wiki/File:Dictyostelium_Fruiting_Bodies.JPG) (Public Domain, $698 \times 1024$ px).
   - SHA-256: `e2d2b739211bda975bc175167debf554af4991ce21bc8b1106d1053e6bc95a0b`.

4. **`paper_citations.bib`** — BibTeX citations for Schnakenberg (1979), Turing (1952), Goldbeter (1975), and Grace & Hütt (2015).
