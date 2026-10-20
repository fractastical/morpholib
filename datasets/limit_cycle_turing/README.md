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

4. **`paper_citations.bib`** — BibTeX citations for Schnakenberg (1979), Turing (1952), Goldbeter (1975), Grace & Hütt (2015), and (Pass 3) Martiel & Goldbeter (1987), Gerisch & Hess (1974), Crampin, Gaffney & Maini (1999).

5. **`pass3_curved_geometry_pde_reference.json`** (Pass 3, audit `2026-10-19-limit-cycle-turing-audit-v3`)
   - **Curved geometry (remediation R1).** Laplace–Beltrami ladder on a hemispherical mound $-l(l+1)/R^2$, the planar Turing band $q_- = 15.144 < k^2 < q_+ = 118.856$ ($0.576 < \lambda < 1.615$, $\lambda_c = 0.9413$), the critical mound radius $R_{\min} = \sqrt{2/q_+} = 0.1297$ below which no spherical harmonic is unstable, the Lean Law 4 integer table $\det_J(18, l(l+1)) = 100, -96, -56, 1084$ for $l = 0..3$, the mound-radius sweep $R \in \{0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50\}$ with predicted fastest-growing degrees $l^* = 0, 1, 2, 3, 5, 6, 10$ and hexagonal-packing spot counts, and the elongating-stalk setup ($R_{\text{stalk}} = 0.08$, $L: 1 \to 4$, Crampin–Gaffney–Maini uniform-growth dilution).
   - **Phosphodiesterase kinetics (remediation R2).** First-order PDE extension with generalized Hopf curve $(1+\kappa)^3 (b-a) = (a+b)^3$ (Lean Law 5; integer instance $(a,b) = (1,9)$: $\kappa = 0 \to -992$, $\kappa = 4 \to 0$, $\kappa = 5 \to 728$), Michaelis–Menten PDE sink $-k_e u/(K_m + u)$ with closed-form equilibrium and Hopf thresholds $k_e^{\text{Hopf}}(0.1, 0.9; K_m = 0.5, 1, 2) = 0.0878, 0.1249, 0.2003$, and the **Martiel & Goldbeter (1987) *Biophys. J.* 52:807–828 (`doi:10.1016/S0006-3495(87)83275-7`)** 3-variable receptor-desensitization relay model with its Table I parameter set ($k_e = 5.4$ min$^{-1}$ extracellular PDE rate), the $k_e$ sweep window, and the experimental cAMP period anchor (~7 min, range 5–10 min; **Gerisch & Hess (1974) *PNAS* 71:2118–2122, `doi:10.1073/pnas.71.5.2118`**).
   - All three new DOIs were resolved through the Crossref API on 2026-10-19 (recorded in the JSON `resolved_via` fields). This file is not SHA-256-pinned by the script (it is a reference table, re-derived by `--self-test` check 16), unlike `scout_manifest.json` and `dictyostelium_specimen.jpg`.

## Geometry caveat (Pass 3)

The hemispherical mound and cylindrical stalk are *surface* (2-manifold) reductions of the three-dimensional *Dictyostelium* culminant: reaction-diffusion is solved on the curved surface, not through the cell mass. Radii and lengths are in the same dimensionless domain units as the planar $L = 6$ simulations and are **not** yet calibrated to the specimen's sorus diameter (32.9 px) or stalk length; that calibration and a volumetric solver are carried as Pass 4 remediation items.
