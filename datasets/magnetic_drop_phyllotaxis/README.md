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

## Pass 3 (v3) errata and specimen-lattice analysis

The three SHA-256-pinned files above are unchanged in Pass 3 (re-pinning them would break the v2 provenance chain), so the following corrections are recorded here and in `morphoeval/audits/2026-10-19-magnetic-drop-phyllotaxis-audit-v3.yaml`:

- **Key names in `magnetic_drop_phyllotaxis_reference.json`.** `analytical_benchmarks.pitchfork_bifurcation_1_1_to_1_2.two_drop_exponential_G_c1 = 0.77348399`, `full_chain_exponential_G_c1 = 0.77253817`, `two_drop_linear_G_c1 = 2.51456033` and `full_chain_linear_G_c1 = 2.51419182` are the roots of the *static* energy-Hessian fold $\partial^2 E/\partial\phi^2|_{\phi=0} = 0$ (frozen history), **not** the dynamical pitchfork threshold $G_{c,1}$ of the iterated placement map. The dynamical roots of $S_{\text{dyn}}(G) = \sum_k (-1)^{k-1} k\, r_k/(r_k - (-1)^k)^5 = 0$ are $G_{c,1}^{(2,\exp)} = 0.86053703$, $G_{c,1}^{(\infty,\exp)} = 0.85901889$, $G_{c,1}^{(2,\text{lin})} = 3.63143185$, $G_{c,1}^{(\infty,\text{lin})} = 3.61755512$; the *emergent* threshold of the unclamped iterated map is $G_{c,1}^{\text{emp}} = 0.8595 \pm 0.0006$ (bisection, 160 iterations, 0.5° criterion). Lean 4 Theorem 4 (`douady_couder_dynamical_threshold_exceeds_static_fold`) gives the integer witness $\mathcal{C}(4,7) = +27687 > 0$, $\mathcal{D}(4,7) = -12646 < 0$, $\mathcal{D}(5,9) = +23872 > 0$, i.e. $G_{\text{fold}} < 3 < G_{c,1}$ under linear advection.
- **`detected_scale_umbos_count: 123`** is the number of Difference-of-Gaussians local maxima in the full disc, not the number of scales. Several scales produce two maxima (umbo and scale tip). The lattice-consistent scale count inside the $r \in [55, 240]$ px annulus, $\Delta u / G_{\text{emp}} = 1.473 / 0.0308 \approx 48$, is the better estimate.
- **`parastichy_pair` of the Fibonacci row `G = 0.35`** is tabulated as `[2, 3]`; the nearest-neighbour contact definition of Douady & Couder on the simulated lattice gives `(1, 2)` at this $G$ (chord distances $d_2 = 2.153 < d_1 = 2.263 < d_3 = 2.41$), so this row sits on the $(1,2) \to (2,3)$ van Iterson transition. All other 9 rows agree.
- **`experimental_regimes[].mean_divergence_deg`** (180.0, 139.2, 138.1, 137.4, 137.5, 99.4) are regime-level summaries read from Douady & Couder (1992) Figs. 1–2 and the text; the PRL does not tabulate per-regime divergence means, so these carry an unquantified reading uncertainty (Pass 4 item: digitise Fig. 3 with error bars).

### Specimen reciprocal-lattice inversion (`conifer_cone_specimen.jpg`)

For a regular spiral lattice with divergence $\alpha$ and plastochron ratio $G$ the log-polar 2D FFT peaks $(m, \omega_m)$ satisfy $m\alpha + \omega_m G = 2\pi j_m$. With eight-fold zero padding in $u$, the two dominant harmonics are $(m, \omega)$ = $(5, -16.93)$ and $(8, +13.75)$ rad per unit $u$, and solving with $j_5 = 2$, $j_8 = 3$ (fixed by Lean 4 Theorem 5, `douady_couder_fibonacci_convergent_window`: $135^\circ \le \alpha \le 138.46^\circ$ for a $(5,8)$ lattice) gives

| quantity | specimen | model (5,8) regime | golden angle |
|---|---:|---:|---:|
| divergence $\alpha$ | **138.03°** | 137.32° (G = 0.06) | 137.508° |
| plastochron ratio $G$ | **0.0308** | 0.06 (table: 0.04–0.10 for (5,8)) | — |
| chirality | $-1$ (mirror of the simulated handedness) | $+1$ | — |
| closure residual $|\omega_3 - (\omega_8 - \omega_5)|$ | 2.2 bins | 0 for a perfect log-periodic lattice | — |
| radial law $R^2$ (linear / exponential vs rank) | 0.977 / 0.899 | — | — |

The same inversion applied to a synthetic log-polar image of the simulated $G = 0.06$ lattice recovers $\alpha = 137.306^\circ$ (error 0.014°) and $G = 0.0600$ (error 0.06 %). The non-zero closure residual and the better linear radial fit show that the mature cone's apparent scale lattice is closer to the silicone-oil dish's linear advection $r_k = R_0(1 + kG)$ (uniform mature scale size, conical surface projected onto the image plane) than to exponential meristem growth — a transfer-boundary caveat carried into the audit.

### Additional reference used in Pass 3

- van Iterson G (1907). *Mathematische und mikroskopisch-anatomische Studien über Blattstellungen, nebst Betrachtungen über den Schalenbau der Miliolinen*. Gustav Fischer, Jena. (Contact-parastichy / divergence-angle convergent windows; not yet in the SHA-pinned `paper_citations.bib` — Pass 4 re-pin item.)
