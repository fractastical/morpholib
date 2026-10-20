# `datasets/vertex_glass_jamming/` — Bi et al. (2015) *Nature Physics* & Park et al. (2015) *Nature Materials* Zero-Temperature Shape-Index Unjamming Reference Package

Formal SHA-256-locked reference dataset and biological specimen micrograph for `2015-bi-vertex-glass-jamming.py` (Slot 045: Zero-Temperature Shape-Index Unjamming in Biological Epithelia).

## Contents

1. **`vertex_glass_jamming_reference.json`**
   - **Bi, Lopez, Schwarz & Manning (2015) *Nature Physics* 11(12):1074–1079 (`doi:10.1038/nphys3471`)**:
     - Dimensionless vertex model energy functional (Eq. 2):
       $$e = \frac{1}{2}\sum_{i=1}^N \left[ (a_i - 1)^2 + r (p_i - p_0)^2 \right], \qquad a_i = \frac{A_i}{A_0}, \quad p_i = \frac{P_i}{\sqrt{A_0}}, \quad r = \frac{K_P}{K_A A_0}, \quad p_0 = \frac{P_0}{\sqrt{A_0}}$$
     - Geometric shape-index constants:
       - Regular hexagon: $p_{\text{hex}} = \sqrt{8\sqrt{3}} \approx 3.7224$ ($p_{\text{hex}}^2 \approx 13.8564$)
       - Regular pentagon: $p_5 = \sqrt{20\tan(\pi/5)} \approx 3.8119$ ($p_5^2 \approx 14.5309$)
       - Critical zero-temperature rigidity transition in disordered confluent monolayers: $p_0^* \approx 3.813$
     - T1 topological neighbor-exchange energy barrier $\Delta e_{\text{T1}}(p_0)$ and shear modulus $G(p_0)$ scaling across solid jammed ($p_0 < 3.813$) and fluid unjammed ($p_0 \ge 3.813$) regimes.
   - **Park et al. (2015) *Nature Materials* 14(10):1040–1048 (`doi:10.1038/nmat4357`)**:
     - Air-liquid interface (ALI) maturation timecourse (Days 3, 6, 10, 14, 18) of primary human bronchial epithelial cells (HBECs) from non-asthmatic control donors ($\bar{q}_{14} = 3.792 < p_0^*$, jammed solid) vs. asthmatic donors ($\bar{q}_{14} = 3.955 > p_0^*$, unjammed fluid) and mechanically compressed bronchospasm mimic ($\bar{q} = 3.945$).
   - **Atia et al. (2018) *Nature Physics* 14(6):613–620 (`doi:10.1038/s41567-018-0089-9`)**:
     - Universal $k$-Gamma distribution of standardized cell shape index $x = (q - q_{\min})/(\bar{q} - q_{\min})$.

2. **`human_epithelial_monolayer_specimen.jpg`**
   - High-magnification in situ micrograph of a human monolayer sheet (`Homo sapiens`) with dark junctional cell boundaries and blue-stained central nuclei exhibiting contiguous pentagonal, hexagonal, and heptagonal cell packing near the jamming transition.
   - Source: [Wikimedia Commons (`File:Morphology_of_human_corneal_endothelium_in_situ.jpg`)](https://commons.wikimedia.org/wiki/File:Morphology_of_human_corneal_endothelium_in_situ.jpg) (CC BY-SA 4.0).
   - SHA-256: `a4c4d65c5e12e9cb0dbd323a58abb8d923119a494b2216faf119329822c839cb`.

3. **`paper_citations.bib`** — BibTeX citations for Bi et al. (2015), Park et al. (2015), Bi et al. (2016), Atia et al. (2018), and Staple et al. (2010).

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.

5. **`bi_2016_spv_motility_reference.json`** (added in Pass 3, audit `2026-10-19-vertex-glass-jamming-audit-v3`)
   - **Bi, Yang, Marchetti & Manning (2016) *Physical Review X* 6(2):021011 (`doi:10.1103/PhysRevX.6.021011`)** — Self-Propelled Voronoi (SPV) model:
     - Energy $E = \sum_i [K_A (A_i - A_0)^2 + K_P (P_i - P_0)^2]$ (no $1/2$; $K_A = K_P = A_0 = \mu = 1$).
     - Dynamics $\dot{\mathbf r}_i = \mu \mathbf F_i + v_0 \hat{\mathbf n}_i$, $\dot\theta_i = \eta_i$, $\langle\eta_i\eta_j\rangle = 2 D_r \delta_{ij}\delta(t-t')$.
     - Dynamical order parameter $D_{\text{eff}} = D_s / D_0$, $D_0 = v_0^2/(2D_r)$, long-time solid criterion $D_{\text{eff}} < 10^{-3}$.
     - Headline claims used as reference values: zero-motility glass point $p_0^c(0) = 3.81$; structural order parameter $q_c \approx 3.81$ at the boundary for all $(v_0, D_r)$; motility lowers $p_0^c$; shorter persistence (larger $D_r$) raises $p_0^c$.
   - **Provenance caveat:** written offline from the published text; the single figure-derived number (`approximate_boundary_slope_alpha_dr1 = 0.6 ± 0.3`) is an approximate reading of Fig. 1(a) and is flagged as such. It is not digitised data and is used only as an order-of-magnitude check.
   - Also records the finite-time estimators the script uses (window $T = 150$, $N = 100$, $dt = \min(0.1, 0.2/D_r)$, FIRE quench, time-origin-averaged MSD slope, per-free-path T1 rate $\Gamma_{T1}/v_0 > 0.02$ as the fluid criterion).
