# `datasets/differential_interfacial_tension_sorting/` — Brodland (2002) / Krieg et al. (2008) / Maître et al. (2012) DITH & Zebrafish Germ-Layer Cortical Tension Reference Package

Formal SHA-256-locked dataset and zebrafish blastoderm/gastrula cross-transplantation fluorescence specimen for `2002-brodland-differential-interfacial-tension-sorting.py` (Slot 044: Differential Interfacial Tension Sorting).

## Contents

1. **`differential_interfacial_tension_sorting_reference.json`**
   - **Brodland (2002) *J. Biomech. Eng.* 124(2):188–197 (`doi:10.1115/1.1449491`)**:
     - Differential Interfacial Tension Hypothesis (DITH) decomposing cell–cell interfacial tension $\gamma_{\alpha\beta}$ into actomyosin cortical tension $\beta_\alpha^{\text{cort}}$ and cadherin adhesion energy $\omega_{\alpha\beta}^{\text{adh}}$:
       $$\gamma_{\alpha\beta} = \beta_\alpha^{\text{cort}} + \beta_\beta^{\text{cort}} - \omega_{\alpha\beta}^{\text{adh}}, \qquad \gamma_{\alpha M} = \beta_\alpha^{\text{cort}}$$
   - **Krieg et al. (2008) *Nature Cell Biology* 10(4):429–436 (`doi:10.1038/ncb1705`)**:
     - Atomic Force Microscopy (AFM) cortical tension hierarchy across zebrafish shield-stage germ-layer progenitors:
       $$\beta_{\text{ectoderm}}^{\text{cort}} \approx 50\ \text{pN}/\mu\text{m} < \beta_{\text{mesoderm}}^{\text{cort}} \approx 95\ \text{pN}/\mu\text{m} < \beta_{\text{endoderm}}^{\text{cort}} \approx 130\ \text{pN}/\mu\text{m}$$
     - Perturbation controls:
       1. **Wild-type engulfment**: $0 < \gamma_{EM} = 30 < \gamma_{M0} - \gamma_{E0} = 45\ \text{pN}/\mu\text{m}$ (ectoderm envelops mesoderm/endoderm core).
       2. **`caMLCK` cortical stiffening reversal**: $\beta_{E+\text{caMLCK}}^{\text{cort}} = 145\ \text{pN}/\mu\text{m} > \beta_M^{\text{cort}} = 95\ \text{pN}/\mu\text{m}$ (inverts engulfment so ectoderm sorts to the interior core).
       3. **Blebbistatin Myosin-II inhibition**: $\beta_E^{\text{cort}} = \beta_M^{\text{cort}} = 25\ \text{pN}/\mu\text{m}$ ($\Delta\beta = 0$, arresting sorting).
   - **Maître et al. (2012) *Science* 338(6104):253–256 (`doi:10.1126/science.1225399`)**:
     - Young–Dupré cell-doublet contact angle and normalized contact radius laws:
       $$\cos\theta = 1 - \frac{\gamma_{cc}}{2\gamma_{cm}}, \qquad \frac{r_c}{R_{\text{cell}}} = \sin\theta = \sqrt{1 - \cos^2\theta}$$

2. **`germ_layer_doublet_specimen.jpg`**
   - Zebrafish (*Danio rerio*) blastoderm/gastrula cross-transplantation fluorescence micrographs (GFP-labeled green and RFP-labeled red progenitor populations in layered engulfment Panels B'–B''' and protruding mushroom doublet Panels C'–C'').
   - Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Whole_blastoderm_transplants_show_yolk_cell_domain_of_function.png) (CC BY 4.0).
   - SHA-256: `6857cce1c9915410213376ff832b64ea4894dfa4e9adf84d7168a7ee59b86ecd`.

   - **Pass 3 — time-dependent Nodal/TGF-β → cortical tension coupling** (`nodal_tgfb_cortical_tension_coupling_pass3` block):
     - Marginal Nodal field rising from sphere stage ($t_{on} = 4$ hpf) with range $\lambda$ cell tiers:
       $$C(x,t) = C_0\,\bigl(1 - e^{-(t - t_{on})/\tau_N}\bigr)\,e^{-x/\lambda}$$
     - Dose-dependent fate → cortical tension target (two-threshold Hill, $K_M = 0.2$, $K_N = 0.6$, $n = 4$), mirroring how Krieg et al. (2008) obtained their progenitors (lefty1 → ectoderm, cyclops → mesoderm, cyclops + casanova → endoderm):
       $$\beta^*(C) = \beta_E + (\beta_M - \beta_E)\,H(C;K_M) + (\beta_N - \beta_M)\,H(C;K_N)$$
     - First-order cortex relaxation $\dot\beta_i = (\beta^*(C(x_i,t)) - \beta_i)/\tau_\beta$ with $\tau_\beta = 0.5$ h, and the continuous-cortex DITH closure $\gamma_{i0} = \beta_i$, $\gamma_{ij} = \kappa\,|\beta_i - \beta_j|$ with $\kappa = 2/3$ (so $\gamma_{EM} = 30$ pN/µm and $0 < \gamma_{ij} < |\gamma_{i0} - \gamma_{j0}|$ always holds).
     - Literature anchors (approximate, recalled, **not digitised** in this pass): ~5 marginal tiers in the shield-stage Nodal/mesendoderm domain (van Boxtel et al. 2015; Dubrulle et al. 2015), endoderm from marginal tiers 1–2 (up to 4) (Warga & Nüsslein-Volhard 1999), MZoep / lefty1 → no mesendoderm (Gritsman et al. 1999), Nodal requirement window sphere → shield (Hagos & Dougan 2007), staging 4 / 6 hpf (Kimmel et al. 1995).
     - Pass 3 also corrected the heterotypic adhesion entries (`omega_EM_adh`, `omega_MN_adh`, `omega_EN_adh`), which previously contradicted the $\gamma_{het} = 30$ pN/µm anchor under $\gamma = \beta_a + \beta_b - \omega$; the corrected values are effective closures, not measured adhesion energies.

3. **`paper_citations.bib`** — BibTeX entries for Brodland (2002), Krieg et al. (2008), Maître et al. (2012), Käfer et al. (2007), Canty et al. (2017) and, since Pass 3, Chen & Schier (2001), Dubrulle et al. (2015), van Boxtel et al. (2015), Hagos & Dougan (2007), Gritsman et al. (1999), Warga & Nüsslein-Volhard (1999) and Kimmel et al. (1995). The Pass 3 DOIs were recorded offline and have not been resolved against the registry.

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.

## Reduction disclosure (Pass 3)

The Nodal coupling is a deliberately reduced model of shield-stage internalization: the gradient is one-dimensional in tier distance from the margin, each cell's Nodal exposure is fixed to its tier of origin (specification memory; exposure is not re-evaluated after the cell moves), there is no epiboly / yolk-syncytial-layer motion, no Lefty feedback, and no digitised pSmad2 profile. The specimen (`germ_layer_doublet_specimen.jpg`) is unchanged and remains SHA-256-pinned.
