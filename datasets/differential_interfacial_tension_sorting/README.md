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

3. **`paper_citations.bib`** — BibTeX entries for Brodland (2002), Krieg et al. (2008), Maître et al. (2012), Käfer et al. (2007), and Canty et al. (2017).

4. **`scout_manifest.json`** — Step 1 literature and Wikimedia Commons specimen scout dossier.
