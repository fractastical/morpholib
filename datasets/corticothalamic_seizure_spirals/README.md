# Huang et al. (2004, 2010) & Pinto & Ermentrout (2001) Neocortical Spiral Waves & Phase Singularity Dataset

Primary reference dataset and biological specimen metadata for `2013-ermentrout-corticothalamic-seizure-spirals.py` (Slot 089 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Huang et al. (2004, 2010) *J. Neurosci.* & Pinto & Ermentrout (2001) *SIAM J. Appl. Math.*

1. **2D Excitatory–Adaptation Wilson–Cowan / Pinto–Ermentrout Cortical Neural Field:**

$$\tau_u \frac{\partial u(\mathbf{x}, t)}{\partial t} = -u + D_u \nabla^2 u + F\!\Big(w_{ee}(K_e * u)(\mathbf{x}, t) - g_a a(\mathbf{x}, t) + I_{\text{ext}}\Big), \qquad \tau_a \frac{\partial a(\mathbf{x}, t)}{\partial t} = -a + u$$

where $u(\mathbf{x}, t)$ is local excitatory population activity across the 2D tangential neocortical sheet, $a(\mathbf{x}, t)$ is slow spike-frequency adaptation / synaptic recovery, $K_e(\mathbf{r}) = \frac{1}{2\pi\sigma_e^2}\exp(-\|\mathbf{r}\|^2 / 2\sigma_e^2)$ is the horizontal intracortical axonal footprint, and $F(z) = [1 + \exp(-\beta_f(z - \theta_f))]^{-1}$.

2. **Topological Phase Winding Number & Core Amplitude Deficit (Huang et al. 2010):**

$$Z(\mathbf{x}, t) = (u - \bar{u}) + i(a - \bar{a}) = A(\mathbf{x}, t)e^{i\phi(\mathbf{x}, t)}, \qquad W = \frac{1}{2\pi}\oint_C \nabla \phi \cdot d\mathbf{l} = \pm 1$$

At the spiral phase singularity $\mathbf{x}_{\text{core}}$, all isophase lines $\phi \in [-\pi, \pi)$ converge, forcing oscillation amplitude $A(\mathbf{x}_{\text{core}}) = 0$ with radial recovery $A(r) = A_\infty \tanh(r / r_{\text{core}})$. When a planar cortical wave breaks, singularities nucleate in counter-rotating pairs ($W_+ = +1, W_- = -1$) conserving net topological charge $\sum_i W_i = 0$.

3. **Spiral Dispersion & Axonal Footprint Scaling (Pinto & Ermentrout 2001):**

$$\lambda_{\text{spiral}} = \frac{c_{\text{wave}}}{f_{\text{rot}}}, \qquad c_{\text{wave}} \propto \sigma_e f_{\text{rot}}$$

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `rat_neocortical_tangential_sheet_specimen.jpg` | Tangential histological section of *Rattus norvegicus* somatosensory neocortex (Layer IV barrel field, cytochrome oxidase stain, 2048×1536 px, Wikimedia Commons `File:RatBarrelFieldCOstain.jpg`, Public domain) | `c4fdfb27cd39c842406116f00674c56013f6fc90d872f9995682c7bd47c1e32c` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 089 | `e31d28291faa189abf4428c8ce7e4a7fbe5eb2e39eb5212b53cc4b7035371444` |
| `paper_citations.bib` | BibTeX entries for Huang et al. (2010), Huang et al. (2004), Pinto & Ermentrout (2001), and Townsend et al. (2015) | `2ea747169ffdca2a5b2a4934f8f30b240906360bafa48622c27c0076f752b559` |
| `corticothalamic_seizure_spirals_reference.json` | Canonical Wilson–Cowan / Pinto–Ermentrout neural field equations, 6 experimental regimes, and SHA-256 locks | verified at runtime |
