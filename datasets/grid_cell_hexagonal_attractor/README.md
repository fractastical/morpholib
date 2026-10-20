# Burak & Fiete (2009), Hafting et al. (2005) & Stensola et al. (2012) Entorhinal Grid-Cell Hexagonal Attractor Dataset

Primary reference dataset and biological specimen metadata for `2008-burak-grid-cell-hexagonal-attractor.py` (Slot 088 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Burak & Fiete (2009) *PLoS Comput. Biol.* 5:e1000291, Hafting et al. (2005) *Nature* 436:801-806 & Stensola et al. (2012) *Nature* 492:72-78

1. **2D Toroidal Continuous Attractor Neural Network (CANN) with Velocity-Shifted Surround Inhibition:**

$$\tau \frac{\partial s_{\theta}(\mathbf{x}, t)}{\partial t} + s_{\theta}(\mathbf{x}, t) = \left[\sum_{\theta'} \int W_0\big(\mathbf{x} - \mathbf{x}' - l\hat{\mathbf{e}}(\theta')\big)\,s_{\theta'}(\mathbf{x}', t)\,d\mathbf{x}' + B\big(1 + \alpha \mathbf{v}(t)\cdot \hat{\mathbf{e}}(\theta)\big)\right]_+$$

where $\theta \in \{0, \pi/2, \pi, 3\pi/2\}$ indexes the 4 head-direction-conjunctive neural subpopulations $(\text{E, N, W, S})$, $[u]_+ = \max(u, 0)$ is rectifying threshold-linear positivity, $l$ is the outgoing synaptic shift along preferred heading $\hat{\mathbf{e}}(\theta)$, and $\alpha \mathbf{v}(t)\cdot\hat{\mathbf{e}}(\theta)$ is velocity-modulated feedforward input from medial-septal / head-direction cells.

2. **Annular Surround-Inhibition Kernel & Quadratic Resonant Triad Selection of $60^\circ$ Hexagonal Planforms:**

$$W_0(\Delta\mathbf{x}) = a \exp(-\gamma \|\Delta\mathbf{x}\|^2) - \exp(-\beta \|\Delta\mathbf{x}\|^2), \qquad \gamma = 1.05\,\beta$$

In Fourier space, $\widehat{W}_0(\mathbf{k})$ peaks on an isotropic ring $|\mathbf{k}| = k_c$. Because the firing rate is rectified ($[u]_+ \approx \frac{1}{2}u + \frac{1}{4}u^2 + \dots$), three wavevectors at $120^\circ$ separation ($\mathbf{k}_1 + \mathbf{k}_2 + \mathbf{k}_3 = \mathbf{0}$, $|\mathbf{k}_i| = k_c$) form a closed resonant triad with positive cubic energy gain $\propto A_1 A_2 A_3 > 0$, selecting a $60^\circ$ hexagonal lattice ($G_{60} > 1.0$) over $90^\circ$ orthogonal square modes ($G_{60} < 0$).

3. **Stensola et al. (2012) Discrete Dorso-Ventral Module Scale Hierarchy ($\lambda_{m+1}/\lambda_m = \sqrt{2}$):**

$$\lambda_m = \lambda_1 (\sqrt{2})^{m-1} \in \{38, 54, 76, 108\}\text{ cm}, \qquad \frac{\lambda_{m+1}^2}{\lambda_m^2} = 2$$

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `rat_entorhinal_hippocampal_specimen.jpg` | Multi-channel confocal immunofluorescence micrograph of *Rattus norvegicus* hippocampal-entorhinal circuit stained for NeuN (green somata), myelin basic protein / MBP (red perforant-path fibers), and DNA (blue nuclei) (Wikimedia Commons, GerryShaw, CC BY-SA 3.0, 1800×1200 px) | `63b3a392f8d0b25f6ddac088b95d8c7152c39374ead9a87089cafc48d2785666` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 088 | `77ac05875318888d1fe936a2f765548930637a526aa76f126602fb68f3270d0b` |
| `paper_citations.bib` | BibTeX entries for Burak & Fiete (2009), Hafting et al. (2005), Stensola et al. (2012), and Brandon et al. (2011) | `e1074e8d98b9dc0070e4415cc7600e00a06bde8222ba97d67d1d4e6ed9aa75b8` |
| `grid_cell_hexagonal_attractor_reference.json` | Canonical Burak–Fiete CANN equations, Stensola 4-module hierarchy, 6 experimental regimes, and SHA-256 locks | verified at runtime |
