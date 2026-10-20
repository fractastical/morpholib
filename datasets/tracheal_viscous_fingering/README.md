# `datasets/tracheal_viscous_fingering/` — Nelson et al. (2017) *Development*, Kim et al. (2015) *Dev. Cell* & Unbekandt et al. (2008) *Mech. Dev.* Reference Package

Formal SHA-256-locked reference dataset and *Mus musculus* pseudoglandular embryonic lung branching specimen for `2015-sugimura-tracheal-viscous-fingering.py` (Slot 059 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Mathematical & Biomechanical Framework

1. **Transmural Pressure Gauge Invariance in Microfluidic Chest Cavities (`Nelson et al. 2017` *Development* 144:4328–4335, `doi:10.1242/dev.154823`)**:
   - Chloride-driven luminal fluid secretion against hydraulic laryngeal/tracheal resistance maintains a positive transmural pressure difference across the developing airway wall:
     $$\Delta P(P_{\text{lumen}}, P_{\text{pleura}}) = P_{\text{lumen}} - P_{\text{pleura}} \approx 100\text{–}300\text{ Pa}.$$
   - By independently clamping luminal pressure $P_{\text{lumen}}$ and pleural cavity pressure $P_{\text{pleura}}$ in a microfluidic chest organ culture device, Nelson et al. (2017) demonstrated exact common-mode gauge invariance:
     $$\Delta P(P_{\text{lumen}} + P_0, \; P_{\text{pleura}} + P_0) = \Delta P(P_{\text{lumen}}, P_{\text{pleura}}).$$

2. **Laplace–Young Circumferential Epithelial Wall Stress (`Nelson et al. 2017`; `Jaslove & Nelson 2018` *Development* 145:dev167395)**:
   - The distending transmural pressure $\Delta P$ generates circumferential (hoop) tensile stress across the distal epithelial bud wall of luminal radius $R$ and thickness $h$:
     $$\sigma_\theta(\Delta P, R, h) = \frac{\Delta P \cdot R}{h}.$$
   - Strictly increasing $\Delta P_2 > \Delta P_1$ at $R, h > 0$ monotonically elevates epithelial wall tension $\sigma_\theta(\Delta P_2) > \sigma_\theta(\Delta P_1)$, promoting Etv4/FGF10/ERK1/2 proliferation at distal tips.

3. **Mechanosensitive Tip Bifurcation Rate & Synchronized Airway Smooth Muscle (ASM) Peristalsis (`Kim et al. 2015` *Dev. Cell* 34:668–681; `Unbekandt et al. 2008` *Mech. Dev.* 125:314–324)**:
   - Above the critical distension threshold $\Delta P_c \approx 50\text{ Pa}$, synchronized ASM $\text{Ca}^{2+}$ peristaltic wave frequency $f_{\text{ASM}}(\Delta P)$ and tip bifurcation rate $\Gamma_{\text{branch}}(\Delta P)$ follow:
     $$f_{\text{ASM}}(\Delta P) = f_0 + f_{\max}\frac{\Delta P^2}{K_{\text{ASM}}^2 + \Delta P^2}, \qquad \Gamma_{\text{branch}}(\Delta P) = \Gamma_0 \, \alpha_{\text{ASM}} \frac{(\Delta P - \Delta P_c)_+^n}{K_P^n + (\Delta P - \Delta P_c)_+^n},$$
     where tracheal drainage ($\Delta P \le \Delta P_c$) or nifedipine L-type $\text{Ca}^{2+}$ channel blockade ($\alpha_{\text{ASM}} \to 0.05$) collapses terminal bud bifurcation.

## Vendored Files

- `tracheal_viscous_fingering_reference.json` — SHA-256-locked reference parameters, 8-point $\Delta P \in [0, 400]\text{ Pa}$ microfluidic chest sweep, 4 common-mode gauge invariance $(P_{\text{lumen}}, P_{\text{pleura}})$ pairs, nifedipine and tracheal drainage controls, and empirical specimen morphometrics.
- `embryonic_lung_branching_specimen.jpg` — *Mus musculus* pseudoglandular (E11–E16.5) embryonic lung branching tree and H&E epithelial tubule cross-section (`File:Development of the respiratory system in mice..jpg`, Wikimedia Commons CC BY 3.0, `600x343` px, `sha256: 8086ebb8b44cbdff4002f62707afa7ec0732edf41f9c53bfef049bf09eef6bae`).
- `paper_citations.bib` — BibTeX citations with verified DOIs.
- `scout_manifest.json` — Step 1 (`01_scout`) literature and specimen manifest.
