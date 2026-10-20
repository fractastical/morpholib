# Miura et al. (2009) & Jaslow (1990) Cranial Suture Fractal Interdigitation Reference Dataset

Primary reference dataset and biological specimen metadata for `2020-bailleul-allometric-cranial-sutures.py` (Slot 020 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Miura et al. (2009) *PNAS* 106:11218-11223 & Jaslow (1990) *J. Biomech.* 23:313-321

1. **Damped Kuramoto–Sivashinsky / KPZ Osteogenic Front Instability PDE:**

$$\frac{\partial h(x, t)}{\partial t} = -\alpha \frac{\partial^2 h}{\partial x^2} - \beta \frac{\partial^4 h}{\partial x^4} + \frac{v_0}{2}\left(\frac{\partial h}{\partial x}\right)^2 - \gamma h + \eta(x, t), \qquad \alpha > 0,\ \beta > 0$$

where $h(x, t)$ is the transverse osteogenic front displacement along the calvarial suture midline $x \in [0, L]$, $\alpha > 0$ is the Mullins–Sekerka-like short-wavelength osteogenic front destabilization rate driven by coupled BMP4/FGF2 osteoblast differentiation and tensile mechanical strain, $\beta > 0$ is fourth-order curvature/surface-tension smoothing across the mesenchymal suture width, $\frac{v_0}{2}(\partial_x h)^2$ is KPZ local normal-growth nonlinearity, $\gamma \ge 0$ is mesenchymal boundary confinement, and $\eta(x, t)$ is spatial osteogenic stochasticity.

2. **Linear Dispersion Relation & Primary Interdigitation Wavelength Selection:**

$$\omega(k) = \alpha k^2 - \beta k^4 - \gamma, \qquad k_{\max} = \sqrt{\frac{\alpha}{2\beta}}, \qquad \lambda_{\max} = 2\pi\sqrt{\frac{2\beta}{\alpha}}, \qquad \omega(k_{\max}) = \frac{\alpha^2}{4\beta} - \gamma$$

When intracranial/masticatory tensile strain is unloaded or *FGFR2* gain-of-function craniosynostosis (Apert/Crouzon syndrome) obliterates the mesenchymal inhibition zone ($\alpha \le 0$), $\omega(k) \le 0$ for all wavenumbers $k$, collapsing the suture to a flat or fused seam ($D_f \to 1.0$, $\text{SII} \to 1.0$).

3. **2D Bidirectional Overhanging Suture Front & Jaslow (1990) Bending Energy Absorption:**
   - Coupled 2D reaction-diffusion phase-field / bidirectional normal-front propagation produces multi-valued overhanging bone fingers with Suture Interdigitation Index $\text{SII} = L_{\text{arc}} / L_{\text{chord}} > 2.5$ and box-counting fractal dimension $D_f \in [1.30, 1.45]$.
   - By Jaslow (1990) *J. Biomech.* 23:313-321, dynamic impact and bending energy absorption per unit chord length increases monotonically with interdigitation complexity: $E_{\text{bend}}(\text{SII}) = E_0 \big(1 + \kappa_J (\text{SII} - 1)\big)$.

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `human_sagittal_suture_specimen.jpg` | Superior view photograph of an adult *Homo sapiens* calvaria showing the fractal interdigitated sagittal suture along the midline, coronal suture, and lambdoid suture (Wikimedia Commons `File:Sagittal suture 2.jpg`, CC BY-SA 3.0, 960×720 px) | `b3c411a6a47137708e4922c1b0f0381b67a3c1e885dc4d016fb0f7bb2f4eaf13` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 020 | `226c072307597439c420ed63e072337b85a3f7589d616e9fdcb015f37426d5d6` |
| `paper_citations.bib` | BibTeX entries for Miura et al. (2009), Jaslow (1990), Khonsari et al. (2013), and Zollikofer & Weissmann (2011) | `bffc86bbbd2dd3d64b987c843a02f10519e40cfc84375c288037b982b5770d46` |
| `allometric_cranial_sutures_reference.json` | Canonical Kuramoto–Sivashinsky / KPZ equations, 5 developmental/pathological suture regimes, and SHA-256 locks | verified at runtime |
