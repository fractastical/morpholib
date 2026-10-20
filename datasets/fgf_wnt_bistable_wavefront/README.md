# Anterior RA vs Posterior FGF8/Wnt Bistable Determination-Front Dataset (`datasets/fgf_wnt_bistable_wavefront/`)

Reference parameters, reaction-diffusion PDEs, saddle-node hysteresis equations, experimental perturbation tables, and biological specimen photography for **Slot 025 (`slot-025-fgf-wnt-bistable-wavefront`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Goldbeter, A., Gonze, D. & Pourquié, O. (2007).** *"Sharp developmental thresholds defined through bistability by antagonistic gradients of retinoic acid and FGF signaling."* *Dev. Cell* / *Dev. Dyn.* 236(6):1495–1508. DOI: `10.1016/j.devcel.2007.05.015` / `10.1002/dvdy.21193`
- **Baker, R. E., Schnell, S. & Maini, P. K. (2006).** *"A mathematical investigation of a clock and wavefront model for somitogenesis."* *J. Theor. Biol.* 239(4):413–429. DOI: `10.1016/j.jtbi.2005.07.019`
- **Diez del Corral, R. et al. (2003).** *"Opposing FGF and retinoid pathways control ventral neural pattern, neuronal differentiation, and segmentation during body axis extension."* *Neuron* 40(1):65–79. DOI: `10.1016/S0896-6273(03)00565-8`
- **Vermot, J. & Pourquié, O. (2005).** *"Retinoic acid coordinates somitogenesis and left-right patterning in vertebrate embryos."* *Nature* 435(7039):215–220. DOI: `10.1038/nature03488`

## Governing Mathematical Laws

1. **Mutual Cross-Inhibition + CYP26A1 RA-Degradation Reaction-Diffusion System:**
   \[
   \frac{\partial R}{\partial t} = D_R \frac{\partial^2 R}{\partial x^2} + v_{\text{sc}}(x) - k_{d1} C(F) R - k_{d2} R, \qquad C(F) = \frac{V_c F^m}{K_a^m + F^m}
   \]
   \[
   \frac{\partial F}{\partial t} = D_F \frac{\partial^2 F}{\partial x^2} + k_s M(x) \frac{K_I^n}{K_I^n + R^n} - k_{dF} F, \qquad M(x) = M_0 \exp\!\left(-\frac{x}{\lambda_M}\right)
   \]
   Posterior FGF8/Wnt signaling \(F\) induces CYP26A1 (\(C(F)\)) to enzymatically degrade Retinoic Acid \(R\), while anterior RALDH2-derived Retinoic Acid \(R\) inhibits `fgf8` mRNA translation/signaling with Hill cooperativity \(n\).

2. **Analytical Inverse Hysteresis S-Curve & Saddle-Node Bifurcation Window \([M_1, M_2]\):**
   \[
   R^*(F) = \frac{v_{\text{sc}}}{k_{d2} + k_{d1} C(F)}, \qquad M(F) = \frac{k_{dF} F}{k_s} \left[1 + \left(\frac{R^*(F)}{K_I}\right)^n\right]
   \]
   For canonical chick PSM parameters (\(v_{\text{sc}} = 0.25\text{ nM/min}\), \(k_{d1} = 1.0\text{ min}^{-1}\), \(k_{d2} = 0.10\text{ min}^{-1}\), \(K_a = K_I = 0.2\text{ nM}\), \(n = m = 2\)), the system exhibits a bistable hysteresis window between lower saddle-node \(M_1 = 1.0254\text{ nM}\) and upper saddle-node \(M_2 = 3.3038\text{ nM}\) (\(\Delta M_{\text{bistable}} = 2.2784\text{ nM}\)).

3. **Sharp Determination-Front Switch (\(\Delta x_{\text{front}} \ll \lambda_M\)) & VAD / RA / SU5402 Somite Perturbations:**
   As tailbud cells advect anteriorly along the PSM from \(M(0) = M_0 = 5.0\text{ nM} > M_2\), they remain locked in the high-FGF8 / low-RA posterior state until \(M(x)\) crosses the lower saddle-node threshold \(M_1\) at \(x_{\text{front}} = \lambda_M \ln(M_0 / M_1) \approx 571.6\text{ }\mu\text{m}\), triggering an all-or-none bistable jump (\(\Delta x_{\text{front}} \approx 24.0\text{ }\mu\text{m} \ll \lambda_M = 360.67\text{ }\mu\text{m}\)). Vitamin A Deficiency (VAD, reduced \(v_{\text{sc}}\)) lowers \(M_1\) and shifts the front anteriorly (forming smaller somites, \(S_{\text{VAD}} = 105\text{ }\mu\text{m}\)), whereas exogenous RA or SU5402 FGFR1 inhibition shifts the front posteriorly (forming enlarged somites, \(S_{\text{RA}} = 195\text{ }\mu\text{m}\), \(S_{\text{SU5402}} = 215\text{ }\mu\text{m}\)).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`783faa9ee6b3db0ca97883bcfa0afa191e76130bc5ce2b5d82630afc19e36e48`)
- `fgf_wnt_bistable_wavefront_reference.json` — Formal RA/FGF8/CYP26A1 bistable PDE parameters, saddle-node hysteresis thresholds, and 5-regime perturbation dataset
- `chick_ra_fgf_somites_specimen.jpg` — Dorsal scanning electron micrograph of a vertebrate/avian embryo showing the central neural tube, anterior RALDH2-expressing somites, determination front, and posterior PSM (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Goldbeter et al. (2007), Baker et al. (2006), Diez del Corral et al. (2003), and Vermot & Pourquié (2005)
