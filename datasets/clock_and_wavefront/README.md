# Clock-and-Wavefront Somite Segmentation Dataset (`datasets/clock_and_wavefront/`)

Reference parameters, formal equations, and biological specimen photography for **Slot 021 (`slot-021-clock-and-wavefront`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing **Cooke & Zeeman (1976)** *"A clock and wavefront model for control of the number of repeated structures during animal morphogenesis"* (*J. Theor. Biol.* 58(2):455–476, DOI: `10.1016/S0022-5193(76)80131-2`).

## Governing Equations

1. **Somite Wavelength Law:**
   \[
   S = v \, T_c
   \]
   where \(v\) is the speed of the anteriorly-to-posteriorly regressing maturation wavefront and \(T_c\) is the autonomous intracellular segmentation-clock period.
2. **Somite Count & Size-Invariance Scaling Law:**
   \[
   N = \frac{L}{S} = \frac{L}{v \, T_c}, \qquad v = \frac{L}{T_{\text{total}}} \implies N = \frac{T_{\text{total}}}{T_c}
   \]
   explaining Cooke's (1975) experimental discovery in *Xenopus laevis* that surgically reduced miniature embryos (\(L' = \alpha L\) with scaled wavefront speed \(v' = \alpha v\)) form the exact same number of somites \(N\) as normal embryos, with each somite scaled to length \(S' = \alpha S\).
3. **Thom–Zeeman Cusp Catastrophe Wavefront Dynamics:**
   \[
   \tau_\chi \frac{\partial \chi(x,t)}{\partial t} = -\bigl(\chi^3 - a(x,t)\,\chi - b_{\text{osc}}(x,t)\bigr)
   \]
   where \(a(x,t) = \tanh\!\bigl((v t - x)/\sigma_w\bigr)\) is the regressing positional wavefront bifurcation parameter and \(b_{\text{osc}}(x,t) = b_0 + A_c \cos(2\pi t / T_c - k_\phi x)\) is the intracellular segmentation clock gating catastrophic epithelial somite cleft formation.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`0beabd3277c705e62b894794a2448ec9572377059c72a185a6148e2f17f0b327`)
- `clock_and_wavefront_reference.json` — Formal parameterization and Cooke (1975, 1976) empirical *Xenopus* somite size-invariance table
- `somite_embryo_specimen.jpg` — Dorsal scanning electron micrograph of a vertebrate embryo showing the central neural tube and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Cooke & Zeeman (1976), Cooke (1975), Palmeirim et al. (1997), and Goldbeter et al. (2007)
