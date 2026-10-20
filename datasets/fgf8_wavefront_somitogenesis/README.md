# FGF8 mRNA Decay Wavefront & Determination Front Positioning Dataset (`datasets/fgf8_wavefront_somitogenesis/`)

Reference parameters, formal advection-decay PDE equations, experimental perturbation tables, and biological specimen photography for **Slot 024 (`slot-024-fgf8-wavefront-somitogenesis`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Dubrulle, J., McGrew, M. J. & Pourquié, O. (2001).** *"FGF signaling controls somite boundary position and regulates segmentation clock control of spatiotemporal Hox gene activation."* *Cell* 106(2):219–232. DOI: `10.1016/S0092-8674(01)00437-8`
- **Dubrulle, J. & Pourquié, O. (2004).** *"fgf8 mRNA decay establishes a gradient that couples axial elongation to patterning in the vertebrate embryo."* *Nature* 427(6973):419–422. DOI: `10.1038/nature02332`

## Governing Mathematical Laws

1. **Cell-Autonomous *fgf8* mRNA Decay Advection-Reaction PDE (Tailbud Frame \(x \ge 0\)):**
   \[
   \frac{\partial m}{\partial t} + v_g \frac{\partial m}{\partial x} = -k_d m(x,t) \implies m(x) = m_0 \exp\!\left(-\frac{k_d x}{v_g}\right) = m_0 \exp\!\left(-\frac{x}{\lambda_m}\right), \quad \lambda_m = \frac{v_g}{k_d}
   \]
   Because *fgf8* transcription is restricted to the posterior tailbud/primitive streak progenitor zone (\(x = 0\)) and ceases as cells enter the presomitic mesoderm (PSM), posterior axis elongation at velocity \(v_g\) advects cells anteriorly while intracellular mRNA decays at rate \(k_d\), converting temporal mRNA decay into a spatial exponential gradient.
2. **Determination Front Positioning Law (Level-II / Level-III PSM Boundary \(x^*\)):**
   \[
   m(x^*) = m_{\text{crit}} \implies x^* = \frac{v_g}{k_d} \ln\!\left(\frac{m_0}{m_{\text{crit}}}\right)
   \]
3. **Somite Scaling & Ectopic FGF8 Bead / SU5402 Perturbation Conservation Law:**
   \[
   S_n = \left(v_g - \frac{dx^*}{dt}\right) T_c = S_0 - \Delta x^*_n, \qquad S_{\text{anterior}} + S_{\text{posterior}} = (S_0 - \delta x^*) + (S_0 + \delta x^*) = 2 S_0
   \]
   Implanting an ectopic FGF8-soaked bead in the PSM transiently shifts \(x^*\) anteriorly (\(\Delta x^* > 0\)), producing compressed small somites anterior to the bead followed by a compensatory large somite posterior to the bead once the wavefront sweeps past the bead; conversely, FGFR1 inhibition via SU5402 shifts \(x^*\) posteriorly (\(\Delta x^* < 0\)), producing enlarged somites.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`a38a686592f1ffd0bb211b7de4f084d692b0bf29b88ddcd9231846cc99f7d392`)
- `fgf8_wavefront_somitogenesis_reference.json` — Formal advection-decay parameters and Dubrulle et al. (2001, 2004) 5-regime perturbation dataset
- `chick_somite_psm_specimen.jpg` — Dorsal scanning electron micrograph of a vertebrate/avian embryo showing the central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Dubrulle et al. (2001), Dubrulle & Pourquié (2004), Baker et al. (2006), and Goldbeter et al. (2007)
