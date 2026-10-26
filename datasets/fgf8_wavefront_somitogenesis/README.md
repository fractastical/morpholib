# FGF8 mRNA Decay Wavefront & Determination Front Positioning Dataset (`datasets/fgf8_wavefront_somitogenesis/`)

Reference parameters, formal advection-decay PDE equations, experimental perturbation tables, and biological specimen photography for **Slot 024 (`slot-024-fgf8-wavefront-somitogenesis`)** of the Project Delphi & MorphoEval Campaign, reproducing:

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

## Pass 3 Additions (2026-10-19, remediation of audit v2)

4. **Goldbeter, Gonze & Pourquié (2007) RA / FGF8 mutual-inhibition bistable switch** (`pass3_goldbeter_2007_ra_fgf8_bistable_switch` in the reference JSON):
   \[
   \dot F = v_F\, m(x)\,\frac{K_I^n}{K_I^n + RA^n} - k_F F,\quad
   \dot X = v_X\,\frac{F^n}{K_A^n + F^n} - k_X X,\quad
   \dot{RA} = v_R - k_R\,RA - k_C\,\frac{X\,RA}{K_C + RA}
   \]
   The double-negative loop RA ⊣ FGF → Cyp26 ⊣ RA is bistable for tailbud-derived *fgf8* mRNA input \(m \in [m_{SN,lo}, m_{SN,hi}] = [0.184, 0.546]\); a PSM cell advected from the tailbud stays on the high-FGF branch until the saddle-node at \(x_{SN} = \lambda_m \ln(m_0/m_{SN,lo}) = 610\,\mu\)m, giving a hysteretic front (cell-trajectory switch at 720 µm, 90→10 % transition width 198 µm vs 792 µm for the graded threshold). **The dimensionless parameter values are this repository's bistable-regime choice, not the published table.**
5. **Cell-autonomous *fgf8* mRNA dilution by tailbud progenitor division** (`pass3_2d_cell_strip_division_dilution`): one division equals one half-life of decay (\(\log_2 m = \log_2 m_0 - D - n\)); the periodic exported level is \(m_{exit} = (k_s/k_d)(1-e^{-k_d T_{div}})/(2-e^{-k_d T_{div}})\) and removing division at fixed transcription shifts the front anteriorly by \(\Delta x_{dil} = \lambda_m \ln[(2-e^{-k_d T_{div}})/(1-e^{-k_d T_{div}})] = 271\,\mu\)m at \(T_{div}=480\) min. Realised in a 2D cell-based strip (8 rows × 10 µm cells, self-renewing progenitor pool, lognormal cell cycle, binomial-like mRNA partitioning, juxtacrine FGF8 smoothing σ = 25 µm, per-row somite boundaries at clock ticks). Known simplifications: PSM cells do not divide, all elongation comes from progenitor division (no cell-motility contribution, Bénazéraf et al. 2010), and rows are coupled only through FGF8 smoothing.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`a38a686592f1ffd0bb211b7de4f084d692b0bf29b88ddcd9231846cc99f7d392`)
- `fgf8_wavefront_somitogenesis_reference.json` — Formal advection-decay parameters, Dubrulle et al. (2001, 2004) 5-regime perturbation dataset, and (Pass 3) the Goldbeter (2007) switch parameters and 2D cell-strip reference outputs (not SHA-pinned at runtime; the specimen and scout manifest are)
- `chick_somite_psm_specimen.jpg` — Dorsal scanning electron micrograph of a vertebrate/avian embryo showing the central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Dubrulle et al. (2001), Dubrulle & Pourquié (2004), Baker et al. (2006), Goldbeter et al. (2007), and (Pass 3) Bénazéraf et al. (2010)
