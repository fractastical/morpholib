# Ex Vivo Monolayer Presomitic Mesoderm (mPSM) Phase-Gradient Scaling Dataset (`datasets/ex_vivo_scaling_mPSM/`)

Reference parameters, formal phase-gradient scaling laws, multi-size explant scaling tables, and biological specimen photography for **Slot 026 (`slot-026-ex-vivo-scaling-mPSM`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Lauschke, V. M., Tsiairis, C. D., François, P. & Aulehla, A. (2013).** *"Scaling of embryonic patterning based on phase-gradient encoding."* *Nature* 493(7430):101–105. DOI: `10.1038/nature11804`
- **Aulehla, A. et al. (2008).** *"A \(\beta\)-catenin gradient links the clock and wavefront systems in mouse embryo segmentation."* *Nature Cell Biology* 10(2):186–193. DOI: `10.1038/ncb1679`
- **Tsiairis, C. D. & Aulehla, A. (2016).** *"Self-organization of embryonic genetic oscillators into spatiotemporal wave patterns."* *Cell* 164(4):656–667. DOI: `10.1016/j.cell.2016.01.028`
- **Beaupeux, M. & François, P. (2016).** *"Positional information from oscillatory phase shifts: insights from in silico evolution."* *Physical Biology* 13(3):036009. DOI: `10.1088/1478-3975/13/3/036009`

## Governing Mathematical Laws

1. **Scale-Invariant Phase-Gradient Encoding Across Residual mPSM Radius \(R(t)\):**
   \[
   \phi(r,t) = \omega_0 t - \Phi_0 \left(\frac{r}{R(t)}\right)^p, \qquad \Delta\phi_{\text{total}}(t) = \phi(0,t) - \phi(R(t),t) = \Phi_0 = 2\pi
   \]
   Across the active unsegmented monolayer presomitic mesoderm (mPSM) disc of shrinking radius \(R(t)\), the total Notch/LuVeLu (*Lfng*-Venus) oscillation phase span between the central posterior-like pacemaker (\(r=0\), period \(T_{\text{osc}} = 140\text{ min}\)) and the peripheral determination boundary (\(r = R(t)\)) remains invariant at \(\Phi_0 \approx 2\pi\text{ rad}\). Consequently, the spatial phase gradient \(\left|\partial\phi/\partial r\right| \sim \Phi_0 / R(t)\) steepens inversely with \(R(t)\), and the radial phase-wave velocity scales linearly with residual mPSM size:
   \[
   \lambda(t) = \frac{2\pi}{\Phi_0} R(t), \qquad v_{\text{wave}}(t) = \frac{\lambda(t)}{T_{\text{osc}}} = \frac{2\pi}{\Phi_0} \frac{R(t)}{T_{\text{osc}}}
   \]

2. **Linear Proportional Segment Scaling Law (`S_n = alpha * R(t_n)`):**
   \[
   S_n = \alpha \, R(t_n), \qquad \alpha \approx 0.11
   \]
   Each newly specified concentric segment \(S_n\) at oscillation cycle \(n\) is gated by a fixed fraction of the invariant phase gradient, so segment width \(S_n\) scales strictly linearly with residual mPSM radius \(R(t_n)\) with dimensionless proportionality constant \(\alpha \approx 0.11\) across a four-fold variation in explant size (\(R_0 \in [150, 450]\ \mu\text{m}\)).

3. **Geometric Progression of Residual mPSM Radius & Telescoping Segment Conservation:**
   \[
   R_{n+1} = R_n - S_n = (1 - \alpha) R_n = R_0 (1 - \alpha)^n, \qquad S_n = \alpha R_0 (1 - \alpha)^{n-1}, \qquad \sum_{k=1}^{N} S_k = R_0 - R_N
   \]

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`ed97fcae27d8072b2f22765258f3afa717f6fae7f63ac2ec67b5ec78ebceaaf9`)
- `ex_vivo_scaling_mPSM_reference.json` — Formal phase-gradient scaling equations, canonical E10.5 mouse LuVeLu mPSM parameters, and 4-explant scaling series (\(R_0 \in \{150, 250, 350, 450\}\ \mu\text{m}\))
- `mouse_mpsm_somites_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Lauschke et al. (2013), Aulehla et al. (2008), Tsiairis & Aulehla (2016), and Beaupeux & François (2016)
