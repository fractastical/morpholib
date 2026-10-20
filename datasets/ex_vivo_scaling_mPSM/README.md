# Ex Vivo Monolayer Presomitic Mesoderm (mPSM) Phase-Gradient Scaling Dataset (`datasets/ex_vivo_scaling_mPSM/`)

Reference parameters, formal phase-gradient scaling laws, multi-size explant scaling tables, and biological specimen photography for **Slot 026 (`slot-026-ex-vivo-scaling-mPSM`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Lauschke, V. M., Tsiairis, C. D., François, P. & Aulehla, A. (2013).** *"Scaling of embryonic patterning based on phase-gradient encoding."* *Nature* 493(7430):101–105. DOI: `10.1038/nature11804`
- **Aulehla, A. et al. (2008).** *"A \(\beta\)-catenin gradient links the clock and wavefront systems in mouse embryo segmentation."* *Nature Cell Biology* 10(2):186–193. DOI: `10.1038/ncb1679`
- **Tsiairis, C. D. & Aulehla, A. (2016).** *"Self-organization of embryonic genetic oscillators into spatiotemporal wave patterns."* *Cell* 164(4):656–667. DOI: `10.1016/j.cell.2016.01.028`
- **Beaupeux, M. & François, P. (2016).** *"Positional information from oscillatory phase shifts: insights from in silico evolution."* *Physical Biology* 13(3):036009. DOI: `10.1088/1478-3975/13/3/036009`
- **Kuramoto, Y. (1984).** *Chemical Oscillations, Waves, and Turbulence.* Springer Series in Synergetics 19. DOI: `10.1007/978-3-642-69689-3` (Pass 3: target patterns from pacemakers)
- **Dubrulle, J. & Pourquié, O. (2004).** *"fgf8 mRNA decay establishes a gradient that couples axial elongation to patterning in the vertebrate embryo."* *Nature* 427(6973):419–422. DOI: `10.1038/nature02216` (Pass 3: fixed-length decay-gradient control)

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

## Pass 3 (audit v3) Mechanistic Extensions

Parameters and expected default results are recorded under `pass3_mechanistic_parameters` in `ex_vivo_scaling_mPSM_reference.json`; the formal statements are Laws 4–6 of `morphoeval/lean/loops/Loop_026_ex_vivo_scaling_mPSM.lean`.

4. **FGF/Wnt source–sink signalling gradient on the shrinking disc** (Lauschke et al. 2013 Fig. 4; Aulehla et al. 2008):
   \[
   D W'' - kW = 0,\quad W(0)=W_0,\ W(R(t))=0 \;\Rightarrow\; W(r,t) = W_0\,\frac{\sinh((R(t)-r)/\ell_k)}{\sinh(R(t)/\ell_k)},\qquad \ell_k \gg R \Rightarrow W = W_0\Big(1-\frac{r}{R(t)}\Big),\ \ell_{1/2} = \tfrac{R(t)}{2}
   \]
   The half-maximum length scales with the tissue (\(\partial\ln\ell_{1/2}/\partial\ln R = 1\)); the pure decay gradient \(W_0 e^{-r/\ell_k}\) (Dubrulle & Pourquié 2004) has \(\ell_{1/2} = \ell_k\ln 2\) (exponent 0) and is the mechanistic non-scaling control.

5. **Dynamic wavelength adaptation of the Notch phase wave** (Beaupeux & François 2016):
   \[
   \frac{d\lambda}{dt} = -\frac{\lambda - c_\lambda\,\ell_{1/2}(t)}{\tau_\lambda},\qquad \Phi(t) = \frac{2\pi R(t)}{\lambda(t)},\qquad S_n = R(t_n)\Big[1-\big(1-\delta\phi_{\rm seg}/\Phi(t_n)\big)^{1/p}\Big],\ \ \delta\phi_{\rm seg} = \Phi_0\big[1-(1-\alpha)^p\big]
   \]
   With \(\tau_\lambda = T_{\rm osc}/4\) the span stays within 2.5 % of \(2\pi\) and \(\alpha = S_n/R(t_n) = 0.110\) emerges; with \(\tau_\lambda\to\infty\) it drifts to 0.56 by cycle 8, and the fixed-length gradient yields constant 39.5 µm segments.

6. **Disordered cell-reaggregation lattice** (Tsiairis & Aulehla 2016; Kuramoto 1984): 1577 jittered-hexagonal cells in a 250 µm disc, Kuramoto–Sakaguchi coupling \(\dot\theta_i = \omega_i + (K/n_i)\sum_j[\sin(\theta_j-\theta_i-a)+\sin a]\) with spatially correlated frequency heterogeneity \(\omega_i = \omega_0(1+\eta\xi_i)\). Multiple target-wave pacemakers emerge de novo (median 4 in cycles 1–3) and consolidate onto the local frequency maximum (1 in cycles 12–14); \(K=0\) stays incoherent and \(\eta=0\) synchronises globally with no pacemaker.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`ed97fcae27d8072b2f22765258f3afa717f6fae7f63ac2ec67b5ec78ebceaaf9`)
- `ex_vivo_scaling_mPSM_reference.json` — Formal phase-gradient scaling equations, canonical E10.5 mouse LuVeLu mPSM parameters, and 4-explant scaling series (\(R_0 \in \{150, 250, 350, 450\}\ \mu\text{m}\))
- `mouse_mpsm_somites_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Lauschke et al. (2013), Aulehla et al. (2008), Tsiairis & Aulehla (2016), Beaupeux & François (2016), Kuramoto (1984), and Dubrulle & Pourquié (2004)
