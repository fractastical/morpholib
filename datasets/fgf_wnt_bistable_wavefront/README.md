# Anterior RA vs Posterior FGF8/Wnt Bistable Determination-Front Dataset (`datasets/fgf_wnt_bistable_wavefront/`)

Reference parameters, reaction-diffusion PDEs, saddle-node hysteresis equations, experimental perturbation tables, and biological specimen photography for **Slot 025 (`slot-025-fgf-wnt-bistable-wavefront`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Goldbeter, A., Gonze, D. & Pourquié, O. (2007).** *"Sharp developmental thresholds defined through bistability by antagonistic gradients of retinoic acid and FGF signaling."* *Dev. Cell* / *Dev. Dyn.* 236(6):1495–1508. DOI: `10.1016/j.devcel.2007.05.015` / `10.1002/dvdy.21193`
- **Baker, R. E., Schnell, S. & Maini, P. K. (2006).** *"A mathematical investigation of a clock and wavefront model for somitogenesis."* *J. Theor. Biol.* 239(4):413–429. DOI: `10.1016/j.jtbi.2005.07.019`
- **Diez del Corral, R. et al. (2003).** *"Opposing FGF and retinoid pathways control ventral neural pattern, neuronal differentiation, and segmentation during body axis extension."* *Neuron* 40(1):65–79. DOI: `10.1016/S0896-6273(03)00565-8`
- **Vermot, J. & Pourquié, O. (2005).** *"Retinoic acid coordinates somitogenesis and left-right patterning in vertebrate embryos."* *Nature* 435(7039):215–220. DOI: `10.1038/nature03488`
- **Lewis, J. (2003).** *"Autoinhibition with transcriptional delay: a simple mechanism for the zebrafish somitogenesis oscillator."* *Current Biology* 13(16):1398–1408. DOI: `10.1016/S0960-9822(03)00534-7`
- **Palmeirim, I., Henrique, D., Ish-Horowicz, D. & Pourquié, O. (1997).** *"Avian hairy gene expression identifies a molecular clock linked to vertebrate segmentation and somitogenesis."* *Cell* 91(5):639–648. DOI: `10.1016/S0092-8674(00)80451-1`
- **Dubrulle, J., McGrew, M. J. & Pourquié, O. (2001).** *"FGF signaling controls somite boundary position and regulates segmentation clock control of spatiotemporal Hox gene activation."* *Cell* 106(2):219–232. DOI: `10.1016/S0092-8674(01)00437-8`
- **Vermot, J., Gallego Llamas, J., Fraulob, V., Niederreither, K., Chambon, P. & Dollé, P. (2005).** *"Retinoic acid controls the bilateral symmetry of somite formation in the mouse embryo."* *Science* 308(5721):563–566. DOI: `10.1126/science.1108363`

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

4. **Lewis (2003) Delayed Her1/Hes7 Autoinhibition Clock Lattice & Clock-Gated Saltatory Arrest (Pass 3):**
   \[
   \frac{dm_j}{dt} = \frac{k\,N_j(t)}{1 + (p_j(t-T_m)/p_0)^2} - c\,m_j, \qquad \frac{dp_j}{dt} = a\,m_j(t-T_p) - b\,p_j, \qquad T_{\text{osc}} \approx 2\left(T_m + T_p + \tfrac{1}{b} + \tfrac{1}{c}\right)
   \]
   with delayed saturating Notch–Delta input \(N_j(t) = 1 - \epsilon_N + \epsilon_N \langle m_{\text{nb}}(t-T_N)\rangle / (K_D + \langle m_{\text{nb}}(t-T_N)\rangle)\) from adjacent cells. Chick preset \(T_m = 31.5\), \(T_p = 2.8\), \(a = 4.5\), \(b = c = 0.23\), \(k = 33\), \(p_0 = 40\) gives \(T_{\text{osc}} \approx 86\text{–}92\text{ min}\) (observed 90 min). Cells advected through the bistability (clock-competence) window \([x(M_2), x(M_1)] = [149.4, 571.4]\ \mu\text{m}\) (2.8 somites \(\ge 1\), Goldbeter et al. 2007 Fig. 10) arrest only at a Her protein peak, so arrest is saltatory: cohorts of \(n\) cells per cycle, segment \(S = n a\). Notch ablation (\(\epsilon_N = 0\)) collapses nearest-neighbour coherence \(R\) from 0.84 to 0.03 and the saltatory index from 0.87 to 0.42; \(T_N = T_{\text{osc}}/2\) produces Lewis's anti-phase locking.

5. **Vermot & Pourquié (2005) RA-Buffered Left–Right Symmetry on 2D Bilateral Paraxial Mesoderm Strips (Pass 3):**
   \[
   b_{\text{eff}} = b_{LR}\frac{K_{\text{buf}}^h}{K_{\text{buf}}^h + R_{\text{ant}}^h}, \qquad v_{\text{sc},L/R} = v_{\text{sc}}\,(1 \mp b_{\text{eff}}), \qquad \Delta x_{LR} = x_{\text{front},L} - x_{\text{front},R}
   \]
   The 1D PDE is extended to two paraxial strips (\(\pm 80\text{–}250\ \mu\text{m}\)) flanking a neural tube on a \(1500 \times 500\ \mu\text{m}\) domain. Anterior RA (\(R_{\text{ant}} = 2.5\text{ nM}\) WT) buffers the lateralizing bias so \(\Delta x_{LR} \approx 6\ \mu\text{m}\) (0.04 somites), whereas VAD/Raldh2⁻/⁻ (\(R_{\text{ant}} = 0.8\text{ nM}\)) yields \(\Delta x_{LR} \approx 150\ \mu\text{m}\) = one somite with the left side delayed; situs inversus flips the delayed side. Four regimes (WT, VAD chick, Raldh2⁻/⁻ mouse, situs inversus) are reproduced with a maximum error of 0.04 somites.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`783faa9ee6b3db0ca97883bcfa0afa191e76130bc5ce2b5d82630afc19e36e48`)
- `fgf_wnt_bistable_wavefront_reference.json` — Formal RA/FGF8/CYP26A1 bistable PDE parameters, saddle-node hysteresis thresholds, and 5-regime perturbation dataset
- `chick_ra_fgf_somites_specimen.jpg` — Dorsal scanning electron micrograph of a vertebrate/avian embryo showing the central neural tube, anterior RALDH2-expressing somites, determination front, and posterior PSM (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `fgf_wnt_clock_lr_reference.json` — Pass 3 Lewis (2003) delayed-oscillator parameters and period table, bistability competence window, Vermot & Pourquié (2005) RA-buffered LR law and 4 LR regimes, bilateral geometry (`16deab8565d154cb27b8f2f5b956d579b9a703d4eba11b2d1b7b4057deaf3b45`)
- `paper_citations.bib` — BibTeX citations for Goldbeter et al. (2007), Baker et al. (2006), Diez del Corral et al. (2003), Vermot & Pourquié (2005), Lewis (2003), Palmeirim et al. (1997), Dubrulle et al. (2001), and Vermot et al. (2005)
