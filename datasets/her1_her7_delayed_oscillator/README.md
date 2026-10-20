# Zebrafish *her1*/*her7* Delayed Autoinhibitory Oscillator Dataset (`datasets/her1_her7_delayed_oscillator/`)

Reference parameters, delay differential equations (DDEs), Hopf bifurcation threshold, experimental perturbation regimes, and biological specimen photography for **Slot 022 (`slot-022-her1-her7-delayed-oscillator`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Lewis, J. (2003).** *"Autoinhibition with transcriptional delay: a simple mechanism for the zebrafish somitogenesis oscillator."* *Current Biology* 13(16):1398–1408. DOI: `10.1016/S0960-9822(03)00534-7`
- **Giudicelli, F., Özbudak, E. M., Wright, G. J. & Lewis, J. (2007).** *"Setting the tempo in development: an investigation of the zebrafish somite clock mechanism."* *PLoS Biology* 5(6):e150. DOI: `10.1371/journal.pbio.0050150`
- **Herrgen, L., Ares, S., Morelli, L. G., Schröter, C., Jülicher, F. & Oates, A. C. (2010).** *"Intercellular coupling regulates the period of the segmentation clock."* *Current Biology* 20(14):1244–1253. DOI: `10.1016/j.cub.2010.06.034`
- **Schröter, C., Ares, S., Morelli, L. G., Isakova, A., Koo, B.-K., Jülicher, F. & Oates, A. C. (2012).** *"Topology and dynamics of the zebrafish segmentation clock core circuit."* *PLoS Biology* 10(7):e1001364. DOI: `10.1371/journal.pbio.1001364`

## Governing Mathematical Laws

1. **Lewis (2003) Single-Cell *her1*/*her7* Autoinhibitory Delay Differential Equations (Eqs. 1–2, p. 1399):**
   \[
   \frac{dp(t)}{dt} = a \, m(t - T_p) - b \, p(t), \qquad
   \frac{dm(t)}{dt} = \frac{k}{1 + \left(\frac{p(t - T_m)}{p_0}\right)^n} - c \, m(t)
   \]
   where \(m(t)\) is *her1*/*her7* mRNA molecules per cell, \(p(t)\) is Her1/Her7 repressor protein molecules per cell, \(T_m\) is transcriptional/splicing/export delay, \(T_p\) is translational delay, \(a = 4.5\text{ min}^{-1}\), \(b = c = 0.23\text{ min}^{-1}\) (\(t_{1/2} \approx 3.01\text{ min}\)), \(k = 33\text{ min}^{-1}\), \(p_0 = 40\), and \(n = 2\). Under the time-shift transformation \(\tilde{p}(t) = p(t + T_p)\), the system dynamics and characteristic equation depend solely on the total loop delay \(T_{\text{tot}} = T_m + T_p\).
2. **Hopf Bifurcation Critical Delay Threshold & Oscillation Period Law (Box 1, p. 1400):**
   \[
   T_{\text{tot}} = T_m + T_p > T_{\text{crit}} \approx 4.62\text{ min}, \qquad
   T_{\text{osc}} \approx 2\left(T_m + T_p + t_{1/2,p} + t_{1/2,m}\right) \approx 30.0\text{ min}
   \]
   Sustained limit-cycle oscillations emerge only when \(T_m + T_p > T_{\text{crit}}\) and when mRNA/protein decay rates \(b, c\) are sufficiently rapid compared to the delay. Deleting introns (\(T_{\text{tot}} < T_{\text{crit}}\)) or blocking translation with cycloheximide (\(a = 0 \implies p^* = 0, m^* = k/c\)) abolishes oscillations.
3. **Intercellular DeltaC–Notch Delayed Synchronization vs *bea*/*des*/*aei* Desynchronization (Figs. 5–6, pp. 1403–1406):**
   In wild-type embryos (\(\kappa_{\text{Notch}} > 0\)), delayed intercellular DeltaC–Notch signaling locks neighboring PSM oscillators in phase (\(R_{\text{sync}} \approx 0.96\)). In Notch pathway mutants (*beamter*/*deltaC*, *deadly seven*/*notch1a*, *after eight*/*deltaD*; \(\kappa_{\text{Notch}} = 0\)), individual cells continue oscillating autonomously at \(T_{\text{osc}} \approx 30\text{ min}\), but stochastic copy-number fluctuations progressively desynchronize the tissue over cycles 1–6 (\(R_{\text{sync}}\) falls below \(0.25\) after cycle 6), preserving the first 5–7 anterior somites before posterior segmentation breaks down into "salt-and-pepper" defects.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`ebe3f2cee38772dd464724c736e6bf735dc125004d55cd7f5be8ba91301ffffa`)
- `her1_her7_delayed_oscillator_reference.json` — Formal DDE parameters, Hopf bifurcation locus, and 5-regime perturbation table
- `zebrafish_somites_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Lewis (2003), Giudicelli et al. (2007), Herrgen et al. (2010), and Schröter et al. (2012)
