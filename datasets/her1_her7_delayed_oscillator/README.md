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
   T_{\text{tot}} = T_m + T_p > T_{\text{crit}} = 7.54\text{ min}, \qquad
   T_{\text{osc}} \approx 2\left(T_m + T_p + t_{1/2,p} + t_{1/2,m}\right) \approx 30.0\text{ min}
   \]
   Sustained limit-cycle oscillations emerge only when \(T_m + T_p > T_{\text{crit}}\) and when mRNA/protein decay rates \(b, c\) are sufficiently rapid compared to the delay. Deleting introns (\(T_{\text{tot}} < T_{\text{crit}}\)) or blocking translation with cycloheximide (\(a = 0 \implies p^* = 0, m^* = k/c\)) abolishes oscillations.
   *Pass 3 corrections:* \(T_{\text{crit}} = 7.54\) min is the value computed from the characteristic equation around the fixed point for the canonical parameters (the "\(\approx 4.62\) min" quoted in v1/v2 was not derived from that linearization and has been retired). The Box 1 law is an approximation: the RK4 DDE gives \(T_{\text{osc}} = 33.6\) min (12% above 30.0 min). In the 5-regime table, `predicted_period_min` are *tabulated* Box 1 estimates, and the "pure *her7* fast loop" regime (\(T_{\text{tot}} = 6.3\) min) is Hopf-subcritical under the computed threshold, so the model predicts a damped clock there; the script reports tabulated, Box-1-computed and DDE periods side by side (model-computed Pearson \(r = 0.79\) vs the v2 tabulated-vs-tabulated \(r = 0.9999\)).
3. **Intercellular DeltaC–Notch Delayed Synchronization vs *bea*/*des*/*aei* Desynchronization (Figs. 5–6, pp. 1403–1406):**
   In wild-type embryos (\(\kappa_{\text{Notch}} > 0\)), delayed intercellular DeltaC–Notch signaling locks neighboring PSM oscillators in phase (\(R_{\text{sync}} \approx 0.96\)). In Notch pathway mutants (*beamter*/*deltaC*, *deadly seven*/*notch1a*, *after eight*/*deltaD*; \(\kappa_{\text{Notch}} = 0\)), individual cells continue oscillating autonomously at \(T_{\text{osc}} \approx 30\text{ min}\), but stochastic copy-number fluctuations progressively desynchronize the tissue over cycles 1–6 (\(R_{\text{sync}}\) falls below \(0.25\) after cycle 6), preserving the first 5–7 anterior somites before posterior segmentation breaks down into "salt-and-pepper" defects.

4. **(Pass 3) 2D Multicellular DDE Lattice with Delayed DeltaC Translation & NICD Cleavage Kinetics (Lewis 2003 Figs. 5–6 coupled-cell model):**
   Each cell of an \(8 \times 24\) PSM sheet carries 7 delayed variables — *her1* mRNA/protein (\(T_{m1} = 9.5,\ T_{p1} = 2.8\) min), *her7* mRNA/protein (\(T_{m7} = 4.6,\ T_{p7} = 1.7\) min), *deltaC* mRNA/protein (\(T_{mD} = 16.0,\ T_{pD} = 20.5\) min) and NICD:
   \[
   \frac{dN_i}{dt} = k_N \frac{\bar D_i(t - T_N)}{K_D + \bar D_i(t - T_N)} - b_N N_i, \qquad
   G_N(N) = (1 - \epsilon_N) + \epsilon_N \frac{N^h}{N_0^h + N^h}
   \]
   where \(\bar D_i\) is the mean DeltaC protein of the 4 von Neumann neighbours and \(G_N\) gates *her1*/*her7* transcription (\(\epsilon_N = 0\) in *bea*/*des*/*aei*). Langevin molecular noise acts on the mRNA species. Section `lattice_model_parameters` of the reference JSON pins every parameter (asserted against the script by `--self-test`). The Notch-mutant lattice keeps 5–7 anterior cycles coherent (seeds 2003/7/11 → 7/7/5); the wild-type lattice keeps all 14 but its column synchrony decays from 0.98 to ≈0.6, i.e. the reduced coupling only partly compensates heterogeneity.
5. **(Pass 3) Schröter et al. (2012) Her1/Her7/Hes6 Dimer Truth Table on the AP Hes6 Gradient (PLoS Biol. Figs. 2–3, 5–7):**
   \[
   \text{Rep}_i = \frac{[\text{Her1:Her1}]_i}{K_{11}} + \frac{[\text{Her7:Hes6}]_i}{K_{76}}, \qquad [A{:}B] = A \cdot B
   \]
   with Her1:Her7, Her1:Hes6, Her7:Her7 and Hes6:Hes6 non-binding (`schroter_2012_dimer_truth_table.dna_binding`). Hes6 is non-oscillating and graded posterior-high (\(4 \to 40\) molecules). Posterior cells cycle through the short-delay Her7:Hes6 loop (faster), anterior cells through the long-delay Her1 homodimer loop (slower): single-cell period 41.2 → 36.4 min for Hes6 0 → 90, lattice gradient 40.8 → 38.6 min, *hes6*-null tissue uniform at 41.4 min (= her1-only DDE). Caveats: the truth table is hand-encoded from the paper's qualitative EMSA/Y2H summary; \(K_{11}, K_{76}\), the Hes6 levels and the NICD kinetics are reduced parameters of this repository; Hes6 ≳ 80 quenches the oscillation because the *her7* branch alone is Hopf-subcritical, whereas real posterior PSM oscillates robustly.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`ebe3f2cee38772dd464724c736e6bf735dc125004d55cd7f5be8ba91301ffffa`)
- `her1_her7_delayed_oscillator_reference.json` — Formal DDE parameters, Hopf bifurcation locus, 5-regime perturbation table (with Pass 3 caveats), Schröter (2012) dimer truth table and 2D lattice parameters (`7e974cd678ea54895969c98ea0ed9943190ae98beae7ed24efb6a815deb22f88`, SHA-256-pinned in the script since Pass 3)
- `zebrafish_somites_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons). **Pass 3 disclosure:** the Commons source title identifies this as a *chick* (*Gallus gallus*) embryo, not *Danio rerio*; it is a cross-species proxy for vertebrate somite metamerism and the file name is kept only because its SHA-256 is pinned. Vendoring a zebrafish *her1* in situ / PSM image is an open Pass 4 item.
- `paper_citations.bib` — BibTeX citations for Lewis (2003), Giudicelli et al. (2007), Herrgen et al. (2010), and Schröter et al. (2012)
