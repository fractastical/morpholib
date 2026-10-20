# Human Pluripotent Stem-Cell Segmentation Clock & Metabolic Tempo Scaling Dataset (`datasets/human_somite_organoid_clock/`)

Reference parameters, formal *HES7* delay-differential oscillator equations, species-swap and metabolic perturbation tables, 6-species vertebrate segmentation clock allometry, and biological specimen photography for **Slot 028 (`slot-028-human-somite-organoid-clock`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Diaz-Cuadros, M. et al. (2020).** *"In vitro characterization of the human segmentation clock."* *Nature* 580(7801):113–118. DOI: `10.1038/s41586-019-1885-9`
- **Matsuda, M. et al. (2020).** *"Species-specific oscillation periods of human and mouse segmentation clocks are due to cell autonomous properties of HES7."* *Science* 369(6510):1450–1455. DOI: `10.1126/science.aba7668`
- **Diaz-Cuadros, M. et al. (2023).** *"Metabolic regulation of species-specific developmental rates."* *Nature* 613(7944):550–557. DOI: `10.1038/s41586-022-05574-4`
- **Yamanaka, Y. et al. (2023).** *"Reconstituting human somitogenesis in vitro."* *Nature* 614(7948):509–520. DOI: `10.1038/s41586-022-05649-2`

## Governing Mathematical Laws

1. **Delayed Autoinhibitory *HES7* Transcriptional-Translational Feedback Loop (TTFL) & Exact Tempo Scaling Invariance (`diaza-human-somite-organoid-clock-1`):**
   \[
   \frac{dm}{dt} = \frac{\alpha_m}{1 + \left(p(t - \tau_{\text{tx}})/K\right)^n} - \delta_m m(t), \qquad
   \frac{dp}{dt} = \alpha_p m(t - \tau_{\text{tl}}) - \delta_p p(t), \qquad \tau = \tau_{\text{tx}} + \tau_{\text{tl}}
   \]
   Under the uniform biochemical tempo rescaling \(t \mapsto \lambda t\), \(\tau \mapsto \lambda \tau\), \(\delta_{m,p} \mapsto \delta_{m,p}/\lambda\), \(\alpha_{m,p} \mapsto \alpha_{m,p}/\lambda\), the *HES7* DDE system is strictly invariant and the oscillation period scales linearly as \(T_{\text{osc}}(\lambda) = \lambda T_{\text{osc}}(1)\). Between mouse (\(T_{\text{mouse}} \approx 120\text{ min}\), \(\tau \approx 22\text{ min}\), \(t_{1/2,p} \approx 21\text{ min}\)) and human (\(T_{\text{human}} \approx 300\text{ min} = 5.0\text{ h}\), \(\tau \approx 51\text{ min}\), \(t_{1/2,p} \approx 40\text{ min}\)), the cell-autonomous tempo slowdown factor is \(\lambda_{\text{human/mouse}} \approx 2.50\).

2. **Intron Splicing Delay Decomposition & Intronless *Hes7* Oscillation Collapse (`diaza-human-somite-organoid-clock-2`):**
   \[
   \tau = \tau_{\text{intron}} + \tau_{\text{basal}}, \qquad \tau_{\Delta\text{intron}} = \tau_{\text{basal}} < \tau_{\text{crit}} < \tau_{\text{WT}}
   \]
   Deleting the three introns of *Hes7* eliminates the splicing delay \(\tau_{\text{intron}} > 0\), dropping total delay below the Hopf bifurcation threshold \(\tau_{\text{crit}}\) and abolishing sustained *HES7* oscillations.

3. **Mitochondrial \(\text{NAD}^+/\text{NADH}\) Metabolic Rate Scaling of Clock Period & Axioloid Somite Budding (`diaza-human-somite-organoid-clock-3`):**
   \[
   k_{\text{met}} \propto \left(\frac{\text{NAD}^+}{\text{NADH}}\right)^{\gamma}, \qquad
   \delta_{m,p}(k_{\text{met}}) = k_{\text{met}} \delta_{m,p}^{(0)}, \qquad
   T_{\text{osc}}(k_{\text{met}}) = \frac{T_{\text{osc}}^{(0)}}{k_{\text{met}}}, \qquad
   S_{\text{somite}} = v_{\text{wf}} T_{\text{osc}}
   \]
   Elevating cytosolic \(\text{NAD}^+/\text{NADH}\) via *LbNOX* accelerates global translation and *HES7* turnover, shortening the human clock period from \(300\text{ min}\) to \(\sim 228\text{ min}\), whereas Complex I/IV electron transport chain (ETC) inhibition (rotenone / sodium azide) slows the clock to \(\sim 402\text{ min}\).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`a3af01b1b295695480c2b1f3e2f4bd9f49a62674684e15ad8c1c4b2ef4053d54`)
- `human_somite_organoid_clock_reference.json` — Formal *HES7* DDE equations, mouse vs human parameters, Matsuda (2020) & Diaz-Cuadros (2020, 2023) perturbation series, and 6-species vertebrate clock allometry
- `human_somite_organoid_specimen.jpg` — Dorsal vertebrate embryo scanning electron micrograph showing central neural tube, presomitic mesoderm, and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Diaz-Cuadros et al. (2020, 2023), Matsuda et al. (2020), and Yamanaka et al. (2023)
