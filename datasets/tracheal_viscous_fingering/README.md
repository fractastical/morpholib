# `datasets/tracheal_viscous_fingering/` — Nelson et al. (2017) *Development*, Kim et al. (2015) *Dev. Cell* & Unbekandt et al. (2008) *Mech. Dev.* Reference Package

Formal SHA-256-locked reference dataset and *Mus musculus* pseudoglandular embryonic lung branching specimen for `2015-sugimura-tracheal-viscous-fingering.py` (Slot 059 of the Project Delphi & MorphoEval Campaign).

## Governing Mathematical & Biomechanical Framework

1. **Transmural Pressure Gauge Invariance in Microfluidic Chest Cavities (`Nelson et al. 2017` *Development* 144:4328–4335, `doi:10.1242/dev.154823`)**:
   - Chloride-driven luminal fluid secretion against hydraulic laryngeal/tracheal resistance maintains a positive transmural pressure difference across the developing airway wall:
     $$\Delta P(P_{\text{lumen}}, P_{\text{pleura}}) = P_{\text{lumen}} - P_{\text{pleura}} \approx 100\text{–}300\text{ Pa}.$$
   - By independently clamping luminal pressure $P_{\text{lumen}}$ and pleural cavity pressure $P_{\text{pleura}}$ in a microfluidic chest organ culture device, Nelson et al. (2017) demonstrated exact common-mode gauge invariance:
     $$\Delta P(P_{\text{lumen}} + P_0, \; P_{\text{pleura}} + P_0) = \Delta P(P_{\text{lumen}}, P_{\text{pleura}}).$$

2. **Laplace–Young Circumferential Epithelial Wall Stress (`Nelson et al. 2017`; `Jaslove & Nelson 2018` *Development* 145:dev167395)**:
   - The distending transmural pressure $\Delta P$ generates circumferential (hoop) tensile stress across the distal epithelial bud wall of luminal radius $R$ and thickness $h$:
     $$\sigma_\theta(\Delta P, R, h) = \frac{\Delta P \cdot R}{h}.$$
   - Strictly increasing $\Delta P_2 > \Delta P_1$ at $R, h > 0$ monotonically elevates epithelial wall tension $\sigma_\theta(\Delta P_2) > \sigma_\theta(\Delta P_1)$, promoting Etv4/FGF10/ERK1/2 proliferation at distal tips.

3. **Mechanosensitive Tip Bifurcation Rate & Synchronized Airway Smooth Muscle (ASM) Peristalsis (`Kim et al. 2015` *Dev. Cell* 34:668–681; `Unbekandt et al. 2008` *Mech. Dev.* 125:314–324)**:
   - Above the critical distension threshold $\Delta P_c \approx 50\text{ Pa}$, synchronized ASM $\text{Ca}^{2+}$ peristaltic wave frequency $f_{\text{ASM}}(\Delta P)$ and tip bifurcation rate $\Gamma_{\text{branch}}(\Delta P)$ follow:
     $$f_{\text{ASM}}(\Delta P) = f_0 + f_{\max}\frac{\Delta P^2}{K_{\text{ASM}}^2 + \Delta P^2}, \qquad \Gamma_{\text{branch}}(\Delta P) = \Gamma_0 \, \alpha_{\text{ASM}} \frac{(\Delta P - \Delta P_c)_+^n}{K_P^n + (\Delta P - \Delta P_c)_+^n},$$
     where tracheal drainage ($\Delta P \le \Delta P_c$) or nifedipine L-type $\text{Ca}^{2+}$ channel blockade ($\alpha_{\text{ASM}} \to 0.05$) collapses terminal bud bifurcation.

4. **(Pass 3) Spatially Resolved CFTR Secretion, Poiseuille Luminal Pressure Network & 3D Pleural Cavity (`Olver & Strang 1974` *J. Physiol.* 241:327–357; `Harding, Bocking & Sigger 1986` *J. Appl. Physiol.* 60:160–165; `Metzger et al. 2008` *Nature* 453:745–750)**:
   - Chloride-driven secretion per unit epithelial area increases from trachea to distal tips, $j_{\text{Cl}}(g) = j_{\text{prox}} (j_{\text{dist}}/j_{\text{prox}})^{g/g_{\max}}$; segment flows $Q_i$ are subtree sums and luminal pressure accumulates along Hagen–Poiseuille resistances $R_i = 8\mu L_i/(\pi r_i^4)$:
     $$P_{\text{lumen}}(\text{node}) = P_{\text{amn}} + R_{\text{larynx}} Q_{\text{tot}} + \sum_{\text{path}} R_i Q_i \;\Rightarrow\; \Delta P(\text{tip}) \ge \Delta P(\text{trachea})$$
     (Lean 4 Law 4 `poiseuille_tree_pressure_accumulation`). Numerically the intra-tree drop is $<0.01\%$ of $\Delta P$: the larynx dominates, as in Harding et al. (1986).
   - The tree is grown in 3D with 90° rotating bifurcation planes inside an **analytic ellipsoidal pleural cavity** (a reduced stand-in for a stereolithographic chest mesh); tips reaching the pleura are arrested, so hyperplastic pressures (≥ 300 Pa) saturate against the chest wall.

5. **(Pass 3) Laryngeal Valve Opening/Closing Cycles (`Harding & Hooper 1996` *J. Appl. Physiol.* 81:209–224)**:
   - $C_{\text{lung}}\,dP/dt = Q_{\text{sec}} - (P - P_{\text{amn}})/R_{\text{larynx}}(t)$ with $R_{\text{larynx}}$ switching between the adducted-glottis value (closed plateau 250 Pa ≈ 1–2 mmHg) and $0.25\times$ that during fetal breathing-movement episodes (open ≈ 40 % of a 40 min period). The time-integrated $\Delta P$ — and so branching $N(T)=N_0\exp\int\Gamma_{\text{branch}}(\Delta P(t))dt$ — decreases strictly with the open duty cycle (Lean 4 Law 5 `laryngeal_valve_duty_cycle_monotonicity`): occlusion ($d=0$) → hyperplasia, open larynx ($d=1$) → hypoplasia.

## Vendored Files

- `pleural_cavity_valve_reference.json` — **(Pass 3)** ellipsoidal cavity geometry, CFTR secretion profile and Poiseuille network constants, laryngeal valve / FBM cycle parameters, each flagged `provenance: literature` or `provenance: model`. The four new DOIs (Olver & Strang 1974, Harding et al. 1986, Harding & Hooper 1996, Metzger et al. 2008) were recorded from memory and are flagged `doi_verified_offline: false`.
- `tracheal_viscous_fingering_reference.json` — SHA-256-locked reference parameters, 8-point $\Delta P \in [0, 400]\text{ Pa}$ microfluidic chest sweep, 4 common-mode gauge invariance $(P_{\text{lumen}}, P_{\text{pleura}})$ pairs, nifedipine and tracheal drainage controls, and empirical specimen morphometrics.
- `embryonic_lung_branching_specimen.jpg` — *Mus musculus* pseudoglandular (E11–E16.5) embryonic lung branching tree and H&E epithelial tubule cross-section (`File:Development of the respiratory system in mice..jpg`, Wikimedia Commons CC BY 3.0, `600x343` px, `sha256: 8086ebb8b44cbdff4002f62707afa7ec0732edf41f9c53bfef049bf09eef6bae`).
- `paper_citations.bib` — BibTeX citations with verified DOIs.
- `scout_manifest.json` — Step 1 (`01_scout`) literature and specimen manifest.
