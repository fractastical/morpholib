# YAP-Gated Excitable FitzHugh–Nagumo Segmentation Clock Dataset (`datasets/yap_excitable_segmentation/`)

Reference parameters, formal equations, and biological specimen photography for **Slot 027 (`slot-027-yap-excitable-segmentation`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing **Hubaud, Regev, Mahadevan & Pourquié (2017)** *"Excitable dynamics and Yap-dependent mechanical cues drive the segmentation clock"* (*Cell* 171(3):668–682, DOI: `10.1016/j.cell.2017.08.043`).

## Governing Equations

1. **FitzHugh–Nagumo Fast-Slow Excitable Segmentation Clock PDE:**
   \[
   \tau_u \frac{\partial u}{\partial t} = u - \frac{u^3}{3} - v + I_{\text{eff}}(Y_{\text{nuc}}, J_{\text{Dll1}}) + D_u \nabla^2 u, \qquad
   \tau_v \frac{\partial v}{\partial t} = u + a - b v
   \]
   where \(u\) represents the fast activator (Notch / LuVeLu *Lfng-Venus* / *Hes7* oscillatory activity), \(v\) is the slow negative-feedback recovery variable, and \(\epsilon = \tau_u / \tau_v \ll 1\).

2. **YAP Mechanotransduction Excitability Gating Law:**
   \[
   I_{\text{eff}} = I_{\text{basal}} + J_{\text{Dll1}} - \gamma_{\text{YAP}}\,Y_{\text{nuc}}(A_{\text{cell}}, E_{\text{sub}})
   \]
   where nuclear YAP levels \(Y_{\text{nuc}} \in [0, 1]\) increase with cell spreading area \(A_{\text{cell}}\) and cytoskeletal F-actin tension on stiff fibronectin substrates, shifting the fixed point \((u^*, v^*)\) across the supercritical Hopf bifurcation at \(|u_H| = \sqrt{1 - \epsilon b}\).

3. **Quiescent-to-Oscillatory Bifurcation & Latrunculin A / `Yap5SA` Perturbations:**
   - **Spread isolated PSM cells on fibronectin** (\(Y_{\text{nuc}} = 0.85\)): \(|u^*| > 1 > |u_H|\), \(\operatorname{Tr}(J) < 0\) — monostable excitable quiescent state (single suprathreshold pulse response without sustained oscillation).
   - **Latrunculin A / Y-27632 (Rockout) / `Yap1` KO / intact high-density mPSM** (\(Y_{\text{nuc}} \le 0.30 < Y_{\text{crit}}\)): \(|u^*| < |u_H| < 1\), \(\operatorname{Tr}(J) > 0\) — autonomous limit-cycle oscillations (\(T_{\text{osc}} \approx 140\text{ min}\)) and 2D traveling target waves.
   - **Constitutively active `Yap5SA` overexpression** (\(Y_{\text{nuc}} = 1.0\)): locks \(|u^*| > 1\) in the quiescent branch even under Latrunculin A or high cell density.

## Pass 3 Extensions (audit `2026-10-19-yap-excitable-segmentation-audit-v3`)

4. **FGF/Wnt gradient clock-and-wavefront arrest** (`clock_wavefront_fgf_gradient_reference` in the reference JSON; Cooke & Zeeman 1976, Dubrulle et al. 2001):
   \[
   F(x,t) = \min\!\big(1, e^{-(x_{tb}(t) - x)/\lambda_F}\big), \quad x_{tb}(t) = x_{tb0} + v_{wf} t, \quad g(F) = \frac{F^h}{F_{thr}^h + F^h},
   \]
   \[
   I_{\text{eff}}(x,t) = I_{\text{basal}} + J_{\text{Dll1}}(x,t) + \gamma_F\,(g(F) - 1) - \gamma_{\text{YAP}} Y_{\text{nuc}}, \qquad
   J_{\text{Dll1},i} = J_{\max}\,\overline{\sigma(u_j)}_{j \in N(i)} .
   \]
   A cell arrests when \(\gamma_F (1 - g) > I_{\text{post}} - I_H\) (Lean Theorem 5); the frozen phase \(u_{\text{arrest}}(x)\) thresholded into a Mesp2-like switch gives discrete stripes with spacing \(S = v_{wf} T\) (Lean Theorem 4). With \(\gamma_F = 1.6\) the anterior cells are non-excitable and the pulses truly arrest at the front; with \(\gamma_F = 0.8\) they leak anteriorly (control). Parameters: \(\lambda_F = 350\) µm, \(v_{wf} = 0.70\) µm/min, \(F_{thr} = 0.15\), \(h = 4\), \(J_{\max} = 0.15\), \(D_u = 24\) µm²/min. Empirical anchor: mouse somite AP length ≈ 100 µm (Gomez et al. 2008) with a ~142 min clock (Hubaud et al. 2017).

5. **Substrate stiffness → F-actin → LATS1/2 → nuclear YAP kinetics** (`yap_mechanotransduction_kinetics_reference`; Dupont et al. 2011, Wada et al. 2011):
   \[
   T(E, A) = \frac{E^{h_E}}{E_{50}^{h_E} + E^{h_E}}\,\frac{A}{A_{50} + A}\,\rho_{\text{actin}}\,c_{\text{ROCK}}, \qquad
   L(T) = \frac{1}{1 + (T/T_L)^{m}}, \qquad
   Y_n = \frac{r_n}{1 + r_n + r_p},\; r_n = r_0(1 + \alpha_T T),\; r_p = \kappa_{\text{LATS}} L(T).
   \]
   \((\alpha_T, \kappa_{\text{LATS}}, T_L)\) are fitted at runtime to the five wild-type regime \(Y_{\text{nuc}}\) values (RMSE ≈ 0.054); `Yap5SA` is predicted with \(\kappa_{\text{LATS}} = 0\) and a 2× overexpression factor. The predicted critical stiffness \(E_{crit} \approx 12.6\) kPa lies inside Dupont's 0.7 kPa (cytoplasmic) – 40 kPa (nuclear) bracket. *Honesty note:* Dupont et al. (2011) found the stiffness response largely LATS-independent; the LATS branch follows Wada et al. (2011). The table's non-monotonic Y-27632 (0.30) vs high-density mPSM (0.18) pair cannot be captured by a tension-only cascade (contact inhibition is not modelled).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`852a6ed7852f291d9ade2ce79493d64bf6d5fffb4fa6a6f93d4132cc18dff7c1`)
- `yap_excitable_segmentation_reference.json` — Formal FHN parameterization and Hubaud et al. (2017) empirical mouse PSM perturbation regime benchmarks
- `mouse_psm_somites_specimen.jpg` — Dorsal scanning electron micrograph of vertebrate paraxial mesoderm and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Hubaud et al. (2017), FitzHugh (1961), Lauschke et al. (2013), Dupont et al. (2011), and (Pass 3) Cooke & Zeeman (1976), Dubrulle et al. (2001), Wada et al. (2011), Gomez et al. (2008)
