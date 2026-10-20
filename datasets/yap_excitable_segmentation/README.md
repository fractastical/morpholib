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

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`852a6ed7852f291d9ade2ce79493d64bf6d5fffb4fa6a6f93d4132cc18dff7c1`)
- `yap_excitable_segmentation_reference.json` — Formal FHN parameterization and Hubaud et al. (2017) empirical mouse PSM perturbation regime benchmarks
- `mouse_psm_somites_specimen.jpg` — Dorsal scanning electron micrograph of vertebrate paraxial mesoderm and bilateral segmented somite rows (`8b80971796cb29e290c10886c0026092bc30f3f624d2a75fa72f9aaf05187d5f`, CC BY 4.0, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Hubaud et al. (2017), FitzHugh (1961), Lauschke et al. (2013), and Dupont et al. (2011)
