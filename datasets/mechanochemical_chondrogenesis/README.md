# Mechanochemical Chondrogenesis & Tetrapod Limb Branching Dataset (`datasets/mechanochemical_chondrogenesis/`)

Reference parameters, formal mechanochemical equations, and salamander limb skeleton specimen micro-CT reconstructions for **Slot 013 (`slot-013-mechanochemical-chondrogenesis`)** of the Project Delphi & MorphoEval Campaign, reproducing **Oster, Murray & Harris (1983)** *"Mechanical aspects of mesenchymal morphogenesis"* (*J. Embryol. Exp. Morph.* 78:83–125, DOI: `10.1242/dev.78.1.83`) and **Oster, Murray & Maini (1985)** *"A model for chondrogenic condensations in the developing limb: the role of extracellular matrix and cell tractions"* (*J. Embryol. Exp. Morph.* 89:93–112, DOI: `10.1242/dev.89.1.93`).

## Governing Mechanochemical Equations

1. **Mesenchymal Cell Conservation with Long-Range Biharmonic Diffusion & ECM Passive Advection:**
   \[
   \frac{\partial n}{\partial t} = D_1 \nabla^2 n - D_2 \nabla^4 n - \nabla \cdot \left(n \frac{\partial \mathbf{u}}{\partial t}\right) + r n (N_0 - n)
   \]
2. **Viscoelastic Extracellular Matrix (ECM) Force Balance with Contact-Inhibited Cell Traction:**
   \[
   \nabla \cdot \left[ \mu_1 \frac{\partial \boldsymbol{\varepsilon}}{\partial t} + \mu_2 \left(\nabla \cdot \frac{\partial \mathbf{u}}{\partial t}\right)\mathbf{I} + E'\left(\boldsymbol{\varepsilon} + \nu'\theta \mathbf{I}\right) + \frac{\tau n^2}{1 + \lambda n^2}\mathbf{I} \right] = s \rho \mathbf{u}
   \]
3. **Critical Cell Traction Threshold \(\tau_c\) & Proximo-Distal Chondrogenic Branching (\(1 \to 2 \to 4/5\)):**
   When effective active cell traction \(\tau_{\text{eff}}\) exceeds the elastic-diffusive stabilization threshold \(\tau_c\), the homogeneous mesenchymal state \(n = N_0, \mathbf{u} = \mathbf{0}\) undergoes a spatial mechanochemical instability at characteristic wavenumber \(k_c > 0\), generating focal condensation (stylopod: humerus/femur, \(m=1\)), dichotomous Y-bifurcation (zeugopod: radius+ulna / tibia+fibula, \(m=2\)), and distal digital branching (autopod: \(m=4\) forelimb digits I–IV and \(m=5\) hindlimb digits I–V in urodele salamanders).

## Pass 3 Additions (audit `2026-10-19-mechanochemical-chondrogenesis-audit-v3`)

4. **Moving free-boundary AER outgrowth (`aer_free_boundary_outgrowth_pass3`):** the 2D coupled cell–ECM PDE is solved on a limb paddle whose distal boundary \(L(t) = L_0 + v_{\text{AER}} t\) advances; a progress zone of depth \(d_{pz}\) behind the front suppresses traction and maintains proliferation (Oster, Murray & Maini 1985 §4; ten Berge et al. 2008). Only the proximal humeral nucleus is seeded; the \(1 \to 2 \to 4/5\) hierarchy emerges. The **Saunders (1948)** AER-removal experiment (DOI `10.1002/jez.1401080304`) is reproduced by stopping the front at \(t_{\text{ablate}}\): expected segments formed \([1, 1, 2, 3]\) for removal at \(t = 0, 14, 30\) and the intact limb.
5. **Hybrid Bmp–Sox9–Wnt coupling (`hybrid_bsw_coupling_pass3`):** Raspopovic et al. (2014, DOI `10.1126/science.1252960`) linearised BSW kinetics (parameters vendored from `datasets/bmp_sox9_wnt_digits/`) are coupled to the mechanics through \(\tau(S) = \tau(1 + \alpha_S S)\) and \(\partial_t S \mathrel{+}= \kappa_S (n - N_0)\) (Hall & Miyake 2000). The table records the predicted synergy window \(\tau_c^{\text{hyb}} = 0.346 < 0.85 < \tau_c^{\text{mech}} = 1.440\) and the 2×2 factorial expectation \([0, 0, 0, 4]\).
6. **`script_default_parameters`:** the parameters actually used by the script (\(D_1 = 0.28, D_2 = 0.015, s = 0.85, r = 0.32\)) are now recorded next to the scout-time `canonical_parameters` (\(D_1 = 0.08, \dots\)), which the script never adopted; the discrepancy is disclosed in audit v3 and left for Pass 4 reconciliation. The analytic \(\tau_c = 1.2776\) and the solver drift-kernel \(\tau_c = 1.4397\) are both recorded.

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`cb9d73c33c4113e46b58a8e3992861e2d91d4965461911e4808e2802e1ede6b0`)
- `mechanochemical_chondrogenesis_reference.json` — Formal Oster–Murray–Maini (1983, 1985) parameters and salamander limb branching hierarchy
- `salamander_limb_cartilage_specimen.jpg` — Urodele salamander (*Oedipina ecuatoriana*, PeerJ 9934 Fig. 13, CC BY 4.0) forelimb (humerus \(\to\) ulna+radius \(\to\) 4 digits I–IV) and hindlimb (femur \(\to\) tibia+fibula \(\to\) 5 digits I–V) 3D skeletal reconstruction (`8030c8291c3a6d5b0efa746c63c1d5d9b143027645613dc1446e4431a6d60b94`)
- `paper_citations.bib` — BibTeX citations for Oster, Murray & Harris (1983), Oster, Murray & Maini (1985), Murray & Oster (1984), Shubin & Alberch (1986), and (Pass 3) Saunders (1948), Raspopovic et al. (2014), ten Berge et al. (2008), Hall & Miyake (2000)

The reference JSON is hashed at runtime (`verify_dataset_integrity`) and its digest recorded in the audit measurements, but it is not pinned (it is extended each pass); the specimen JPG and scout manifest are SHA-256-pinned in the script.
