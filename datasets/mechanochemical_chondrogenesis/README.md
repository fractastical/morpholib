# Mechanochemical Chondrogenesis & Tetrapod Limb Branching Dataset (`datasets/mechanochemical_chondrogenesis/`)

Reference parameters, formal mechanochemical equations, and salamander limb skeleton specimen micro-CT reconstructions for **Slot 013 (`slot-013-mechanochemical-chondrogenesis`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing **Oster, Murray & Harris (1983)** *"Mechanical aspects of mesenchymal morphogenesis"* (*J. Embryol. Exp. Morph.* 78:83–125, DOI: `10.1242/dev.78.1.83`) and **Oster, Murray & Maini (1985)** *"A model for chondrogenic condensations in the developing limb: the role of extracellular matrix and cell tractions"* (*J. Embryol. Exp. Morph.* 89:93–112, DOI: `10.1242/dev.89.1.93`).

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

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`cb9d73c33c4113e46b58a8e3992861e2d91d4965461911e4808e2802e1ede6b0`)
- `mechanochemical_chondrogenesis_reference.json` — Formal Oster–Murray–Maini (1983, 1985) parameters and salamander limb branching hierarchy
- `salamander_limb_cartilage_specimen.jpg` — Urodele salamander (*Oedipina ecuatoriana*, PeerJ 9934 Fig. 13, CC BY 4.0) forelimb (humerus \(\to\) ulna+radius \(\to\) 4 digits I–IV) and hindlimb (femur \(\to\) tibia+fibula \(\to\) 5 digits I–V) 3D skeletal reconstruction (`8030c8291c3a6d5b0efa746c63c1d5d9b143027645613dc1446e4431a6d60b94`)
- `paper_citations.bib` — BibTeX citations for Oster, Murray & Harris (1983), Oster, Murray & Maini (1985), Murray & Oster (1984), and Shubin & Alberch (1986)
