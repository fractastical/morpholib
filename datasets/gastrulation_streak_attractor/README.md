# Avian Gastrulation Tensile Ring, Polonaise Twin-Vortex Flow & Primitive Streak Attractor (`datasets/gastrulation_streak_attractor/`)

Reference parameters, formal 2D viscous epiblast Stokes equations, strain-rate-driven active nematic Myosin-II feedback laws, 5 experimental perturbation regimes, and whole-blastoderm *Gallus gallus* micrographs for **Slot 050 (`slot-050-gastrulation-streak-attractor`, `serra-gastrulation-streak-attractor`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Saadaoui, M., Rocancourt, D., Roussel, J., Corson, F. & Gros, J. (2020).** *"A tensile ring drives tissue flows to shape the gastrulating amniote embryo."* *Science* 367(6476):453–458. DOI: `10.1126/science.aaw1965`
- **Serra, M., Serrano Nájera, G., Chuai, M., Plum, A. M., Santhosh, S., Spurlin, J. W., Saadaoui, M., Gros, J., Weijer, C. J. & Mahadevan, L. (2023).** *"A mechanochemical model recapitulates distinct vertebrate gastrulation modes / Dynamic Morphogenetic Skeletons."* *Nature Methods / Science Advances* 20:1739–1748. DOI: `10.1038/s41592-023-02009-y`
- **Rozbicki, E., Chuai, M., Karjalainen, A. I., Song, F., Sang, H. M., Martin, R., Knölker, H.-J., MacDonald, M. P. & Weijer, C. J. (2015).** *"Myosin-II-mediated cell shape changes and cell intercalation contribute to primitive streak formation."* *Nature Cell Biology* 17(4):397–408. DOI: `10.1038/ncb3138`
- **Vasiev, B., Balter, A., Chaplain, M., Glazier, J. A. & Weijer, C. J. (2010).** *"Modeling gastrulation in the chick embryo: formation of the primitive streak."* *PLoS ONE* 5(5):e10571. DOI: `10.1371/journal.pone.0010571`

## Governing Mathematical Laws

1. **Polonaise Twin-Vortex Circulation Conservation & Midline Antisymmetry (`polonaise_twin_vortex_circulation_conservation`):**
   \[
   \eta \nabla^2 \mathbf{v} + \eta_b \nabla(\nabla \cdot \mathbf{v}) = -\nabla \cdot (m \mathbf{Q}), \qquad \psi(r, \theta) \propto \Delta m \, r \left(1 - \frac{r^2}{R^2}\right)^2 \sin\theta.
   \]
   Across the anterior–posterior (AP) midline \(y = 0\), the vorticity field satisfies \(\omega(x, -y) = -\omega(x, y)\), so the total circulation of the left and right counter-rotating Polonaise vortices sums identically to zero (\(\Gamma_L + \Gamma_R = 0\)) while driving positive midline anterior velocity (\(v_x(x, 0) > 0\)).
2. **Strain-Rate Alignment & Midline Primitive Streak Extension (`strain_rate_myosin_feedback_streak_extension`):**
   \[
   \frac{Dm}{Dt} = \alpha_m m \left(1 - \frac{m}{m_{\text{sat}}}\right) - \beta_m (\nabla \cdot \mathbf{v}) m, \qquad \frac{D\mathbf{Q}}{Dt} = \lambda_E m \mathbf{E}^{\text{dev}} + [\boldsymbol{\omega}, \mathbf{Q}] - \gamma_Q \mathbf{Q} + D_Q \nabla^2 \mathbf{Q}.
   \]
   Along the posterior-to-central midline where mediolateral convergence (\(\partial_y v_y < 0\)), AP extension (\(\partial_x v_x > 0\)), and apical cell ingression (\(\nabla \cdot \mathbf{v} < 0\)) occur simultaneously, the deviatoric strain rate \(E^{\text{dev}}_{xx} = \frac{1}{2}(\partial_x v_x - \partial_y v_y) > 0\) rotates and amplifies tangential posterior crescent Myosin-II fibers into an elongated anteroposterior primitive streak attractor ridge.
3. **FGF-Inhibited Symmetric Ring Collapse vs. Wild-Type Dipolar Streak Hierarchy (`saadaoui_serra_five_regime_streak_hierarchy`):**
   When the posterior-to-anterior ring asymmetry vanishes (\(\Delta m = 0\), as in SU5402 FGF receptor inhibition), the dipolar Polonaise vortex amplitude collapses to zero and isotropic marginal ring contraction forms a circular central ring pit (\(\mathrm{AP/ML} = 1.00\)) rather than an elongated linear primitive streak (\(\mathrm{AP/ML} = 4.25\) in wild-type HH3).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`1860c7ca251938c95cd864c1fe68b8f85f0a11818c5a220cb547e4461f25ae0d`)
- `gastrulation_streak_attractor_reference.json` — Formal viscous epiblast Stokes parameters, active nematic Myosin-II feedback laws, and 5-regime Saadaoui et al. (2020) / Serra et al. (2023) dataset (`7f91351689d35382d2a2cd2f475256140992c4f0a8194a8a4bc878d1317a736e`)
- `chick_primitive_streak_specimen.jpg` — Whole-blastoderm micrographs of *Gallus gallus* primitive streak formation across stages HH1, HH2, and HH3+ with *Wnt8c*/*Chordin* expression and Polonaise twin-vortex PIV velocity fields (`2e25656cfaa35f3d6b30f491170384d41f7d35cc651d88a743baadc9dd816e08`, CC BY 4.0, Vasiev et al. 2010 *PLoS ONE* 5(5):e10571, Wikimedia Commons)
- `paper_citations.bib` — BibTeX citations for Saadaoui et al. (2020), Serra et al. (2023), Rozbicki et al. (2015), and Vasiev et al. (2010) (`6cc141a824fe227d944ecaf3b84ce21ce2cf5c22abd920e8c3346b09bd806503`)

## Pass 3 additions (2026-10-19-gastrulation-streak-attractor-audit-v3)

- `ligand_emt_reference.json` — Pass 3 reference table (SHA-256 pinned in the script's `EXPECTED_SHA256`): FGF8 / Nodal / Wnt8c reaction-diffusion kinetics (D_FGF8 = 53 µm²/s, Yu et al. 2009; D_Nodal = 3.2 µm²/s, t½ = 95 min, Müller et al. 2012; Wnt8c values **assumed**), Hill constants and cascade gains for the pMLC activation `P = ρ_ROCK·H(a_ALK N)·(0.35 + 0.65 H(a_FGFR F))`, the per-regime receptor activities (SU5402 → a_FGFR = 0, SB431542 → a_ALK = 0, ectopic Vg1/Wnt → anterior Wnt source), and the reduced apical–basal wedging / basement-membrane column model (a, b, h = 2h₀/(a+b), Wdg, B, I = k_ing·Wdg·(1−B)) with its reference values (Rozbicki et al. 2015 ~2× apical constriction; Voiculescu et al. 2014 SB431542 ingression block; Nakaya et al. 2008 streak-restricted BM breakdown). Parameters marked `ASSUMED` / `DERIVED` in the file are not independently sourced and are carried as Pass 4 items.
- `paper_citations.bib` now also lists Yu et al. (2009), Müller et al. (2012), Nakaya et al. (2008) and Voiculescu et al. (2014); their DOIs were recalled offline and are **unverified**.

### Governing laws added in Pass 3 (Lean: `ligand_cascade_posterior_myosin_asymmetry`, `apical_basal_wedging_ingression_sink`)

4. **Wnt8c → Nodal → FGF8 ligand cascade upstream of ring myosin phosphorylation:** \(\partial_t W = D_W\nabla^2 W + s_W S_{\text{post}} - k_W W\), \(\partial_t N = D_N\nabla^2 N + s_N c_{\text{ind}}(1 + f_N H(N)) - k_N N\), \(\partial_t F = D_F\nabla^2 F + s_F H(N) - k_F F\), \(Dm/Dt \mathrel{+}= \alpha_P P (1 - m/m_{\text{sat}})\). The posterior-restricted marginal Wnt source *produces* the ring asymmetry Δm; decay lengths \(\lambda = \sqrt{D/k}\) (FGF8 371 µm, Nodal 162 µm) are verified against the spectral solver.
5. **Reduced trans-epithelial wedging / BM remodeling / ingression sink:** one columnar cell per planar grid point; Myosin-driven apical constriction + basal expansion wedge the cell and thicken it at constant volume; Nodal/FGF-licensed basement-membrane breakdown permits ingression \(I\), which becomes the planar Stokes sink \(\nabla\cdot\mathbf v = -I\) (replacing the Pass 2 prescribed Gaussian midline sink). This is a 2.5D column reduction, not a 3D FEM.

**Known residual (honest):** the coupled 2D model forms a thickened, ingressing posterior crescent (field-measured EMT-zone AP/ML ≈ 0.5–0.6) but does not elongate it into the 4.25:1 midline streak within 8 h; mediolateral cell intercalation (Rozbicki et al. 2015) is not modelled. The script's `streak_ap_ml_aspect_ratio` / `peak_polonaise_speed_um_min` readouts remain *calibrated lookups* and are flagged as circular in the code.
