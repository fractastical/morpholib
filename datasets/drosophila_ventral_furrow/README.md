# Pulsed Apical Constriction & Viscous-Shell Biomechanics of *Drosophila* Ventral Furrow Invagination (`datasets/drosophila_ventral_furrow/`)

Reference parameters, formal 2D incompressible viscous-vertex cross-sectional equations, Conte et al. (2012) biomechanical synergy regimes, Martin et al. (2009) ratcheted actomyosin pulse parameters, and biological specimen photography for **Slot 046 (`slot-046-drosophila-ventral-furrow`, `he-drosophila-ventral-furrow`)** of the Infinite Morphospace & MorphoEval Campaign, reproducing:

- **Conte, V., Muñoz, J. J., Baum, B. & Miodownik, M. (2012).** *"Biomechanical analysis of ventral furrow formation in Drosophila melanogaster."* *PLoS ONE* 7(11):e49973. DOI: `10.1371/journal.pone.0049973`
- **Martin, A. C., Kaschube, M. & Wieschaus, E. F. (2009).** *"Pulsed contractions of an actin–myosin network drive apical constriction."* *Nature* 457(7228):495–499. DOI: `10.1038/nature07522`
- **He, B., Doubrovinski, K., Polyakov, O. & Wieschaus, E. (2014).** *"Apical constriction drives tissue-scale hydrodynamic flow to mediate cell elongation."* *Nature* 508(7496):392–396. DOI: `10.1038/nature13070`
- **Polyakov, O., He, B., Swan, M., Shaevitz, J. W., Kaschube, M. & Wieschaus, E. (2014).** *"Passive mechanical forces control cell-shape change during Drosophila ventral furrow formation."* *Biophysical Journal* 107(4):998–1010. DOI: `10.1016/j.bpj.2014.07.013`

## Governing Mathematical Laws

1. **Incompressible Trapezoidal Wedge Geometry (`incompressible_wedge_cell_lengthening_and_basal_expansion`):**
   \[
   A_i = \frac{1}{2}\left(w_{\text{ap},i} + w_{\text{ba},i}\right) h_i = A_0 = 262.4\,\mu\text{m}^2.
   \]
   Under apical constriction (\(w_{\text{ap},i} < w_0\)), cytoplasmic incompressibility forces apicobasal cell lengthening (\(h_i > h_0\) when basal width is held fixed by a stiff basal cortex) or basal expansion (\(w_{\text{ba},i} > w_0\) when basal relaxation and apicobasal shortening occur).
2. **Martin et al. (2009) Twist-Stabilized Ratcheted Myosin-II Pulse Dynamics (`twist_ratcheted_pulse_stepwise_constriction_vs_recoil`):**
   \[
   \frac{dL_{\text{ap}}}{dt} = -\alpha_{\text{pulse}} M(t) + \beta_{\text{relax}}\left(1 - r_{\text{twist}}\right)\left(L_0 - L_{\text{ap}}\right).
   \]
   With full Twist ratchet stabilization (\(r_{\text{twist}} = 1\)), each actomyosin pulse produces a strictly negative net step-wise apical width change (\(\Delta L_{\text{net}} < 0\)), whereas unratcheted recoil (\(r_{\text{twist}} = 0\)) relaxes back toward \(L_0\) between pulses.
3. **Conte et al. (2012) Synergistic Biomechanical Invagination Law (`conte_2012_synergistic_ventral_furrow_invagination`):**
   Apical constriction alone generates only a shallow groove (\(D_{\text{furrow}} = 16.5\,\mu\text{m}\)) with over-elongated mesoderm cells (\(h = 48.1\,\mu\text{m}\)); adding basal relaxation increases invagination to \(29.5\,\mu\text{m}\); full synergy of apical constriction + basal relaxation + lateral shortening + ectodermal pushing achieves complete ventral furrow invagination (\(D_{\text{furrow}} = 41.0\,\mu\text{m}\)).

4. **[Pass 3] Reduced 3D AP Shell Coupling (`ap_tension_and_curvature_stiffening_reduce_furrow_depth`):**
   \[
   \ell_B^4 D'''' - \ell_N^2 D'' + k_{\text{rel}}(z)\,D = D_{2D}(t)\,\chi_{\text{meso}}(z),\qquad
   k_{\text{rel}}(z) = \frac{\kappa_z^2 + \kappa_\theta^2}{(\kappa_z^2 + \kappa_\theta^2)_{\text{eq}}}.
   \]
   The 2D ring depth drives a Donnell/Winkler elastic line along the AP axis of the prolate ellipsoid (\(a = 250\,\mu\text{m}\), \(b = 83.2\,\mu\text{m}\)); meridional/hoop curvature stiffens the shell toward the poles (\(\sim 13\times\)) and germband-extension tension \(\ell_N\) flattens the profile. Both strictly reduce the modal depth \(D_m = q/(Bm^4 + Nm^2 + k)\). The profile is swept into a \(\theta \times z\) shell mesh (5265 nodes). This is a reduced model, not a 3D finite-element shell.
5. **[Pass 3] Intracellular Stokes Flow Around the Nucleus (`affine_lengthening_flow_divergence_free_and_basalward`):**
   \[
   4\,\partial_{xy}\!\left[\mu\,\psi_{xy}\right] + (\partial_{xx} - \partial_{yy})\!\left[\mu\,(\psi_{xx} - \psi_{yy})\right] = 0,\qquad u = \psi_y,\; v = -\psi_x .
   \]
   Variable-viscosity creeping flow inside the Phase-I lengthening mesoderm cell (He et al. 2014) with the nucleus as a \(50\times\)-viscosity inclusion; the uniform-viscosity affine solution \(\psi = \varepsilon x (h - y)\) is recovered to \(10^{-12}\) and the nucleus is a passive tracer (\(v_{\text{nuc}}/v_{\text{cyto}} = 0.98\)).

## Vendored Files & Cryptographic Digests

- `scout_manifest.json` — OpenAlex + Wikimedia Commons literature & specimen scout dossier (`0c6354ab1d934ba37fd34e35649c381535c9a595e26224e52959895c156e8c1f`)
- `drosophila_ventral_furrow_reference.json` — Formal viscous-vertex parameters, ratcheted pulse ODE, and 6-regime Conte/Martin/He dataset
- `drosophila_ventral_furrow_specimen.jpg` — Fluorescent confocal micrographs of *Drosophila melanogaster* ventral furrow invagination stained for DAPI (nuclei, blue), Neurotactin (membranes, green), and Myosin II heavy chain (apical cortex, red) across 4 transverse invagination stages (`f426366f7ede436da44782b28ebac9bd013cace1f8fe44b6ed382b77ad86c419`, CC BY 4.0, Wikimedia Commons)
- `ap_shell_stokes_reference.json` — **[Pass 3]** Egg geometry (\(a, b\), mesoderm AP extent), AP shell line-model parameters (\(\ell_B\), \(\ell_N\) ramp, sweeps), and Stokes nucleus parameters, with per-source `doi_verified_offline` flags (`8f196893e04e645768937dc73095a86af08014f8ec207b5ffc7d718b137fbf9d`; SHA-256-pinned in the script)
- `paper_citations.bib` — BibTeX citations for Conte et al. (2012), Martin et al. (2009), He et al. (2014), Polyakov et al. (2014), and the Pass 3 additions Sweeton et al. (1991), Leptin & Grunewald (1990), Lye et al. (2015)

## Pass 3 Caveats

- The AP extent of the mesoderm (0.15–0.80 EL), the reference furrow span (≈0.65 EL), and the germband-extension onset (≈9 min) are approximate literature values, not digitised measurements.
- \(\ell_B = 22\,\mu\text{m}\) and the WT \(\ell_N\) ramp to \(30\,\mu\text{m}\) are model assumptions; \(\ell_N\) is swept 0–150 µm but is not calibrated against laser-ablation recoil (open Pass 4 item).
- The Stokes domain is a fixed rectangle at the instant of maximal lengthening rate; it does not move with the trapezoid and carries no cortical actomyosin stress boundary condition (open Pass 4 item).
- The Sweeton (1991), Leptin & Grunewald (1990) and Lye (2015) DOIs could not be re-verified offline during Pass 3.
