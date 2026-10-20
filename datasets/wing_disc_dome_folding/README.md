# Dataset: Differential Planar Growth, Basal ECM Constraint & Precise Hinge Folding in *Drosophila* Wing Imaginal Disc (`wing_disc_dome_folding`)

## Primary Literature & DOIs

1. **Tozluoglu M, Duda M, Kirkland NJ, Barrientos R, Burden JJ, Muñoz JJ, Mao Y (2019)**.
   *Planar Differential Growth Rates Initiate Precise Fold Positions in Complex Epithelia.* **Developmental Cell** 51(3):299–312.e4.
   DOI: [`10.1016/j.devcel.2019.09.009`](https://doi.org/10.1016/j.devcel.2019.09.009)
2. **Sui L, Alt S, Weigert M, Dye N, Eaton S, Jülicher F, Dahmann C (2018)**.
   *Differential lateral and basal tension drive folding of Drosophila wing discs through two distinct mechanisms.* **Nature Communications** 9:4620.
   DOI: [`10.1038/s41467-018-06497-3`](https://doi.org/10.1038/s41467-018-06497-3)
3. **Villa-Fombuena G, Martín-Bermudo MD et al. (2021)**.
   *Live imaging of the Drosophila wing disc reveals the dynamics of fold formation and the role of the extracellular matrix.* **Development** 148(19):dev199728.
   DOI: [`10.1242/dev.199728`](https://doi.org/10.1242/dev.199728)
4. **Nelson CM (2016)**.
   *On buckling morphogenesis.* **Journal of Biomechanical Engineering** 138(2):021005.
   DOI: [`10.1115/1.4032128`](https://doi.org/10.1115/1.4032128)

## Governing Mathematical & Biomechanical Framework

During third-instar larval development ($64\text{ h} \to 96\text{ h}$ AEL), the *Drosophila melanogaster* wing imaginal disc transitions from a nearly flat columnar monolayer into a convex **wing pouch dome** flanked by three invariant stereotypic apical folds at the **Hinge–Notum (H/N)** ($s_{\text{HN}}^*/L = 0.26$), **Hinge–Hinge (H/H)** ($s_{\text{HH}}^*/L = 0.39$), and **Hinge–Pouch (H/P)** ($s_{\text{HP}}^*/L = 0.52$) boundaries.

In Tozluoglu et al. (2019) *Dev. Cell* 51:299–312, the total deformation gradient is decomposed multiplicatively into elastic and growth parts:
$$\mathbf{F} = \mathbf{F}_e \mathbf{F}_g, \qquad \mathbf{F}_g(s, t) = \operatorname{diag}\!\left(\gamma_s(s, t),\, 1,\, \gamma_z(s, t)\right), \qquad \gamma_s(s, t) = \exp\!\left(\int_{64\text{ h}}^{t} g(s, \tau)\,d\tau\right),$$
while the underlying **Collagen IV / Viking basal extracellular matrix (basement membrane, BM)** remains non-growing ($g_{\text{BM}} = 0$) with high elastic shear modulus $\mu_{\text{BM}} \gg \mu_{\text{cell}}$ ($\mu_{\text{BM}} / \mu_{\text{cell}} = 12.5$).

1. **Law 1 (Apical-vs-Basal Differential Growth & Basal ECM Stiffness Driving Positive Pouch Dome Curvature):**
   Differential tangential expansion between the proliferating apical/cellular epithelium ($g_{\text{apical}} > g_{\text{BM}}$) and the stiff elastic basal basement membrane ($\mu_{\text{BM}} > 0$) generates upward convex dome curvature:
   $$\kappa_{\text{dome}} \propto \mu_{\text{BM}} (g_{\text{apical}} - g_{\text{BM}}) > 0.$$
2. **Law 2 (Stereotyped H/N, H/H, H/P Fold Pinning at Local Growth Troughs $g''(s^*) > 0$ vs. Unpinned Uniform Buckling):**
   Early spatially heterogeneous proliferation $g(s)$ exhibits local minima (troughs, $g'(s^*) = 0, g''(s^*) > 0$) at $s_{\text{HN}}^*/L = 0.26$, $s_{\text{HH}}^*/L = 0.39$, and $s_{\text{HP}}^*/L = 0.52$ flanked by faster-growing hinge ridges and pouch/notum domains. Compressive bending moments focus precisely at the three growth troughs ($\sigma_s/L < 0.01$), whereas spatially uniform growth ($g'(s) = 0$) produces phase-random unpositioned buckling ($\sigma_s/L > 0.10$).
3. **Law 3 (Collagenase / *vkg*-RNAi ECM Ablation Collapse & Local Basal/Lateral Remodeling Synergy):**
   Enzymatic removal of the basal basement membrane ($\mu_{\text{BM}} \to 0$ via Collagenase or *viking* RNAi) abolishes both pouch dome curvature ($\kappa_{\text{dome}} = 0$) and apical fold initiation ($d_{\text{fold}} = 0$). In wild type, local basal ECM softening (MMP2) and lateral/apico-basal tension redistribution (*dysfusion* / *omb*, Sui et al. 2018) synergistically deepen the pre-positioned H/N, H/H, and H/P folds.

4. **Law 4 (Pass 3 — Reduced Disc-Proper + Peripodial Epithelium + Lumen Sac):**
   The Pass 2 audit asked for a full 3D finite-element mesh coupling the columnar disc proper (DP) to the squamous peripodial epithelium (PE) and lumenal hydrostatic pressure. Pass 3 implements the *closest honest reduced version*, a quasi-1D plane-strain sagittal sac (`solve_coupled_disc_peripodial_lumen`): the PE is a tensioned membrane $T_{\text{PE}} z_{\text{PE}}'' = -p$ clamped at the disc margins and in unilateral contact with the DP; the lumen is an incompressible fluid whose pressure $p$ is closed by a prescribed volume $V = h_{\text{lumen}} L(t)$ (or imposed directly); and $p$ lowers the growth-driven pouch dome through the shallow-cap Laplace response
   $$\Delta z_{\text{dome}} = \frac{p\,L_{\text{pouch}}^2}{16\,T_{\text{DP}}}, \qquad T_{\text{DP}} = \mu_{\text{BM}} h_{\text{BM}} (\gamma_{\text{pouch}}^2 - \gamma_{\text{pouch}}^{-4}) + T_{\text{cell}}.$$
   Because $p$ is a uniform transverse load, the growth-trough second difference of Law 2 is unchanged and the H/N, H/H, H/P loci are invariant to $p$ (machine-checked in `MathOfBiology.Loop_049.lumen_pressure_laplace_dome_deflection` and `uniform_lumen_load_preserves_growth_trough_second_diff`). The model predicts that a 1 mN/m PE cannot sustain more than a few Pa over the 340 µm disc, so the self-consistent lumen pressure is $\lesssim 3$ Pa and the DP dome/folds are growth-dominated, consistent with Tozluoglu et al.'s DP-only mechanism. **Residual:** no dorsoventral curvature or 3D FEM; $T_{\text{PE}}$, $h_{\text{lumen}}$, $h_{\text{margin}}$, $T_{\text{cell}}$ are order-of-magnitude assumptions (tagged in `peripodial_lumen_reference.json`), and no wing-disc lumen pressure measurement exists to validate $p$.

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 049.
- `paper_citations.bib` — BibTeX citations with DOIs for Tozluoglu et al. (2019), Sui et al. (2018), Villa-Fombuena et al. (2021), and Nelson (2016).
- `wing_disc_dome_folding_reference.json` — Quantitative reference dataset ($64\text{--}96\text{ h}$ AEL developmental timecourse, 7-zone spatial growth & curvature profile, 5 perturbation controls, and specimen reference metrics).
- `peripodial_lumen_reference.json` — **Pass 3** provenance-tagged parameter table for the reduced DP + PE + lumen sac (tags: `literature` / `derived` / `assumption` / `prediction`), including the imposed-pressure sweep grid and the stated residual. SHA-256 pinned in the script (`972d43d7…183a`).
- `drosophila_wing_disc_specimen.jpg` — Wikimedia Commons CC BY-SA 2.5 micrograph of three third-instar larval *Drosophila melanogaster* wing imaginal discs stained for pouch/hinge/notum reporter domains (`File:Drosophila imaginal discs.jpg`, 635×535 px, SHA-256 `b523296d0ee292e169a5578eaf278b3f48546d01566ed8ac71c34f928fa0a960`).
