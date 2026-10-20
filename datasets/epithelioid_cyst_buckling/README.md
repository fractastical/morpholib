# Dataset: Buckling of an Epithelium Growing Under Spherical Confinement (`epithelioid_cyst_buckling`)

## Primary Literature & DOIs

1. **Trushko A, Di Meglio I, Merzouki A, Blanch-Mercader C, Abuhattum S, Guck J, Alessandri K, Chopard B, Kruse K, Roux A (2020)**.
   *Buckling of an epithelium growing under spherical confinement.* **Developmental Cell** 54(5):655–668.e6.
   DOI: [`10.1016/j.devcel.2020.07.019`](https://doi.org/10.1016/j.devcel.2020.07.019)
2. **Hannezo E, Prost J, Joanny J-F (2014)**.
   *Theory of epithelial sheet morphology in three dimensions.* **Proceedings of the National Academy of Sciences** 111(1):27–32.
   DOI: [`10.1073/pnas.1312076111`](https://doi.org/10.1073/pnas.1312076111)
3. **Alessandri K, Sarangi BR, Gurchenkov VV, Sinha B, Kießling TR, Fetler L, Rico F, Scheuring S, Lamaze C, Simon A, Geraldo S, Vignjevic D, Doméjean H, Rolland L, Funfak A, Bibette J, Bremond N, Nassoy P (2013)**.
   *Cellular capsules as a tool for multicellular spheroid production and for investigating the mechanics of tumor progression in vitro.* **Proceedings of the National Academy of Sciences** 110(37):14843–14848.
   DOI: [`10.1073/pnas.1309482110`](https://doi.org/10.1073/pnas.1309482110)
4. **Zoelly R (1915)**.
   *Über ein Knickungsproblem an der Kugelschale.* **ETH Zürich Doctoral Thesis**.
   DOI: [`10.3929/ethz-a-000096658`](https://doi.org/10.3929/ethz-a-000096658)

## Governing Mathematical & Biomechanical Framework

When Madin–Darby Canine Kidney (*Canis lupus familiaris*, MDCK) or human colorectal epithelioid cells (*Homo sapiens*, Caco-2) form a polarized confluent monolayer of thickness $h$ lining the inner wall of a hollow spherical alginate hydrogel microcapsule of inner radius $R_0 \in [50, 150]\,\mu\text{m}$ and Young's modulus $E_{\text{alg}} \in [1, 10]\,\text{kPa}$ (Trushko et al. 2020), continued cell proliferation beyond confluence ($t > t_c$, doubling time $\tau_d \approx 16\text{–}20\,\text{h}$) generates tangential compressive stress and radial confinement pressure $P(t)$ against the elastic capsule wall:

1. **Zoelly / Föppl–von Kármán Spherical Shell Critical Buckling Pressure Law:**
   Treating the confluent epithelial cyst as a thin spherical elastic shell of Young's modulus $E_{\text{epi}}$, Poisson ratio $\nu \approx 0.45$, thickness $h$, and radius $R_0$, the critical radial confinement pressure required to trigger inward invagination buckling scales quadratically with the aspect ratio $h/R_0$:
   $$P_c = \frac{2 E_{\text{epi}}}{\sqrt{3(1 - \nu^2)}} \left(\frac{h}{R_0}\right)^2, \qquad \frac{\kappa_{\text{bend}}}{Y_{2\text{D}} R_0^2} = \frac{1}{12(1 - \nu^2)}\left(\frac{h}{R_0}\right)^2.$$
   Doubling monolayer thickness ($h \to 2h$) increases $P_c$ fourfold, whereas doubling capsule radius ($R_0 \to 2R_0$) reduces $P_c$ fourfold.

2. **Confluent Proliferative Stress Accumulation & Alginate Confinement Threshold:**
   During post-confluence proliferative growth with stretch ratio $g(t) = \exp((t - t_c)\ln 2 / \tau_d) > 1$, the elastic alginate capsule of wall thickness $H_{\text{alg}}$ and stiffness $E_{\text{alg}}$ exerts radial confining pressure:
   $$P_{\text{alg}}(g) = K_{\text{alg}}\,(g - 1), \qquad K_{\text{alg}} = \frac{2 E_{\text{alg}} H_{\text{alg}}}{(1 - \nu_{\text{alg}}) R_0}.$$
   Buckling occurs at time $t_b = t_c + \tau_d \log_2(g_c)$ when $P_{\text{alg}}(g_c) = P_c$, provided $K_{\text{alg}}$ is sufficiently stiff ($E_{\text{alg}} \ge 1.5\text{–}2.5\,\text{kPa}$). In unconfined cysts or ultra-soft capsules ($P_{\text{alg}} < P_c$), the cyst expands radially without buckling ($\delta / R_0 = 0$).

3. **Post-Buckling Inward Invagination Fold Depth, Mode Selection ($n=1$ vs $n=2$), and Actomyosin Perturbation (Blebbistatin):**
   Above the buckling threshold ($t > t_b$, $g > g_c$), excess epithelial arclength $L(t) = 2\pi R_0 g(t)$ is accommodated by an inward-protruding invagination fold of normalized depth:
   $$\frac{\delta(t)}{R_0} = c_\delta \sqrt{g(t) - g_c} \left(1 + \beta_{\text{apical}} \Gamma_{\text{myo}}\right),$$
   selecting a single deep inward fold ($n = 1$, Pogorelov mirror-buckled cap) in ~75% of standard capsules and two folds ($n = 2$) in larger/thinner capsules ($h/R_0 < 0.085$). Pharmacological inhibition of myosin-II contractility with **Blebbistatin** ($10\text{–}50\,\mu\text{M}$) or ROCK inhibitor **Y-27632** lowers effective cortical modulus $E_{\text{epi}}$ and apical-basal differential tension $\Gamma_{\text{myo}}$, shifting the critical buckling pressure and suppressing spontaneous fold sharpening.

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 069.
- `paper_citations.bib` — BibTeX citations with DOIs for Trushko et al. (2020), Hannezo et al. (2014), Alessandri et al. (2013), and Zoelly (1915).
- `epithelioid_cyst_buckling_reference.json` — Reference parameters, 4 experimental capsule regimes (Unconfined / Soft Control, Standard 1-Fold Buckled MDCK Cyst, Large Thin 2-Fold Buckled Cyst, Blebbistatin Myosin-II Inhibited Cyst), 12-point $(h/R_0)$ critical pressure scaling dataset, post-confluence time-series trajectories, and SHA-256 digests.
- `epithelioid_cyst_specimen.jpg` — Wikimedia Commons CC BY-SA 4.0 H&E histology micrograph of confluent follicular/cystic epithelial monolayers (`File:Thyroid follicle.jpg`, 1145×1419 px) used for lumen contour circularity, epithelial thickness ratio $h/R_0$, and boundary invagination morphometrics.

## Pass 3 additions — 3D triangulated vertex shell with apical-basal differential tension

- `shell3d_apicobasal_reference.json` (SHA-256 pinned in the script) — parameters and literature anchors of the Pass 3 3D shell: icosphere resolution (642 vertices / 1280 faces / 1920 edges), spring constant $k_s = (\sqrt{3}/2)\,Y_{2D}$, self-contact cutoff $= h$, growth step, the apical-basal tension sweep $\Delta\Gamma \in \{-4.26, 0, 2.13, 4.26, 8.52\}$ mN/m and the WT value $\Delta\Gamma_{WT} = 4.26$ mN/m. **$\Delta\Gamma_{WT}$ is calibrated, not measured**: it is the value for which the Hannezo–Prost–Joanny spontaneous curvature equals the confluent-sphere curvature, $H_0 R_0 = 6(1-\nu^2)\,\Delta\Gamma R_0/(E_{epi} h^2) = 1$.
- `pass3_citations.bib` — Helfrich (1973) spontaneous-curvature bending energy, Seung & Nelson (1988) spring-network / discrete bending mapping, Hannezo et al. (2014).

4. **Apical-basal spontaneous curvature (Lean Law 4, `apicobasal_spontaneous_curvature_minimizer`):**
   $$f(c) = \tfrac{\kappa}{2} c^2 - \Delta\Gamma\, h\, c,\quad c = 2H \;\Rightarrow\; c_0 = 2H_0 = \frac{\Delta\Gamma\, h}{\kappa},\qquad H_0 R_0 = \frac{6(1-\nu^2)\,\Delta\Gamma\, R_0}{E_{epi} h^2}.$$
5. **Inverted-cap penalty and unilateral capsule wall (Lean Law 5, `inverted_cap_penalty_and_capsule_wall_contact`):** for $h_m, c_0 \ge 0$, $(h_m + c_0)^2 = (h_m - c_0)^2 + 4 h_m c_0 \ge (h_m - c_0)^2$ — an inward (Pogorelov-inverted) cap is never cheaper than the lumen-concave cap when apical tension dominates; the Winkler wall penalty $\max(0, r - R_0)^2$ vanishes inside the capsule and is positive once the wall is dilated, so the confinement pressure is read as $P_{3D} = K_{alg}(r_{wall}/R_0 - 1)$, exactly as Trushko et al. read $P(t)$ from capsule dilation (Fig. 2E).

Known limitations of the 3D shell (carried to Pass 4): with the dataset's $E_{alg} = 5$ kPa the capsule dilates by ~21 % at $g = 1.527$ and the shell wrinkles into 12 shallow folds that coincide with the 12 five-valent icosphere vertices (mesh-templated), $\delta/R_0 = 0.30$ vs the observed single fold with $\delta/R_0 = 0.64$; reading $g$ as a linear stretch (the 2D ring's convention) over-supplies area and the wrinkles coarsen to 1–3 deep folds ($\delta/R_0 \approx 1.2$). The pressure trajectory read from wall dilation matches Fig. 2E with $r = 0.94$ (RMSE 0.09 kPa).
