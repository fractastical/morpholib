# Miura et al. (2009) & Jaslow (1990) Cranial Suture Fractal Interdigitation Reference Dataset

Primary reference dataset and biological specimen metadata for `2020-bailleul-allometric-cranial-sutures.py` (Slot 020 of the Project Delphi & MorphoEval Campaign).

## Governing Equations — Miura et al. (2009) *PNAS* 106:11218-11223 & Jaslow (1990) *J. Biomech.* 23:313-321

1. **Damped Kuramoto–Sivashinsky / KPZ Osteogenic Front Instability PDE:**

$$\frac{\partial h(x, t)}{\partial t} = -\alpha \frac{\partial^2 h}{\partial x^2} - \beta \frac{\partial^4 h}{\partial x^4} + \frac{v_0}{2}\left(\frac{\partial h}{\partial x}\right)^2 - \gamma h + \eta(x, t), \qquad \alpha > 0,\ \beta > 0$$

where $h(x, t)$ is the transverse osteogenic front displacement along the calvarial suture midline $x \in [0, L]$, $\alpha > 0$ is the Mullins–Sekerka-like short-wavelength osteogenic front destabilization rate driven by coupled BMP4/FGF2 osteoblast differentiation and tensile mechanical strain, $\beta > 0$ is fourth-order curvature/surface-tension smoothing across the mesenchymal suture width, $\frac{v_0}{2}(\partial_x h)^2$ is KPZ local normal-growth nonlinearity, $\gamma \ge 0$ is mesenchymal boundary confinement, and $\eta(x, t)$ is spatial osteogenic stochasticity.

2. **Linear Dispersion Relation & Primary Interdigitation Wavelength Selection:**

$$\omega(k) = \alpha k^2 - \beta k^4 - \gamma, \qquad k_{\max} = \sqrt{\frac{\alpha}{2\beta}}, \qquad \lambda_{\max} = 2\pi\sqrt{\frac{2\beta}{\alpha}}, \qquad \omega(k_{\max}) = \frac{\alpha^2}{4\beta} - \gamma$$

When intracranial/masticatory tensile strain is unloaded or *FGFR2* gain-of-function craniosynostosis (Apert/Crouzon syndrome) obliterates the mesenchymal inhibition zone ($\alpha \le 0$), $\omega(k) \le 0$ for all wavenumbers $k$, collapsing the suture to a flat or fused seam ($D_f \to 1.0$, $\text{SII} \to 1.0$).

3. **2D Bidirectional Overhanging Suture Front & Jaslow (1990) Bending Energy Absorption:**
   - Coupled 2D reaction-diffusion phase-field / bidirectional normal-front propagation produces multi-valued overhanging bone fingers with Suture Interdigitation Index $\text{SII} = L_{\text{arc}} / L_{\text{chord}} > 2.5$ and box-counting fractal dimension $D_f \in [1.30, 1.45]$.
   - By Jaslow (1990) *J. Biomech.* 23:313-321, dynamic impact and bending energy absorption per unit chord length increases monotonically with interdigitation complexity: $E_{\text{bend}}(\text{SII}) = E_0 \big(1 + \kappa_J (\text{SII} - 1)\big)$.

## Pass 3 Extensions — Quasi-3D Plate Layering & Reduced-Order Masticatory Strain Coupling

4. **Strain Mechanotransduction Law** (Lean 4 `strain_mechanotransduction_alpha_monotone`):

$$\alpha(\hat\epsilon) = \alpha_{\text{unl}} + (\alpha_{\text{sag}} - \alpha_{\text{unl}})\,\hat\epsilon, \qquad \hat\epsilon = \frac{\epsilon_{nn}}{\langle\epsilon_{nn}\rangle_{\text{sagittal, ecto}}}, \qquad \hat\epsilon_c = \frac{\alpha_{\text{unl}}}{\alpha_{\text{unl}} - \alpha_{\text{sag}}} = 0.2258$$

so $\hat\epsilon = 0$ reproduces the Miura et al. (2009) unloading regime ($\alpha = -0.35$) and $\hat\epsilon = 1$ the adult sagittal regime ($\alpha = 1.20$); direction after Byron et al. (2004).

5. **Kirchhoff–Love Through-Thickness Law** (Lean 4 `kirchhoff_love_ectocranial_endocranial_ordering`):

$$\epsilon_{nn}(s, \zeta) = \epsilon_{\text{mem}} + \zeta\,\kappa_{nn}(s), \qquad \zeta \in [-T/2, +T/2], \qquad \kappa_{nn} = -\mathbf{n}^{\top}\,\mathrm{Hess}(w)\,\mathbf{n}$$

with plate thickness $T = 6$ mm, five layers (ectocranial table → endocranial table) and vertical coupling $\kappa_z\,\partial_z^2 h$ in the KS/KPZ solver. The ectocranial table is the most tensile/unstable layer and the endocranial table compressive ($\alpha \approx -0.50$, flat), so endocranial sutures remain simpler than ectocranial traces.

6. **Reduced-order vault strain field (NOT finite elements).** The calvarial vault is an analytic ellipsoid (semi-axes 90 × 70 × 65 mm); bilateral temporalis and nuchal muscle groups act as Gaussian normal-deflection patches whose Hessian gives the curvature tensor; intracranial pressure is an isotropic membrane tension. The three suture paths (coronal, sagittal, lambdoid) are parametrised by bregma/lambda/pterion/asterion landmarks and the heterogeneous $\alpha(s,\zeta)$ is fed to the pseudo-spectral solver ($\alpha_{\text{mean}}$ in the exact integrating factor, $-\alpha_{\text{dev}}(x)\,h_{xx}$ dealiased in real space). All traction magnitudes are stated model assumptions in `allometric_cranial_sutures_reference.json` (`calvarial_shell_strain_model`); the nuchal : temporalis ratio is swept. The vendored specimen only shows the sagittal suture fully in frame, so the coronal/lambdoid outputs are compared against literature/vendored expectations, not against this photograph.

Secondary references added in Pass 3 (`paper_citations.bib`): Herring & Teng (2000) `10.1002/1096-8644(200008)112:4<575::AID-AJPA10>3.0.CO;2-0`, Byron et al. (2004) `10.1002/ar.a.20055`, Nicolay & Vaders (2006) `10.1002/jmor.10445`, Curtis et al. (2013) `10.1098/rsif.2013.0442`.

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `human_sagittal_suture_specimen.jpg` | Superior view photograph of an adult *Homo sapiens* calvaria showing the fractal interdigitated sagittal suture along the midline, coronal suture, and lambdoid suture (Wikimedia Commons `File:Sagittal suture 2.jpg`, CC BY-SA 3.0, 960×720 px) | `b3c411a6a47137708e4922c1b0f0381b67a3c1e885dc4d016fb0f7bb2f4eaf13` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 020 | `226c072307597439c420ed63e072337b85a3f7589d616e9fdcb015f37426d5d6` |
| `paper_citations.bib` | BibTeX entries for Miura et al. (2009), Jaslow (1990), Khonsari et al. (2013), Zollikofer & Weissmann (2011), and the Pass 3 secondary references Herring & Teng (2000), Byron et al. (2004), Nicolay & Vaders (2006), Curtis et al. (2013) | `81d1b1b367ddb12cb555d438d56a93df4c44855786a549ba55f399234e2a3342` |
| `allometric_cranial_sutures_reference.json` | Canonical Kuramoto–Sivashinsky / KPZ equations, 5 developmental/pathological suture regimes, Pass 3 `calvarial_shell_strain_model` (ellipsoid, plate layers, mechanotransduction law, muscle tractions), and SHA-256 locks | verified at runtime |
