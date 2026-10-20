# Dataset: Solar-Powered Diurnal Thermosiphon Ventilation & Convective Pheromone Morphogenesis of Termite Mounds (`termite_mound_convective_chimneys`)

## Primary Literature & DOIs

1. **Ocko SA, King H, Andreen D, Bardunias P, Turner JS, Soar R, Mahadevan L (2017)**.
   *Solar-powered ventilation of African termite mounds.* **Proceedings of the National Academy of Sciences** 114(40):10620–10624.
   DOI: [`10.1073/pnas.1713534114`](https://doi.org/10.1073/pnas.1713534114)
2. **Ocko SA, Heyde A, Mahadevan L (2019)**.
   *Morphogenesis of termite mounds.* **Proceedings of the National Academy of Sciences** 116(9):3379–3384.
   DOI: [`10.1073/pnas.1818759116`](https://doi.org/10.1073/pnas.1818759116)
3. **King H, Ocko S, Mahadevan L (2015)**.
   *Termite mounds harness diurnal temperature oscillations for ventilation.* **Proceedings of the National Academy of Sciences** 112(37):11589–11593.
   DOI: [`10.1073/pnas.1423242112`](https://doi.org/10.1073/pnas.1423242112)
4. **Korb J (2003)**.
   *Thermoregulation and ventilation of termite mounds.* **Naturwissenschaften** 90(5):212–219.
   DOI: [`10.1007/s00114-002-0401-4`](https://doi.org/10.1007/s00114-002-0401-4)
5. **Oberst SM, Lai JCS, Martin R, Halkon B (2020)**.
   *Revisiting stigmergy in light of multi-functional, biogenic, termite structures as communication channel.* **Computational and Structural Biotechnology Journal** 18:2522–2534.
   DOI: [`10.1016/j.csbj.2020.08.012`](https://doi.org/10.1016/j.csbj.2020.08.012)

## Governing Mathematical & Architectural Framework

Fungus-cultivating termites (*Macrotermes michaelseni* in Southern Africa and *Odontotermes obesus* in South Asia) construct monumental earthen cathedral mounds up to 3–5 meters tall that function as external solar-powered lungs and self-organize via convective advection of volatile cement pheromone:

1. **Diurnal Solar-Thermal Buoyancy Thermosiphon & Mass Continuity (`King et al. 2015 PNAS; Ocko et al. 2017 PNAS`):**
   Thin exterior surface flutes respond rapidly to diurnal solar heating and nocturnal radiative cooling, whereas the deep central chimney is thermally buffered with large thermal inertia. The resulting radial temperature difference $\Delta T(t) = T_{\text{flute}}(t) - T_{\text{chimney}}(t)$ drives a closed convective thermosiphon loop satisfying Boussinesq Darcy–Brinkman momentum balance and mass conservation:
   $$u_{\text{flute}}(t) = K_{\text{buoy}} \bigl(T_{\text{flute}}(t) - T_{\text{chimney}}(t)\bigr), \qquad A_{\text{chimney}} u_{\text{chimney}}(t) + A_{\text{flutes}} u_{\text{flute}}(t) = 0.$$
   During the day ($\Delta T > 0$), buoyant air rises in the exterior flutes ($u_{\text{flute}} > 0$) and descends down the central chimney ($u_{\text{chimney}} < 0$); at night ($\Delta T < 0$), the circulation reverses ($u_{\text{flute}} < 0, u_{\text{chimney}} > 0$).
2. **Taylor–Aris Oscillatory $\text{CO}_2$ Clearance (`King et al. 2015 PNAS`):**
   Oscillatory diurnal shear flow with amplitude $u_{\text{amp}}$ enhances effective axial gas transport via Taylor–Aris / pendelluft dispersion,
   $$D_{\text{eff}} = D_0 + \alpha_{\text{TA}} u_{\text{amp}}^2 > D_0 \quad (u_{\text{amp}} > 0),$$
   reducing subterranean nest $\text{CO}_2$ from stagnant toxic levels ($\sim 4.82\%$) to $\sim 1.42\%$ at midday and $\sim 2.08\%$ at night.
3. **Convective Péclet Pheromone Advection & Cathedral Spire Morphogenesis (`Ocko, Heyde & Mahadevan 2019 PNAS`):**
   Volatile building pheromone / $\text{CO}_2$ concentration $c(\mathbf{x}, t)$ emitted from the nest satisfies advection–diffusion–decay
   $$\partial_t c + \mathbf{u} \cdot \nabla c = D_c \nabla^2 c - \gamma_c c,$$
   coupled to threshold soil deposition $v_n = k_{\text{dep}} \max(0, c|_{\partial\Omega} - c_{\text{th}}) - k_{\text{curv}} \kappa$. Upward convective advection ($Pe = u_0 H_0 / D_c > 0$) focuses pheromone at the mound apex, elongating the mound aspect ratio $H/R = 1 + \eta_{Pe} Pe > 1$ into a steep conical spire ($H/W = 1.95$ at $Pe = 3.2$) tilted equatorward toward the noon sun ($\theta_{\text{tilt}} = \chi_{\text{solar}} |\phi_{\text{lat}}| \approx 18.8^\circ\text{N}$ at $19.2^\circ\text{S}$ in Namibia), whereas isothermal pure diffusion ($Pe = 0$) yields a hemispherical dome ($H/R = 1.0, H/W = 0.50$).

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 097.
- `paper_citations.bib` — BibTeX citations with DOIs for Ocko et al. (2017, 2019), King et al. (2015), Korb (2003), and Oberst et al. (2020).
- `termite_mound_convective_chimneys_reference.json` — Quantitative reference tables (24-hour diurnal thermosiphon trajectory, 7-point convective Péclet morphogenesis sweep, 6-point latitude spire tilt sweep, taxon/ablation controls, and SHA-256 digests).
- `macrotermes_cathedral_mound_specimen.jpg` — CC BY-SA 4.0 photograph of a cathedral termite mound showing vertical surface flutes and central spire (`File:Cathedral Termite Mound.jpg`, 600×900 px).
