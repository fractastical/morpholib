# Dataset: Osmotic Matrix Swelling, Localized Cell-Death Delamination & Föppl–von Kármán Radial Spokes in *Bacillus subtilis* Biofilms (`bacillus_biofilm_wrinkling`)

## Primary Literature & DOIs

1. **Asally M, Kittisopikul M, Rue P, Du Y, Hu Z, Çağatay T, Robinson AB, Lu H, Garcia-Ojalvo J, Süel GM (2012)**.
   *Localized cell death focuses mechanical forces during 3D patterning in a biofilm.* **Proceedings of the National Academy of Sciences** 109(46):18891–18896.
   DOI: [`10.1073/pnas.1212429109`](https://doi.org/10.1073/pnas.1212429109)
2. **Yan J, Fei C, Mao S, Moreau A, Wingreen NS, Košmrlj A, Stone HA, Bassler BL (2019)**.
   *Mechanical instability and interfacial energy drive biofilm morphogenesis.* **Nature Physics** 15(5):456–462.
   DOI: [`10.1038/s41567-018-0392-9`](https://doi.org/10.1038/s41567-018-0392-9)
3. **Trejo M, Douarche C, Bailleux V, Poulard C, Mariot S, Regeard C, Raspaud E (2013)**.
   *Elasticity and wrinkled morphology of Bacillus subtilis pellicles.* **Proceedings of the National Academy of Sciences** 110(6):2011–2016.
   DOI: [`10.1073/pnas.1217178110`](https://doi.org/10.1073/pnas.1217178110)
4. **Seminara A, Angelini TE, Wilking JN, Vlamakis H, Ebrahim S, Kolter R, Weitz DA, Brenner MP (2012)**.
   *Osmotic spreading of Bacillus subtilis biofilms driven by an extracellular matrix.* **Proceedings of the National Academy of Sciences** 109(4):1116–1121.
   DOI: [`10.1073/pnas.1102300108`](https://doi.org/10.1073/pnas.1102300108)

## Governing Mathematical & Biomechanical Framework

During *Bacillus subtilis* (NCIB 3610) biofilm morphogenesis on compliant nutrient agar, extracellular polysaccharide (EPS) and amyloid fiber (TasA) production drives osmotic swelling and in-plane compressive growth strain $\epsilon_g$, generating biaxial compressive stress:

$$\sigma_0 = \frac{E_f}{1 - \nu_f} \epsilon_g$$

where $E_f$ and $\nu_f$ are the biofilm Young's modulus and Poisson ratio, and $\bar{E}_f = E_f / (1 - \nu_f^2)$, $\bar{E}_s = E_s / (1 - \nu_s^2)$ are the plane-strain moduli of the biofilm (thickness $h$) and compliant agar substrate.

1. **Föppl–von Kármán Bilayer Wrinkling Wavelength & Critical Strain:**
   $$\lambda_c = 2\pi h \left(\frac{\bar{E}_f}{3 \bar{E}_s}\right)^{1/3}, \qquad \epsilon_c = \frac{1}{4}\left(\frac{3 \bar{E}_s}{\bar{E}_f}\right)^{2/3}.$$
2. **Localized Cell-Death Delamination Focusing (`Asally et al. 2012 PNAS`):**
   Spatiotemporally patterned basal cell death ($\phi_{\text{dead}}(r,\theta) \in [0,1]$) locally degrades biofilm–agar interfacial adhesion toughness $G_c \to G_{c,0}(1 - \eta_d \phi_{\text{dead}})$, lowering the local effective buckling threshold:
   $$\epsilon_{c,\text{eff}}(\phi_{\text{dead}}) = \epsilon_c (1 - \gamma_d \phi_{\text{dead}}),$$
   focusing lateral compressive forces into a central concentric ring ($r = R_{\text{ring}}$) and radiating outer spokes, and allowing artificial cell-death/mechanical-score templates ($m \in \{6, 8\}$) to deterministically guide custom radial spoke architectures.
3. **Radial Spoke Number Scaling (`Yan et al. 2019 Nature Physics`):**
   $$N_{\text{spokes}} = \frac{2\pi R_{\text{ring}}}{\lambda_c} = \frac{R_{\text{ring}}}{h} \left(\frac{3 \bar{E}_s}{\bar{E}_f}\right)^{1/3},$$
   scaling linearly with ring radius $R_{\text{ring}}$, inversely with biofilm thickness $h$, and as $(E_s / E_f)^{1/3}$ with agar stiffness. Matrix-deficient $\Delta epsA\text{-}O$ / $\Delta tasA$ mutants fail to generate osmotic swelling strain ($\epsilon_g < \epsilon_c$) and remain flat and unwrinkled ($N_{\text{spokes}} = 0$).

## Vendored Files

- `scout_manifest.json` — Literature & specimen discovery manifest for Slot 094.
- `paper_citations.bib` — BibTeX citations with DOIs for Asally et al. (2012), Yan et al. (2019), Trejo et al. (2013), and Seminara et al. (2012).
- `bacillus_biofilm_wrinkling_reference.json` — Quantitative reference tables (7-point agar stiffness sweep $E_s \in [0.5, 10.0]\text{ kPa}$, 6-point biofilm thickness sweep $h \in [40, 140]\,\mu\text{m}$, genetic/mechanical controls including $\Delta eps/\Delta tasA$ mutant and artificial score-guided spoke patterns, and SHA-256 digests).
- `bacillus_biofilm_specimen.jpg` — Wikimedia Commons CC BY-SA 3.0 photograph of wrinkled *Bacillus subtilis* macrocolony biofilms on nutrient agar (`File:Bacillus subtilis colonies.jpg`, 1200×1243 px).
