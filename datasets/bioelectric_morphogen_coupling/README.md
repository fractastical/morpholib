# Pietak & Levin (2017) / (2016) Bioelectric-Morphogen Coupling Reference Dataset

Primary reference dataset and biological specimen metadata for `2017-pietak-bioelectric-morphogen-coupling.py` (Slot 082 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Pietak & Levin (2017) *J. R. Soc. Interface* 14:20170425 & Pietak & Levin (2016) *Front. Bioeng. Biotechnol.* 4:55

1. **Electrodiffusive Nernst–Planck Morphogen Flux Through Voltage-Gated Gap Junctions:**

$$J_s(x,t) = -D_s\, G_{\text{gj}}(\Delta V)\left(\frac{\partial c}{\partial x} + \frac{z F}{R T}\, c\, \frac{\partial V}{\partial x}\right), \qquad G_{\text{gj}}(\Delta V) = \frac{(1 - \eta_{\text{oct}})\, G_0}{1 + (\Delta V / V_0)^2}$$

where $c(x,t)$ is intracellular serotonin (5-HT, valence $z = +1$ at physiological pH), $V(x,t)$ is resting membrane potential ($V_{\text{ant}} \approx -20\text{ mV}$, $V_{\text{post}} \approx -60\text{ mV}$ in wild-type planarians), and $V_T = \frac{RT}{F} \approx 25.85\text{ mV}$ is the thermal voltage.

2. **Exact Zero-Flux Nernst–Planck Boltzmann Equilibrium Concentration Ratio:**

$$\frac{c_{\text{post}}}{c_{\text{ant}}} = \exp\!\left(-\frac{z F}{R T}(V_{\text{post}} - V_{\text{ant}})\right) = \exp\!\left(-\frac{z(V_{\text{post}} - V_{\text{ant}})}{V_T}\right)$$

For $z = +1$, $V_{\text{ant}} = -20\text{ mV}$, $V_{\text{post}} = -60\text{ mV}$ ($\Delta V = V_{\text{post}} - V_{\text{ant}} = -40\text{ mV}$), cationic 5-HT electrophoreses down the electrical potential gradient into the hyperpolarized posterior wound with equilibrium enrichment ratio $c_{\text{post}}/c_{\text{ant}} = \exp(40 / 25.85) \approx 4.70$.

3. **Cyclic-AMP / PKA / $\beta$-Catenin / ERK Feedback Loop & Cryptic Two-Headed (`2H0T`) Memory Persistence:**

$$\tau_M \frac{\partial M}{\partial t} = \frac{c^2}{K_c^2 + c^2} + \alpha_{\text{loop}} \frac{B^2}{K_B^2 + B^2} - M$$

$$\tau_B \frac{\partial B}{\partial t} = \frac{M^2}{K_M^2 + M^2}\frac{1}{1 + (E / K_E)^2} - B + D_B \frac{\partial^2 B}{\partial x^2}, \qquad \tau_E \frac{\partial E}{\partial t} = \frac{1}{1 + (B / K_{BE})^2} - E + D_E \frac{\partial^2 E}{\partial x^2}$$

Transient 8-octanol gap-junction blockade ($\eta_{\text{oct}} = 0.92$) collapses posterior 5-HT electrophoresis ($c_{\text{post}}/c_{\text{ant}} \to 1$), flipping the posterior blastema into the anterior ERK / depolarized $V_{\text{mem}}$ (`2H0T`) attractor, which persists permanently after 8-octanol washout ($\eta_{\text{oct}} \to 0$).

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `planarian_ap_polarity_specimen.jpg` | Dorsal whole-mount micrograph of *Schmidtea mediterranea* (Wikimedia Commons `File:Smediterranea.jpg`, CC BY 2.5, 1493×564 px) showing anterior head, bilateral eyespots (ocelli), pigmented trunk, and tapering tail (`1H1T` wild-type polarity) | `466029cc0561067890bb0f69d224d8f799745eeb67a0699246107611737fb546` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier for Slot 082 | `bc8a4a2c8e734e1dd555ef8630edf4c34c798ffd1588619a0289290cc42b8004` |
| `paper_citations.bib` | BibTeX entries for Pietak & Levin (2017), Pietak & Levin (2016), Durant et al. (2017), and Oviedo et al. (2010) | `c446e775222f552c0edf0df27f8e7413b1ba68098018a3f3ed604b718bd7cbe7` |
| `bioelectric_morphogen_coupling_reference.json` | Canonical Nernst–Planck parameters, Boltzmann equilibrium ratio, 6 experimental regeneration regimes (`1H1T` WT, `2H0T` octanol, `2H0T` cryptic washout memory, `0H2T` SCH-28080/forskolin, `2H0T` $z=0$ control, `1H1T` bioelectric rescue), and SHA-256 locks | verified at runtime |
