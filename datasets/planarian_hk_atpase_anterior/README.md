# Beane et al. (2011) Planarian H,K-ATPase Bioelectric $V_{\text{mem}}$ Anterior Polarity Reference Dataset

Primary reference dataset and biological specimen metadata for `2011-beane-planarian-hk-atpase-anterior.py` (Slot 081 of the Project Delphi & MorphoEval Campaign).

## Governing Equations — Beane et al. (2011) *Chem. Biol.* 18:77–89 & Pietak–Levin (2017)

1. **Bistable Membrane Potential & Voltage-Gated Gap-Junction PDE:**

$$C_m \frac{\partial V}{\partial t} = -g_{\text{bistable}} (V - V_{\text{hyper}})(V - V_{\text{thresh}})(V - V_{\text{depol}}) + I_{\text{HK}}(x) + I_{\text{IVM}}(x) + \frac{\partial}{\partial x}\left(G_{\text{GJ}}\left(\frac{\partial V}{\partial x}\right)\frac{\partial V}{\partial x}\right)$$

2. **Voltage-Gated Innexin Gap-Junction Conductance (Bell-Form Symmetry & Isolation):**

$$G_{\text{GJ}}(\Delta V) = \frac{(1 - \eta_{\text{oct}}) G_0}{1 + (\Delta V / V_0)^2}$$

3. **L-Type $\text{Ca}_{\text{v}}$ Calcium Influx Gate & Mutually Antagonistic Head ($H$) / Tail ($T$) Morphogen Cascade:**

$$\text{Ca}_{\text{ss}}(V) = \frac{1}{1 + \exp(-(V - V_{\text{thresh}})/k_V)}$$

$$\tau_H \frac{\partial H}{\partial t} = \text{Ca}_{\text{ss}}(V)(1 - c_T T) - H + D_H \frac{\partial^2 H}{\partial x^2}, \qquad \tau_T \frac{\partial T}{\partial t} = (1 - \text{Ca}_{\text{ss}}(V)) W(x)(1 - c_H H) - T + D_T \frac{\partial^2 T}{\partial x^2}$$

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `schmidtea_mediterranea_specimen.jpg` | Dorsal whole-mount photograph of *Schmidtea mediterranea* (Wikimedia Commons `File:Smediterranea.jpg`, CC BY 2.5, 1493×564 px) showing anterior head, bilateral eyespots (ocelli), pigmented trunk, and tapering tail (`1H1T` wild-type polarity) | `466029cc0561067890bb0f69d224d8f799745eeb67a0699246107611737fb546` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier (`morphoeval/scout_literature.py --slot 81`) | `9de72183ede698f04d87a137a1e4d3a235dbcac613bf9da6e5c553eef9d94294` |
| `paper_citations.bib` | BibTeX entries for Beane et al. (2011), Oviedo et al. (2010), Pietak & Levin (2017), and Durant et al. (2017) | `4978cdbc97ad314e55b71348a3fd0a831887eaf7bad4b10a76e834070964277f` |
| `planarian_hk_atpase_anterior_reference.json` | Canonical bioelectric parameters, bistable $V_{\text{mem}}$ attractors, 5 experimental regeneration regimes (`1H1T`, `0H2T` SCH-28080, `2H0T` ivermectin, `2H0T` 8-octanol, `0H2T` nifedipine), and SHA-256 locks | verified at runtime |
