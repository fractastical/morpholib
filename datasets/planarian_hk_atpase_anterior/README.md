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
| `schmidtea_mediterranea_specimen.jpg` | Dorsal bright-field whole-mount of *Schmidtea mediterranea* (720×500 px PlanformDB photograph vendored at commit `398c37e0`; supersedes the Pass 1–2 Wikimedia `File:Smediterranea.jpg` composite) with automated segmentation length 626 px (`1H1T` wild-type polarity) | `0dde9f5e83a87dd9bdc3627e19ebc891cb4f8e99cf632fe637bccf082abfb990` |
| `scout_manifest.json` | Step 1 scout dossier plus `committed_specimen` record for the vendored PlanformDB photograph (`morphoeval/scout_literature.py --slot 81`) | `8f66f6190b32be14c7b31601bc2084f1595f44e65290df46f513331172d8e6ed` |
| `paper_citations.bib` | BibTeX entries for Beane et al. (2011), Oviedo et al. (2010), Pietak & Levin (2017), and Durant et al. (2017) | `4978cdbc97ad314e55b71348a3fd0a831887eaf7bad4b10a76e834070964277f` |
| `planarian_hk_atpase_anterior_reference.json` | Canonical bioelectric parameters, bistable $V_{\text{mem}}$ attractors, 5 experimental regeneration regimes (`1H1T`, `0H2T` SCH-28080, `2H0T` ivermectin, `2H0T` 8-octanol, `0H2T` nifedipine), and SHA-256 locks | verified at runtime |
