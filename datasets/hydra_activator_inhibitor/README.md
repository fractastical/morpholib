# Gierer & Meinhardt (1972) *Hydra vulgaris* Activator-Inhibitor Reference Dataset

Primary reference dataset and biological specimen metadata for `1972-gierer-hydra-activator-inhibitor.py` (Slot 001 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Gierer & Meinhardt (1972) *Kybernetik* 12:30–39, Eqs. (15a)–(15b)

$$\frac{\partial a}{\partial t} = \rho_0 \rho(x) + \frac{c \rho(x) a^2}{h} - \mu_a a + D_a \frac{\partial^2 a}{\partial x^2}$$

$$\frac{\partial h}{\partial t} = c' \rho(x) a^2 - \mu_h h + D_h \frac{\partial^2 h}{\partial x^2}$$

where:
- $a(x,t)$: short-range autocatalytic head/tentacle activator concentration
- $h(x,t)$: long-range lateral inhibitor concentration
- $\rho(x)$: morphogen source density along the *Hydra* apico-basal body column
- $\mu_a, \mu_h$: degradation rates ($\mu_h > \mu_a$ required for homogeneous stability)
- $D_a, D_h$: diffusion coefficients ($D_h / D_a > d_c = (\sqrt{2}+1)^2 \approx 5.8284$ required for Turing-Gierer-Meinhardt instability)

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `hydra_vulgaris_specimen.jpg` | Dark-field whole-mount photograph of *Hydra vulgaris* (Wikimedia Commons `File:Hydravulgaris.jpg`, CC BY-SA 3.0, 640×614 px) showing basal disc, tubular body column, apical hypostome, and 6 radial tentacles | `007d127bcfa8501d3b9737ffaf7a512efec37db8a0c83a5334568b513da214f9` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier (`morphoeval/scout_literature.py --slot 1`) | `b57d0d923283daaf442c82adfb0b709dc507e23ec0d05ae8ea4e9926782e6478` |
| `paper_citations.bib` | BibTeX entries for Gierer & Meinhardt (1972), Meinhardt & Gierer (1974), Meinhardt (1993), and Meinhardt (2012) | `49b5c0a5531a00fe76fed6d969e9e0af0c9c4ec23499de11893523405f080da6` |
| `hydra_activator_inhibitor_reference.json` | Canonical parameters, exact Jacobian & critical diffusivity ratio invariants, regeneration/tentacle ring benchmarks, and SHA-256 locks | verified at runtime |
