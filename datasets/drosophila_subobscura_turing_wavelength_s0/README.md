# *Drosophila subobscura* — Turing wavelength at 3.0 µm (slot 3206)

Detail-scale reaction–diffusion wavelength for the Palearctic latitudinal wing-size cline fly, *Drosophila subobscura* (Sophophora obscura & willistoni clade).

- **Scale:** 3.0 µm (2.97 × 10⁻⁶ m, log10 = −5.527), tier 3/8 `3_subcellular_um`, detail variant. This is the Turing pigment-stripe / ocellus wavelength, not the 124 µm primordium slot (#2901).
- **Age:** Holocene (extant), **0 Ma**.
- **Law:** Turing wavenumber \(k_c\) from linear stability of Schnakenberg kinetics (Turing 1952; Schnakenberg 1979). Critical wavelength \(\lambda_c = 2\pi / k_c\) is pinned to 2.97 µm.
- **Chronoseries (Drosophila 35/41):** predecessor *Drosophila simulans* (cellular stripe, 82.56 µm); this slot; successor *Drosophila suzukii* (wing-spot scale, 98.45 µm). All three are Holocene. What changes at this scale is the selected wavenumber: \(k\) is about 27.8× the simulans wavenumber and about 33.1× the suzukii wavenumber. On a patch the length of the simulans wavelength, this detail law packs 27 modes where the predecessor law packs one.

Huey et al. (2000, Science 287:308–309) report the empirical wing-size fact used as natural history: Old World wing length increases with latitude; North American populations showed no cline one decade after introduction; after two decades a cline had evolved and largely converged on the ancestral cline. That paper does **not** measure a 3 µm pigment wavelength. The 3.0 µm figure is the assigned detail-scale phenomenon at which the Turing law is evaluated.

## Specimen

`drosophila_subobscura_specimen.jpg` — Martin Cooper, *Drosophila (Sophophora) subobscura* (male) collage, Ipswich, 2 Dec 2015. Wikimedia Commons, **CC BY 2.0**. Identification confirmed by Paul Beuk. SHA-256 `5366640b506b531749072d3e4097e68630b8132fd433df786b50e2e16c89711d`.

## Reproduce

```bash
python3 models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py
```

Pass 1 (`--pass 1`) is the pure-field integration. The default command is Pass 2: diffusion sweep, equal-diffusion ablation, specimen-mask composite, and the linear-regime dispersion goodness-of-fit. Deterministic seed `3206`. Stack: NumPy, SciPy, Pillow, Matplotlib.
