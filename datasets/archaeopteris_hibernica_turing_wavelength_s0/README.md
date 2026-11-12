# Archaeopteris hibernica — 31 µm Turing wavelength (slot 3119)

Famennian (370 Ma) progymnosperm. This slot is the cellular detail scale of the planated megaphyll: a Schnakenberg reaction-diffusion field whose critical wavelength is **31 µm** (3.096×10⁻⁵ m, log10 = −4.509, tier `4_cellular_10um`).

The organism-scale sibling `models/1999-meyer-archaeopteris-hibernica-turing-wavelength.py` is a 1.5 mm stub. Parameters here are solved for this wavelength and this age, not copied from that file.

## What changes at 370 Ma / 31 µm

| node | age | habit | thickness | λ |
|---|---:|---|---:|---:|
| *Tetraxylopteris schmidtii* (predecessor) | 385 Ma | unplanated aneurophyte cortex | 42 µm | 66 µm |
| *Archaeopteris hibernica* (this slot) | 370 Ma | planated megaphyll lamina | 12.4 µm | 31 µm |
| *Elkinsia polymorpha* (successor) | 365 Ma | thinner cupulate frond | 6.8 µm | 21 µm |

Scaling used: λ(h) = 31 × (h / 12.4)^0.62 µm. Planation is what drops the wavelength from the cortical predecessor onto this lamina. Later epidermal thinning, not the ovule, drops it again toward *Elkinsia*.

## Model

Schnakenberg kinetics (1979) with Turing’s critical wavenumber (1952):

- f = a − u + u²v, g = b − u²v, a = 0.118, b = 0.764
- Du = 3.10 µm²/time, Dv = 74.40 µm²/time (ratio 24)
- γ = 0.664 per time, solved so the fastest-growing mode is 2π/31 µm⁻¹ on a periodic 186 µm square
- Equal-diffusion ablation (Du = Dv) is stable. Raising Dv/Du shortens λ.

## Specimen

`Archaeopteris_hibernica_NMNH.jpg` — public-domain photograph of an *Archaeopteris hibernica* frond at the National Museum of Natural History (User:Daderot, 2009; Wikimedia `File:Archaeopteris hibernica 2.JPG`). SHA-256 `20c480a93bc91bf301127067e410d2ba9ecb359403975c64b7ced1918e65f3b1`.

The photograph shows the planated frond. It does not resolve 31 µm epidermal cells. The growth GIF uses that frond as the lamina domain and veils it with the upscaled activator; the wavelength is measured on the cellular grid.

## Run

```bash
python3 models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py
```

Outputs: `docs/media/archaeopteris_hibernica_turing_wavelength_s0_comparison.png` and `docs/media/archaeopteris_hibernica_turing_wavelength_s0_growth.gif`.

## Sources

Crossref-checked DOIs are in `paper_citations.bib`. Primary organism paper: Meyer-Berthaud, Scheckler & Wendt (1999) Nature 398:700–701, doi:10.1038/19516. Law: Turing (1952) doi:10.1098/rstb.1952.0012. Kinetics: Schnakenberg (1979) doi:10.1016/0022-5193(79)90042-0. Planation: Boyce & Knoll (2002) Paleobiology 28:70–100.
