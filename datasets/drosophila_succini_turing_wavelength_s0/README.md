# Drosophila succini — Turing wavelength, detail scale s0 (slot 3172)

*Drosophila succini* Hennig, 1965, is the Middle Eocene (Lutetian Baltic amber, **44 Ma**) stem of the Cenozoic amber drosophilid chronoseries (index 1/41). This package is the **2.8 µm** detail scale (exactly **2.759e-06 m**, log10 = −5.559, tier `3_subcellular_um`): the Schnakenberg critical wavelength of an ocellar pigment micro-pattern. It is not the 114 µm primordium-scale sibling.

## Law

Schnakenberg (1979) kinetics. The critical wavenumber is \(k_c = \arg\max_k \sigma(k)\), with \(\lambda_c = 2\pi / k_c\). \(D_u\) is solved so that \(\lambda_c = 2.759\,\mu\mathrm{m}\) at diffusion ratio \(r = D_v/D_u = 18.72\). Turing onset for these kinetics is \(r \approx 10.77\).

## What changes at this age and scale

- **Predecessor** (pre-Baltic acalyptrate stem, not a named *Drosophila* chronospecies): \(r = 8.40 < r_{\mathrm{onset}}\), so \(\sigma_{\max} < 0\) and there is no 2.8 µm lattice.
- **This node (44 Ma):** \(r = 18.72\), \(\lambda = 2.759\,\mu\mathrm{m}\).
- **Successor *Drosophila berryi* (Dominican amber, 19 Ma):** same \(r\), both diffusivities scaled by \((98.43/114)^2\), giving \(\lambda \approx 2.382\,\mu\mathrm{m}\). That factor is the registry ratio of the two species' primordium-scale Turing wavelengths, transferred onto this detail scale. It is not a new measurement of the fossil.

## Specimen

Wikimedia Commons returned no bitmap for *Drosophila succini* or *Electrophortica succini*. The image `drosophila_succini_turing_wavelength_s0_specimen.jpg` is the repository species-gallery card (SHA-256 `1a0de2972916cbfa2e2b0e21ebeb0aed9e2f4f99f11b26993c29f2c9b36464cd`). It is a synthetic amber spiral with a 100 µm scale bar, not a photomicrograph of the Hennig holotype. The registry DOI `10.1163/187631265X00018` did not resolve on Crossref or OpenAlex; the Hennig paper used here is the BHL/Zenodo record `10.5281/zenodo.10748923`.

## Run

```bash
python3 models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py --self-test
```

Outputs: `docs/media/drosophila_succini_turing_wavelength_s0_comparison.png` and `docs/media/drosophila_succini_turing_wavelength_s0_growth.gif`.
