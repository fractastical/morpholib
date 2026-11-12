# *Eusthenopteron foordi* — 104 µm Turing wavelength (slot 3225)

Late Devonian (Frasnian, **379 Ma**) tristichopterid. This slot is the **104 µm** (0.0001042 m, log10 = −3.982) detail scale of tier `5_primordium_100um`, variant `s0`: the fundamental Turing pigment / ocellus wavelength, not the 5.7 mm organ-scale flank bar modelled in `models/1980-jarvik-eusthenopteron-foordi-turing-wavelength.py`.

Chronoseries **1/1**. There is no second *Eusthenopteron* time slice. At fixed wavenumber \(k_c = 2\pi / 104\,\mu\mathrm{m}\), the registry length of the morphological successor *Panderichthys rhombolepis* (0.165 m) holds more wavelengths than the 0.140 m *Eusthenopteron* length. That mode-count difference is the scale-specific change formalized here.

## Run

```bash
python3 models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/eusthenopteron_foordi_turing_wavelength_s0_comparison.png`
- `docs/media/eusthenopteron_foordi_turing_wavelength_s0_growth.gif`
- `datasets/eusthenopteron_foordi_turing_wavelength_s0/metrics.json`

## Specimen

`eusthenopteron_foordi_specimen.jpg` is a photograph of a real fossil mount, not the species-gallery life reconstruction.

- Source: Wikimedia Commons, [File:Eusthenopteron foordi 1.jpg](https://commons.wikimedia.org/wiki/File:Eusthenopteron_foordi_1.jpg)
- Author: Ghedoghedo
- License: CC BY-SA 3.0
- Where photographed: Muséum des Sciences naturelles, Brussels
- SHA-256: `ad5986d4a4f2a605df6547209af63db7d5e6b58e040fefcbf6d269622f2986ce`

The photograph has no scale bar. Micrometres are the model’s length unit (one selected wavelength = 104.2 µm). Pixels in the museum photo are not converted to micrometres.

## What the model does and does not claim

Schnakenberg kinetics (Schnakenberg 1979) are parameterized so the linear Turing wavenumber (Turing 1952) sits at \(k_c = 2\pi\) per 104.2 µm. Pigment-cell reaction–diffusion is transferred from living teleost skin (Kondo & Asai 1995; Nakamasu et al. 2009) onto the dermal field of a 379 Ma tristichopterid (Andrews & Westoll 1970). The fossil does not preserve chromatophore color. The animation draws the simulated ocellus field on the segmented body of the Brussels mount so the photograph and the simulation can be compared; it is not a recovered color pattern.
