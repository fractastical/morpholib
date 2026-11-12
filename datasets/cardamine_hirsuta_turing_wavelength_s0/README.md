# Cardamine hirsuta — Turing wavelength at 154 µm (slot 3140)

Holocene extant hairy bittercress (*Cardamine hirsuta*), detail-scale Turing pigment-ocellus wavelength.

- **Scale:** 154 µm (0.0001541 m, log10 = −3.812), tier `5_primordium_100um`, detail variant.
- **Age:** Holocene (Extant), **0 Ma**. Cardamine chronoseries **1/1** — no predecessor and no successor chronospecies.
- **Law:** Turing wavenumber \(k_c\) for a Schnakenberg reaction–diffusion system, readout as ocelli (spots with a ring) inside a young leaflet.
- **Specimen:** `cardamine_hirsuta_specimen.jpg`, vendored from Wikimedia Commons `File:Cardamine hirsuta.jpg` (public domain). SHA-256 is recorded in `cardamine_hirsuta_turing_wavelength_s0_reference.json`.

## What changes at this scale

The local wavenumber is set by the Jacobian and the diffusion ratio, not by the RCO growth repression that dissects the leaf at millimetre-to-centimetre scales (Vlad et al. 2014; Kierzkowski et al. 2019). On the 1.233 mm detail patch used here, eight classical wavelengths of 154.1 µm fit. A single epidermal cell (~16 µm) is shorter than \(\lambda_c\), so the same mode is cut off. That admissibility change is the scale comparison. The chronoseries itself does not supply an earlier or later named species: the age is the fixed point 0 Ma.

## Run

```bash
python3 models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/cardamine_hirsuta_turing_wavelength_s0_comparison.png`
- `docs/media/cardamine_hirsuta_turing_wavelength_s0_growth.gif`

## Claim boundary

The 154.1 µm figure is the classical Turing wavelength of this parameter point, pinned to the slot scale. It is not a length reported by the Cardamine trichome or leaf-margin papers. Those papers justify the organism, the compound-leaf domain, and the existence of a periodic epidermal pattern at this tier. The transfer from that biology onto Schnakenberg pigment ocelli is a reconstruction.
