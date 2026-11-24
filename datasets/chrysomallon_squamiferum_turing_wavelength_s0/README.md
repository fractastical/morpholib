# Chrysomallon squamiferum — Turing wavelength, detail scale s0 (slot 3147)

Holocene (Extant) scaly-foot gastropod at geological age **0 Ma**. The assigned phenomenon is the **2.0 nm** (2.0×10⁻⁹ m, log10 ≈ −8.699) detail-scale critical wavelength of a Turing reaction–diffusion field (tier `1_molecular_nm`, variant `s0`).

This is chronoseries **1/1** for *Chrysomallon*: the only node is the living species. There is no successor chronospecies. The comparative predecessor is a sclerite-less peltospirid such as *Gigantopelta* (Chen et al. 2015, Zool. J. Linn. Soc.), which shares the vent habitat but does not mineralize foot sclerites with iron sulfide. At this scale, relative to that predecessor and relative to the coarser **20 nm** nanocrystal-packing wavelength (a different phenomenon size, not this slot’s parameters), the extant animal’s detail mode is a tenfold-shorter critical wavelength, λ_c = 2.0 nm ≈ 2 greigite unit cells (a ≈ 0.988 nm; Skinner, Erd & Grimaldi 1964).

## What the sources actually measured

- Warén et al. 2003, Science: dermal sclerites mineralized with pyrite and greigite.
- Yao et al. 2010, PNAS: multilayer iron-sulfide armor (mechanical design, not a Turing equation).
- Chen et al. 2015, J. Molluscan Studies: genus *Chrysomallon*, species *C. squamiferum*.
- Sun et al. 2020, Nat. Commun.: scale and shell biomineralization use different transcription-factor subsets.
- Wong et al. 2025, Commun. Biol.: sclerite matrix proteins co-opted relative to sclerite-less *Gigantopelta aegis*.
- Turing 1952 and Schnakenberg 1979: the wavelength-selection law implemented here.

No paper reports a 2.0 nm pigment stripe or ocellus on this snail. The 2.0 nm figure is the slot’s detail-scale critical wavelength, transferred onto greigite-cluster spacing. Aqueous morphogen Turing wavelengths are micrometres to millimetres; a 2 nm mode requires matrix-confined diffusivities. That transfer is stated in the reference JSON and is not a claim that Yao et al. solved a reaction–diffusion equation.

## Specimen

`chrysomallon_squamiferum_specimen.jpg` is a JPEG derivative of the CC BY 4.0 photograph of paratype MNHN-IM-2000-30072 by Marie Hennion (MNHN molluscs collection, accessed via Wikimedia Commons). SHA-256 is pinned in `chrysomallon_squamiferum_turing_wavelength_s0_reference.json`.

## Reproduce

```bash
python3 models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py
```

Outputs:

- `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_comparison.png`
- `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_growth.gif`

Governing law: Schnakenberg kinetics with Turing wavenumber

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π / k_c = 2.0 nm.

Slot-specific parameters (not the 20 nm sibling): a = 0.062, b = 1.084, D_v/D_u = 36, seed 3147, age 0 Ma.
