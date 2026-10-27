# Bauhinia variegata — Turing wavelength at 69.12 µm

Slot 3132 of the 10,000-animal basic look. One species, one scale, one age.

- **Species:** *Bauhinia variegata* (orchid tree pod)
- **Scale:** 69.12 µm = 6.912×10⁻⁵ m (log10 = −4.160), tier `4_cellular_10um`, detail variant
- **Age:** Holocene, extant, **0 Ma**. Chronoseries length 1. There is no earlier chronospecies in this slot.
- **Law:** Turing wavenumber \(k_c\) on Schnakenberg kinetics, \(D_v/D_u = 16.9\)

At 0 Ma the living pod carries this cellular mode and the centimetre-scale valve at the same time. A 10 µm tier-centre wavelength would sit inside one epidermal cell and is not the mode selected here. Along the printed 1 cm bar the 69.12 µm wave repeats 144 times; a 145th does not fit. Armon, Efrati, Kupferman and Sharon (2011) describe the centimetre-scale opening of this species by incompatible curvature. That law is a scale boundary only. Its parameters are not used.

The 69.12 µm length is the assigned detail scale. It is not a published count of spots on this pericarp. On the vendored scan the 1 cm bar is 37 pixels, so 69.12 µm is 0.26 px and is not resolved. The photograph is the valve envelope. The PDE is the cellular carrier, drawn in the inset.

## Specimen

`bauhinia_variegata_specimen.jpg` — Roger Culos, Muséum de Toulouse MHNT.BOT.2011.3.22, Wikimedia Commons, CC BY-SA 3.0.

SHA-256: `300694d305567ca5b259ab023ea8e40a734061a3f55b78de1a23d30cbd2ffd3a`

## Run

```bash
python3 models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-bauhinia-variegata-turing-wavelength-s0.py
```

Outputs:

- `docs/media/bauhinia_variegata_turing_wavelength_s0_comparison.png`
- `docs/media/bauhinia_variegata_turing_wavelength_s0_growth.gif` (12 frames, 960×623)

Seed 3132. Periodic boundary conditions on a flat patch.
