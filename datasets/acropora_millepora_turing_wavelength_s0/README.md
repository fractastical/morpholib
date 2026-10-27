# *Acropora millepora* — Holocene Turing wavelength, 385.2 µm (slot 3104)

Detail-scale reaction–diffusion model for one animal, one length, and one point in time.

- **Animal.** *Acropora millepora* (staghorn coral). II. Anaima (Invertebrata) → Zoophyta & Vermes (Cnidaria) → *Acropora*.
- **Scale.** 385.2 µm = 3.852×10⁻⁴ m (log10 = −3.414). Tier 5/8 `5_primordium_100um`, detail variant: Turing pigment-stripe / ocellus wavelength.
- **Age.** Holocene, extant, **0 Ma**. Chronoseries length 1/1. There is no predecessor chronospecies and no successor. The Eocene *Acropora* record (Wallace & Rosen 2006) is an out-of-series skeletal contrast: pigment does not fossilize, and colony architecture sits at a larger tier than 385.2 µm. On a ~1.5 mm corallite the same wavenumber would fit about four wavelengths; this slot is the fundamental mode.

## Law

Schnakenberg (1979) activator–inhibitor kinetics on a periodic 3.0816 mm coenenchyme patch (eight wavelengths). The selected wavenumber is the fastest-growing linear mode,

k_c = argmax σ(k), λ_c = 2π / k_c = 385.2 µm,

with D_v / D_u = 15.75. D_u is solved so that peak sits on 385.2 µm. The Murray band-center wavenumber is reported beside it and is not the target: above onset it is shorter. Equal diffusion is the negative control and does not pattern.

Host colour in this species is a GFP-like protein pattern (Dove et al. 2001; D'Angelo et al. 2008; Salih et al. 2000), not a skeletal measurement. The 385.2 µm figure is the assigned primordium-tier scale for the loop, not a published caliper spacing.

## Specimen

`acropora_millepora_gbr_lundgren2013.jpg` — CC BY 4.0 photograph of Great Barrier Reef colonies by Lundgren, Vera, Peplow, Manel and van Oppen, accompanying Lundgren et al. 2013 (*BMC Genetics* 14:9, doi:10.1186/1471-2156-14-9). Wikimedia Commons `File:Acropora millepora, GBR.jpg`. SHA-256 `94c2290198c33f3907ac1f4834b3696f63d39073d583ae7d6fe225e3e09e46b9`. The photograph is a colony, not a 385 µm micrograph; the fitness comparison says so.

## Run

```bash
python3 models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-acropora-millepora-turing-wavelength-s0.py
```

Seed 3104. Writes `docs/media/acropora_millepora_turing_wavelength_s0_comparison.png`, `docs/media/acropora_millepora_turing_wavelength_s0_growth.gif`, and `acropora_millepora_turing_wavelength_s0_metrics.json`.
