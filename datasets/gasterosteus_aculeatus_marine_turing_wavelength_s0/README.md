# *Gasterosteus aculeatus_marine* — 52 µm Turing wavelength (slot 3232)

Late Pleistocene marine ancestor (15 ka, **0.015 Ma**) of the threespine stickleback, complete-plate morph (32 lateral plates). This slot is the cellular-tier detail of Turing pigment-stripe / ocellus wavelength selection: characteristic length **52 µm = 5.2×10⁻⁵ m** (log10 = −4.284), scale tier `4_cellular_10um`.

It is not the 2.4 mm organ-tier sibling (`models/2005-colosimo-gasterosteus-aculeatus-marine-turing-wavelength.py`). At 52 µm the state variable is chromatophore spacing on the flank integument between lateral plates, not the millimetre bar or plate itself.

## Chronoseries (Gasterosteus pelvic-reduction series, node 4/6)

| Node | Age | Plate scenario used here | λ at this tier |
| --- | --- | --- | --- |
| Predecessor *G. doryssus_late* | 9.99 Ma (Truckee Fm.) | low-armor analogy N_eff = 8 | 104 µm |
| **This node, marine ancestor** | **0.015 Ma** | **complete morph N = 32** | **52 µm** |
| Successor *G. aculeatus* | 0 Ma (living species complex) | polymorphic mean N_eff = 14.22 | 78 µm |

The plate-to-wavelength map λ(N) = 52 µm × √(32/N) is a reconstruction in the script (transfer boundary). Bell, Baumgartner & Olson (1985) document temporal change in *G. doryssus* armor characters; they do not publish a 52 µm melanophore spacing. Colosimo et al. (2004, 2005) show that Eda alleles switch the low morph (about eight or fewer plates) and the complete marine morph. N_eff = 8 for the predecessor is that low-morph analogy, not a counted Truckee specimen.

## Specimen

`gasterosteus_aculeatus_marine_turing_wavelength_s0_specimen.jpg`

- Photograph: three-spined stickleback, *Gasterosteus aculeatus*, Lauterach, Munderkingen, Germany, 21 Aug 2020
- Author: Holger Krisp (Wikimedia Commons user Holleday)
- License: CC BY 4.0
- Source: https://commons.wikimedia.org/wiki/File:Dreistachlige_Stichling_Gasterosteus_aculeatus.jpg
- SHA-256: `0c9e2c6dfe107e95adac946fca94b5887561d5229ff7e598385f0664b6cb456c`

The file is an extant anatomical reference. It is not a 15 ka fossil.

## Run

```bash
python3 models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py
```

Outputs:

- `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_comparison.png`
- `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_growth.gif`
- `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_metrics.json`
