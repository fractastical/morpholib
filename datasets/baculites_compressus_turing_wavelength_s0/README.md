# Baculites compressus — Turing wavelength at 94 µm

Slot 3131 of the 10,000-Animal Basic Look (detail variant `s0`).

- **Animal:** *Baculites compressus* Say, a Late Cretaceous uncoiled straight-shaft heteromorph ammonite.
- **Scale:** 94 µm (9.416 × 10⁻⁵ m, log10 = −4.026), tier `4_cellular_10um`. This is the critical Turing pigment wavelength λ_c = 2π / k_c, not the 4.3 mm organ-scale rib of sibling slot 1239.
- **Age:** 73.5 Ma, Late Cretaceous, *Baculites compressus* Zone, Pierre Shale (upper Campanian).
- **Lineage:** Cephalopod Fractal Suture Fossil Series. Baculites chronoseries 1/1. The predecessor in the series is coiled *Placenticeras meeki*; the morphological successor is irregular *Nipponites mirabilis* (Turonian, so older in clock time — the series order is suture geometry, not a literal ancestor–descendant chain).

## What changes at this scale

The Schnakenberg chemistry (a, b, D_u, diffusion ratio) is held fixed in the cell frame. The quantity that changes along the series is the aperture factor χ, which converts the cell-frame critical wavelength into a physical pigment spacing:

| Series position | Taxon | Age used here | χ | λ_c |
| --- | --- | --- | --- | --- |
| Predecessor | *Placenticeras meeki* | ~80.6 Ma (middle Campanian) | 3/2 | 141.24 µm |
| **This slot** | ***Baculites compressus*** | **73.5 Ma** | **1** | **94.16 µm** |
| Successor | *Nipponites mirabilis* | ~90.4 Ma (Turonian; morphologically later) | 6215/9416 | 62.15 µm |

χ = 1 is the normalization that places *B. compressus* on the assigned 94 µm detail wavelength. The other two values are geometric predictions (rounded involute aperture versus whorl recontact), not measurements of fossil pigment.

## Specimen

`baculites_compressus_specimen.jpg` is Kevmin’s photograph of a polished *B. compressus* section (Wikimedia Commons, CC BY-SA 3.0). Six sutures are visible on a 2.7 cm fragment from the Bearpaw Formation, Montana. The Bearpaw is broadly coeval with the Pierre Shale; the numeric age used in the model remains 73.5 Ma.

## Reproduce

```bash
python3 models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py
```

Outputs:

- `docs/media/baculites_compressus_turing_wavelength_s0_comparison.png`
- `docs/media/baculites_compressus_turing_wavelength_s0_growth.gif`

Governing law: Turing wavenumber k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v) for the Schnakenberg Jacobian (Schnakenberg 1979; Turing 1952). Mollusc-shell pigmentation is the biological transfer from Meinhardt & Klingler (1987). The coiled early whorl that makes a spiral pigment record possible is the subject of the 1892 Geological Magazine note.
