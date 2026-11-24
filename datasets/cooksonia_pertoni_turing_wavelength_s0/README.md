# Cooksonia pertoni — Turing wavelength at 1.3 µm (425 Ma)

Detail-scale slot 3153 (`3_subcellular_um`). The phenomenon is a **1.3 µm**
(1.339 × 10⁻⁶ m, log10 = −5.873) Turing wavelength on the tracheid wall of
*Cooksonia pertoni* in the **Late Silurian (Wenlock–Ludlow, 425 Ma)**.

This is not the 73 µm cellular-stripe slot. The sibling script
`models/1992-edwards-cooksonia-pertoni-turing-wavelength.py` uses a different
length. Here the diffusion ratio sits just above the Turing threshold
(Dᵥ/Dᵤ = 10; critical ratio ≈ 8.84) so the selected wavelength is sharp, and
the length unit is fixed so that wavelength is 1.339 µm.

## What changes at this age

*Cooksonia* is chronospecies 1/1 and the first member of the Paleobotanical
Vascular & Leaf Fossil Series.

- **Predecessor (pre-vascular axis, before 425 Ma).** No lignified wall separates
  activator and inhibitor diffusivities. Equal diffusion gives σ_max < 0, so no
  1.3 µm band is selected.
- **This horizon (425 Ma).** Edwards, Davies & Axe (1992) showed a vascular
  strand with annular to spiral tracheid thickenings. The model selects
  λ_c = 1.339 µm for that wall.
- **Successor (*Rhynia gwynne-vaughanii*, 407 Ma).** The same kinetics with a
  longer activator diffusion length move λ_c to about 1.70 µm. That shift is a
  reconstruction, not a length quoted by Kenrick & Crane (1997).

The 1.3 µm figure is the assigned observation scale. Edwards (1992) did not
publish a pigment-stripe or ocellus wavelength, and the museum compression does
not resolve subcellular wall bands.

## Specimen

`cooksonia_pertoni_amgueddfa_specimen.jpg` is a Silurian *Cooksonia pertoni*
compression photographed at Amgueddfa Cymru – National Museum Wales
(Wikimedia Commons `File:Amgueddfa_Cymru_-_Cooksonia_pertoni_-_211800.jpeg`,
CC BY-SA 4.0). A second slab, `cooksonia_pertoni_nmnh_specimen.jpg`
(`File:Cooksonia_NMNH.jpg`, CC BY-SA 4.0), is used only as a template-angle check.

## Run

```bash
python3 models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py --self-test
```

Outputs:

- `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png`
- `docs/media/cooksonia_pertoni_turing_wavelength_s0_growth.gif`

## Sources

See `paper_citations.bib`. Primary fossil paper: Edwards, Davies & Axe (1992),
doi:10.1038/357683a0. Kinetics: Schnakenberg (1979). Dispersion: Turing (1952).
