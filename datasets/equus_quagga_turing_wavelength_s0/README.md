# Equus quagga — Turing wavelength, detail scale s0 (slot 3221)

Holocene plains zebra (*Equus quagga*), age **0 Ma**. The pattern examined here is the Turing pigment-stripe wavelength at the organ-millimetre detail tier: **1.9 mm** (exact slot length **0.001883 m**, log10 = **-2.725**, tier `6_organ_mm`).

Bard (1977) found that plains-zebra body stripes (then *E. burchelli*) are equally spaced at about **0.4 mm** and perpendicular to the dorsal line in the third embryonic week. This slot is that mode after reconstructed post-initiation growth by 0.001883 / 0.0004 = **4.707**, not a new caliper measurement and not the adult centimetre-scale spacing visible in the photograph.

The predecessor in the assigned Equus chronoseries is **Equus scotti** (Pleistocene). No periodic melanin pattern is preserved for that horse. It is represented here by the equal-diffusion control: when inhibitor and activator diffuse at the same rate the Turing numerator changes sign and the 1.883 mm mode is not selected.

## Run

```bash
python3 models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-equus-quagga-turing-wavelength-s0.py
```

Outputs:

- `docs/media/equus_quagga_turing_wavelength_s0_comparison.png`
- `docs/media/equus_quagga_turing_wavelength_s0_growth.gif`
- `equus_quagga_turing_wavelength_s0_metrics.json`

## Specimen

Flank crop of Yathin S Krishnappa, *Plains zebra walking at Klein Namutoni*, Etosha, 2012. Wikimedia Commons `File:Equus quagga (Namutoni, 2012).jpg`, **CC BY-SA 3.0**. Crop box (left, top, right, bottom) = (120, 530, 480, 680) on the 1280×853 frame vendored as `equus_quagga_namutoni_source.jpg`. The crop is `equus_quagga_specimen.jpg`. It shows adult stripes. It is a pattern-class specimen (orientation, contrast, periodicity), not a ruler for the 1.883 mm fetal wavelength.
