# Bacillus subtilis — Turing wavelength at 16 µm (loop 3130)

Holocene extant *Bacillus subtilis* (0 Ma). Bacillus chronoseries node **1/1**: there is no predecessor and no successor chronospecies, so the age change at this node is **0 Ma**. The scale is the detail variant of tier `4_cellular_10um`:

| quantity | this slot | tier center |
| --- | --- | --- |
| wavelength λ | **16 µm** (1.58×10⁻⁵ m, log10 = −4.801) | 10 µm |
| wavenumber k = 2π/λ | 0.393 µm⁻¹ | 0.628 µm⁻¹ |
| k ratio | 0.625 | 1 |
| diffusion scale at fixed kinetics (λ ∝ √D) | ×2.56 | ×1 |

The law is Turing’s critical wavenumber. A Schnakenberg activator–inhibitor system is calibrated so the most unstable continuous mode sits at λ = 16 µm on a 192 µm periodic domain (exactly 12 wavelengths). On a circular front the same wavelength sets the branch count m = round(2π R / λ).

## What this is not

Kawasaki et al. (1997) and Ben-Jacob and colleagues measured **millimetre-scale** colony branches, not a 16 µm pigment stripe. Duran-Nebreda et al. (2021) produce periodic *B. subtilis* colonies by synthetic lateral inhibition, again at colony scale. Sharpe et al. (1998) and Dion et al. (2019) establish a characteristic rod length and diameter; 16 µm is a few cell lengths, which is why this slot treats it as a cellular-detail wavelength. The script does not claim a published measurement of a 16 µm ocellus spacing.

## Specimen

`bacillus_subtilis_ocellus_specimen.jpg` is a 960 px rendition of Korinna’s 2019 micrograph “Eye shaped Bacillus subtilis bacteria colony under the microscope” (Wikimedia Commons, CC BY 4.0, 13 Nov 2019). The file has no scale bar, so the reconstruction radius R = 183.3 µm is not read off the photograph. SHA-256 `94a0bad84960d16e2c8f5c09b3d6d0697f65e535d596ffa4260f0f53427ebed3`.

The species-gallery plate photograph `datasets/species_gallery/IV_Protista_Monera_Mycota/Bacillus_subtilis.jpg` is a different image (macro colonies) and is not the comparison target for this slot.

## Run

```bash
python3 models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py
```

Outputs: `docs/media/bacillus_subtilis_turing_wavelength_s0_comparison.png` and `docs/media/bacillus_subtilis_turing_wavelength_s0_growth.gif`.
