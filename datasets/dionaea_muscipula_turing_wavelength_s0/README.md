# Dionaea muscipula — Turing wavelength at 24 µm (slot 3169)

Holocene (Extant), geological age **0 Ma**. Dionaea chronoseries **1/1** (no other age in this slot).

Characteristic scale: **24 µm** (2.4×10⁻⁵ m, log10 = −4.620), tier `4_cellular_10um`, detail variant. The quantity selected by the model is the Turing wavenumber k_c of an adaxial epidermal pigment / ocellus mode, with critical wavelength λ_c = 2π/k_c = 24 µm.

This is not the centimetre-scale snap of the trap lobe (Forterre et al. 2005). Snap buckling explains how a mature lobe inverts. The law implemented here is chemical Turing wavelength selection (Schnakenberg 1979 kinetics; Turing 1952 dispersion) on a 192 µm × 144 µm patch of adaxial epidermis — eight by six critical wavelengths.

## What changes at this scale and age

The chronoseries has a single point, so the predecessor and successor are the neighbouring *scales* of the same extant species, not other named chronospecies.

- **Predecessor, 8 µm (subcellular).** The lowest Neumann wavenumber that fits, π/8 µm⁻¹, lies above the upper edge of the Turing band. The same kinetics stay uniform. No pigment spots are selected.
- **This slot, 24 µm on a 192 µm patch.** The critical mode fits eight times along the patch. Spots / ocelli with wavelength 24 µm are the unstable pattern.
- **Successor, 480 µm (tissue).** k_c does not move. The patch simply holds twenty wavelengths instead of eight, so the lobe carries a field of glands rather than a single spot.

At **0 Ma** the plant is extant. No fossil Dionaea in this slot preserves a 24 µm epidermal pattern, and this loop does not claim a new caliper measurement that the biological spacing is exactly 24.00 µm. 24 µm is the assigned cellular-detail wavelength for the slot; the script makes the dispersion relation hit that wavelength exactly and then checks the numerical solver against the linear law.

## Specimen

`dionaea_muscipula_trap_lobe_specimen.jpg` is the Wikimedia Commons file “Venus Flytrap showing trigger hairs”, licence **CC BY-SA 2.5**.

- File page: https://commons.wikimedia.org/wiki/File:Venus_Flytrap_showing_trigger_hairs.jpg
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/3/37/Venus_Flytrap_showing_trigger_hairs.jpg

The project gallery copy `datasets/species_gallery/III_Phyta_Embryophyta/Dionaea_muscipula.jpg` is the same trap pose at a smaller size and is recorded only as a cross-check. It is not the pinned specimen.

## Reproduce

```bash
python3 models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py --emit 1
python3 models/2026-delphi10000-dionaea-muscipula-turing-wavelength-s0.py --emit 2
```

`--emit 1` writes the unfitted activator field. `--emit 2` adds the diffusivity sweep, the equal-diffusion ablation, specimen segmentation, the dispersion goodness-of-fit, and the specimen-scaffold animation. Outputs:

- `docs/media/dionaea_muscipula_turing_wavelength_s0_comparison.png`
- `docs/media/dionaea_muscipula_turing_wavelength_s0_growth.gif`
- `datasets/dionaea_muscipula_turing_wavelength_s0/metrics.json`

## Sources (Crossref-verified DOIs)

Turing 1952 (`10.1098/rstb.1952.0012`), Schnakenberg 1979 (`10.1016/0022-5193(79)90042-0`), Gierer & Meinhardt 1972 (`10.1007/BF00289234`), Murray 1981 (`10.1016/0022-5193(81)90334-9`), Kondo & Miura 2010 (`10.1126/science.1179047`), Ball 2015 (`10.1098/rstb.2014.0218`). Trap anatomy, not the wavelength law: Forterre et al. 2005 (`10.1038/nature03185`), Hedrich & Neher 2018 (`10.1016/j.tplants.2017.12.004`), Hodick & Sievers 1988 (`10.1007/BF00394867`), Sachse et al. 2020 (`10.1073/pnas.2002707117`).
