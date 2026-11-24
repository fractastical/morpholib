# Columba livia — Turing wavelength at 256 µm (loop 3151)

Holocene extant rock dove (*Columba livia*), age **0 Ma**, chronoseries **1/1** (no predecessor or successor chronospecies). This slot is the detail variant of tier `5_primordium_100um`: the critical Turing wavelength

**λ_c = 256 µm = 0.000256 m** (tier label log10 = −3.592; exact log10(0.000256) = −3.59176).

The governing law is the Schnakenberg (1979) critical wavenumber k_c, read at the within-feather pigment scale of Prum & Williamson (2002, Proc. R. Soc. B 269:781–792, doi:10.1098/rspb.2001.1896). On a 64 × 16 µm grid the only linearly unstable Fourier mode is the one with λ = 256 µm, so the saturated stripe (wild-type bar element) and the isotropic spot lattice (checker / ocellar centres) are forced to that wavelength.

What changes relative to the neighbouring scales, not to another chronospecies:

- Predecessor scale: activator diffusion length ℓ_D ≈ 24.1 µm. Wavenumbers above the Turing band decay. This is a melanocyte-order length, not a fossil species.
- This scale: λ_c = 256 µm, four repeats on a 1024 µm vane patch.
- Successor scale: a packet of ten wavelengths (2560 µm) is still far below the centimetre wing bars. Those bars are an NDP expression envelope (Vickrey et al. 2018, eLife 7:e34803, doi:10.7554/eLife.34803). Longitudinal body stripes of galliforms are a somite prepattern (Haupaix et al. 2018, Science 361:6408, doi:10.1126/science.aar4777) and are not this wavenumber.

The photograph `columba_livia_specimen.jpg` is Alan D. Wilson’s CC BY-SA 2.5 rock dove at Burnaby Lake (Wikimedia Commons `File:Rock_dove_-_natures_pics.jpg`). It has no micrometre scale bar; 256 µm is below the sampling scale of that flight portrait. The portrait is the phenotype the wavelength is registered onto, not a direct measurement of λ_c.

Run from the repository root:

```bash
python3 models/2026-delphi10000-columba-livia-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-columba-livia-turing-wavelength-s0.py
```
