# *Anolis carolinensis* — Turing wavelength at 101 µm (slot 3112)

Holocene extant green anole, detail scale **101.1 µm** (0.0001011 m, log10 = −3.995), tier `5_primordium_100um`. Geological age **0 Ma**. The Anolis chronoseries for this slot is a single point: Holocene (Extant), 1/1. There is no fossil chronospecies in the series.

This slot is the intra-scale dermal wavelength, not the sibling whole-animal Turing stub `datasets/anolis_carolinensis_turing_wavelength/` (slot 0124). Parameters here are a Schnakenberg pair \((a, b) = (0.055, 0.945)\) with inhibitor/activator diffusivity ratio \(D_v/D_u = 8.2\), chosen so the onset wavenumber satisfies \(\lambda_c = 2\pi/k_c = 101.1\,\mu\mathrm{m}\).

## What changes at this scale

- **Predecessor (tier 4, cellular, domain \(\lt 100\,\mu\mathrm{m}\)).** A 40 µm patch holds zero critical modes (`L / λ_c = 0`). Colour change on that scale is physiological migration of melanosomes inside an existing melanophore (Taylor & Hadley 1970), not selection of a new wavelength.
- **This scale (101.1 µm).** The homogeneous dermal state is stable without diffusion and unstable with diffusion. The selected mode is a spotted melanophore-territory prepattern on an 808.8 µm periodic patch (eight wavelengths).
- **Successor (organ scale, millimetre scales and the dewlap).** Fofonjka & Milinkovitch (2021) show that thinned diffusion at scale borders quantises a continuous reaction-diffusion field into one colour per scale (the *Timon* cellular automaton of Manukyan et al. 2017). That border rule is outside this domain: 101.1 µm is shorter than one mid-body anole scale, so the patch is intra-scale and the boundary condition stays periodic (Pass 2 also reports a no-flux patch).

## Specimen

`anolis_carolinensis_specimen.jpg` — public-domain photograph of a male *Anolis carolinensis*, Danny S. (Wikimedia Commons user Bartiebert), 2011-01-06, file `Anolis_carolinensis-032.jpg`. Vendored at 1400×885. SHA-256 is recorded in `anolis_carolinensis_turing_wavelength_s0_reference.json`.

## Run

```bash
python3 models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-anolis-carolinensis-turing-wavelength-s0.py
```

Outputs:

- `docs/media/anolis_carolinensis_turing_wavelength_s0_comparison.png`
- `docs/media/anolis_carolinensis_turing_wavelength_s0_growth.gif`
- `datasets/anolis_carolinensis_turing_wavelength_s0/metrics.json`
