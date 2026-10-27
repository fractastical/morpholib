# *Cavia porcellus* — Turing wavelength, tier-5 detail s0

Holocene (extant) guinea pig at **0 Ma**. The length modelled here is **126.7 µm** (0.0001267 m, log10 = −3.897), the rounded **127 µm** pigment stripe / ocellus wavelength on tier 5/8 `5_primordium_100um`, detail variant s0.

This is chronoseries node 1/1. There is no earlier or later *Cavia* chronospecies in the series. What changes relative to the neighbouring size tiers, at the same age, is the selected wavenumber:

- Predecessor tier, 1267 µm: exactly 10 of these s0 waves fit in one coarser wavelength. On the 1.0136 mm primordium window that mode is only an envelope.
- This node, 126.7 µm: eight critical wavelengths span the window. The carrier is the primordium-scale stripe / ocellus.
- Successor tier, 12.67 µm: ten finer waves sit inside one s0 wavelength and are averaged out.

The reaction–diffusion law is Schnakenberg kinetics (a = 0.08, b = 1.15, D_v/D_u = 40) with diffusivities scaled so

k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π/k_c = 126.7 µm.

The picture is the Swift–Hohenberg amplitude equation at that same k_c, started from a charge-2 spiral. Parameters are specific to this scale; they are not the flank-patch set used by other *Cavia* slots.

The vendored specimen is the project gallery pigment field (`Cavia_porcellus.jpg`) with a printed 100 µm bar, SHA-256 `42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd`. It is the scale-matched comparison target. Wikimedia CC BY-SA whole-animal portraits are listed in `scout_manifest.json` and are not used for the 127 µm fit.

Run from the repository root:

```bash
python3 models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py
```
