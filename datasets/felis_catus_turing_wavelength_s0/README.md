# *Felis catus* — Turing wavelength at 436 µm (slot 3226)

Holocene domestic cat (*Felis catus*), extant, **0 Ma**. Felis chronoseries **1/1**: there is no earlier chronospecies in this slot. The length under study is **436 µm** (4.356×10⁻⁴ m, log₁₀ = −3.361), tier 5/8 `5_primordium_100um`, detail variant: the Turing pigment-stripe / ocellus wavelength on fetal skin, not the centimetre-scale adult tabby as seen by eye.

## What changes at this scale

The selected wavenumber k\* is a property of the reaction-diffusion linearisation (Turing 1952). It does not retune between geological ages because this series has only the Holocene species.

| Domain | Length | What the same chemistry does |
| --- | --- | --- |
| Cellular predecessor (tier 4) | 80 µm | Fundamental mode lies above the unstable band. The field stays homogeneous. No stripe. |
| This scale | 436 µm | Fastest mode. One primordium wavelength. Fetal epidermal pre-pattern. |
| Successor flank | 4360 µm | Ten repeats of the same 436 µm wave. Stripe *number* changes; k\* does not. |

Adult coat photographs show that pre-pattern after the skin has expanded (Kaelin, McGowan & Barsh 2021). They are not a micrograph of the 436 µm wave.

## Model

Schnakenberg (1979) kinetics, the minimal activator-inhibitor system used to realise Turing's dispersion relation:

- ∂u/∂t = Dᵤ ∇²u + γ (a − u + u²v)
- ∂v/∂t = Dᵥ ∇²v + γ (b − u²v)
- k\* = argmax σ(k), σ the leading eigenvalue of J − k²D
- λ = 2π/k\* calibrated to **436 µm** at the nominal Holocene point (a, b, Dᵤ, Dᵥ, γ) = (0.073, 1.186, 0.031, 1.742, 6)

These rate constants are not published Dkk4/Wnt parameters. Murray (1981) is the coat-pattern reading of the same wavenumber law. Kaelin et al. (2012) identified Taqpep as the gene that sets up the periodic pre-pattern and Edn3 as the pigment implementation; Kaelin et al. (2021) showed a Dkk4 expression pre-pattern in fetal cat epidermis before stripes are visible.

Parameters are not those of the sibling dataset `datasets/felis_catus_turing_wavelength` (slot 103, Bard millimetre-scale template).

## Specimen

`felis_catus_specimen.jpg` — *Felis catus*, ten-month-old female tabby on snow. Photograph by Von.grzanka, 9 February 2010, CC BY-SA 3.0. Wikimedia Commons file `Felis catus-cat on snow.jpg`. Vendored here as the 1280-pixel rendition and SHA-256 pinned in the reference JSON.

## Run

```bash
python3 models/2026-delphi10000-felis-catus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-felis-catus-turing-wavelength-s0.py
```

Outputs: `docs/media/felis_catus_turing_wavelength_s0_comparison.png` and `docs/media/felis_catus_turing_wavelength_s0_growth.gif`.

## Measured on this run

Nominal point: k\* = 6.135, simulated dominant wavelength **431.3 µm** against the **436 µm** slot scale. Dispersion Pearson r = 0.990 (RMSE 0.370, KS 0.111). Inhibitor diffusivity lengthens λ monotonically (r = 0.984). Equal diffusion and removal of the u²v term both collapse the pattern. Isotropic kinetics make a spot lattice; Dy/Dx = 4 raises orientation anisotropy from 0.02 to 0.59 (stripes). The segmented cat is not a circle (circularity 0.146, aspect 1.37).
