# Coturnix japonica — Turing wavelength at 22 µm (slot 3157)

Holocene (extant) Japanese quail, *Coturnix japonica*. Geological age **0 Ma**. Characteristic scale **22 µm** (exact slot length **2.242×10⁻⁵ m**, log10 = **−4.649**), tier `4_cellular_10um`, detail variant.

This slot isolates one wavenumber: the within-feather pigment / ocellus mode selected by a Turing reaction–diffusion system on a barb-ridge patch. It is not the somite-spaced dorsal stripe of the embryonic quail (Haupaix et al. 2018), which is a longer, body-scale prepattern and is already modelled elsewhere in this repository.

## What changes at this scale and age

The assigned chronoseries has a single node (1/1): Holocene extant, 0 Ma, with an empty lineage string. There is no predecessor chronospecies and no successor chronospecies, so the geological age gap is 0.

The contrast formalized here is ontogenetic and spectral, at fixed age 0 Ma:

- **Predecessor state (same species, before the mode exists).** A pre-melanocyte or equal-diffusion collar. The inhibitor does not outrange the activator, the Turing cross-term is negative, and no 22.42 µm wavelength is selected.
- **This state.** Schnakenberg kinetics with Dᵥ/Dᵤ ≈ 52.9 open an unstable band whose critical wavelength is 22.42 µm. That length is one cellular-detail wavelength: on the order of a barb-ridge / melanocyte-cluster spacing inside a feather germ, not a somite.
- **Not a successor chronospecies.** The macro yellow/black bands drawn by somitic *agouti* (Haupaix et al. 2018; Bailleul et al. 2019) sit at a much longer wavelength and are treated as the spatial envelope, not as a later fossil species.

## Model

Schnakenberg (1979) kinetics on the Turing (1952) instability, used as the within-feather activator–inhibitor phenomenology of Prum & Williamson (2002) and the Shh–Bmp2 sign structure of Harris et al. (2005):

```
∂u/∂t = a − u + u²v + ∇·(D_u ∇u)
∂v/∂t = b − u²v + D_v ∇²v
```

Nominal parameters are local to this slot (a = 0.07, b = 0.93, Dᵤ = 0.035, Dᵥ = 1.85). They are not the somite-model parameters of the gallery-scale quail script, and they are not a peacock-ocellus parameter set. The length unit is fixed so the linear critical wavelength equals 2.242×10⁻⁵ m.

## Specimen

`coturnix_japonica_yehliu_taiwan.jpg` — Japanese quail at Yehliu, Taiwan, photographed by 孫鋒 林, CC BY-SA 2.0, via Wikimedia Commons (`File:Coturnix_japonica,_Yehliu,_Taiwan_1.jpg`). The portrait has no micron scale bar. It is used for plumage texture (vermiculated eumelanin / phaeomelanin streaks), not as a photogrammetric measurement of 22.42 µm.

## Run

```bash
python3 models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-coturnix-japonica-turing-wavelength-s0.py
```

Outputs:

- `docs/media/coturnix_japonica_turing_wavelength_s0_comparison.png`
- `docs/media/coturnix_japonica_turing_wavelength_s0_growth.gif`
- `datasets/coturnix_japonica_turing_wavelength_s0/coturnix_japonica_turing_wavelength_s0_metrics.json`
