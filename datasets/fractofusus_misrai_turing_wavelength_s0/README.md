# Fractofusus misrai — Turing wavelength at 63 µm (slot 3227)

Detail-scale reaction–diffusion model for *Fractofusus misrai* Gehling & Narbonne, 2007, a fusiform rangeomorph from the Mistaken Point assemblage, Newfoundland (Late Ediacaran, **565 Ma**).

The assigned phenomenon is one **Turing pigment-stripe / ocellus wavelength of 63 µm (6.33×10⁻⁵ m, log10 = −4.199, tier `4_cellular_10um`)**. This is the cellular-tier detail variant of the same law that, at organ scale, spaces a primary rangeomorph module (3.275 mm; campaign slot 1251). It is not a histological measurement: Mistaken Point preserves external moulds, not 63 µm pigment histology. The length is the MorphoEval scale at which the critical wavenumber \(k_c\) is evaluated.

## What changes at this scale and age

Chronoseries **1/1**. There is no second *Fractofusus* chronospecies in this series. Gehling & Narbonne (2007) place the genus in the mid-Ediacaran, ca. 575–560 Ma; this slot is the 565 Ma Mistaken Point horizon.

- **Versus the scale predecessor (3.275 mm organ module).** One primary-branch module holds **51** detail-scale wavelengths of 63.3 µm and not 52 (`51 × 633 < 32750 < 52 × 633` in tenths of a micrometre). The organ-scale pattern is a single module; the 63 µm pattern is the branchlet / ocellus repeat inside it.
- **Versus a smaller (sub-wavelength) patch.** A domain shorter than \(\lambda_c\) cannot fit \(k_c\), and the homogeneous state stays stable. That is the subcritical-domain control.
- **Versus the Cambrian successor body plans.** Rangeomorphs are gone after the Ediacaran–Cambrian transition. The fractal, diffusion-limited module is replaced by bilaterian segments whose repeat is not this \(k_c\).

## Model

Schnakenberg kinetics (Schnakenberg 1979) with the Turing (1952) dispersion relation

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

Nominal detail-scale parameters (loaded by the script from the reference JSON): activator source \(a = 0.09\), inhibitor source \(b = 0.85\), \(D_u = 0.12\), \(D_v = 2.40\) (ratio 20) in units of the grid spacing fixed so that the analytical \(\lambda_c\) equals \(6.33\times10^{-5}\,\mathrm{m}\).

## Specimen

`fractofusus_misrai_specimen.jpg` — *Fractofusus misrai* on the Mistaken Point bedding plane.

- Source: Wikimedia Commons `File:Fractofusus_misrai.jpg`
- Photographer: djpmapleferryman (Flickr 3907146971, 2009-09-02)
- License: CC BY 2.0 (https://creativecommons.org/licenses/by/2.0)

## Run

```bash
python3 models/2026-delphi10000-fractofusus-misrai-turing-wavelength-s0.py --self-test
```

Writes `docs/media/fractofusus_misrai_turing_wavelength_s0_comparison.png` and `docs/media/fractofusus_misrai_turing_wavelength_s0_growth.gif`.

## Claim boundary

The script reconstructs a Schnakenberg field whose linear \(k_c\) sits at 63 µm. It does not claim that a pigment ocellus of that size was measured on a *Fractofusus* specimen, nor that rangeomorph branches are known to be reaction–diffusion stripes. Mitchell et al. (2015) establish multigenerational, stolon-like reproduction from spatial point patterns; Hoyal Cuthill & Conway Morris (2014) establish a fractal, self-similar rangeomorph body plan. Those empirical claims are the biological context. The wavenumber law is the mathematical object under test.
