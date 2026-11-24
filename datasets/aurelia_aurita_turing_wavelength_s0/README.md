# Aurelia aurita — Turing ocellus wavelength at 201 µm

Slot 3129 of the 10,000-Animal Basic Look. One animal, one scale, one time.

- **Animal.** *Aurelia aurita* (moon jellyfish), Holocene, extant.
- **Age.** 0 Ma. Aurelia chronoseries in this slot is 1/1: there is no predecessor chronospecies and no successor chronospecies. The comparison that belongs at this node is the equal-diffusion control (pattern off) against the living-ectoderm ratio \(D_v/D_u = 20\) (pattern on).
- **Scale.** 201 µm tier label, exactly \(0.0002006\,\mathrm{m}\) (200.6 µm, \(\log_{10} = -3.698\)), tier `5_primordium_100um`, detail variant. This is the pigment-cup / ocellus wavelength, not the bell diameter and not the spacing between the eight rhopalia.

## What the sources actually say

Nakanishi, Hartenstein & Jacobs (2009, doi:10.1007/s00427-009-0291-y) describe the rhopalial nervous system of *Aurelia* sp.1, including the pigment-cup ocellus. Helm (2018, doi:10.1111/brv.12393) reviews the scyphozoan medusa, which carries eight rhopalia. Neither paper, nor the Turing (1952) or Schnakenberg (1979) papers, reports a measured pigment wavelength of 201 µm. That length is the scale assigned to this loop. The script sets the Schnakenberg critical wavelength equal to it and labels the calibration as a reconstruction.

The specimen photograph is Martin Thoma’s CC0 image of living moon jellies in Karlsruhe (Wikimedia Commons `File:Aurelia-aurita-3.jpg`, 24 Jan 2015). It is a whole-medusa photograph. It has no micrometre scale bar, so it cannot confirm 201 µm. It is the organism the law is about.

## Model

Schnakenberg (1979) activator–substrate kinetics

\[
\partial_t u = D_u\nabla^2 u + \gamma(a - u + u^2 v), \qquad
\partial_t v = D_v\nabla^2 v + \gamma(b - u^2 v)
\]

with \(a = 0.10\), \(b = 0.90\), \(D_u = 1\), \(D_v = 20\), \(\gamma = 1\), periodic boundaries on a square patch a few wavelengths across. Steady state \((u_*, v_*) = (1, 0.9)\). The Jacobian trace is \(-0.2\) and the determinant is \(1\), so the uniform state is stable without diffusion. The Turing discriminant holds at \(D_v/D_u = 20\) and fails at \(D_u = D_v = 1\).

\[
k_c = \bigl(\det J / (D_u D_v)\bigr)^{1/4}, \qquad \lambda_c = 2\pi / k_c, \qquad \lambda_c\,\ell = 0.0002006\,\mathrm{m}.
\]

## Run

```bash
python3 models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-aurelia-aurita-turing-wavelength-s0.py
```

Outputs:

- `docs/media/aurelia_aurita_turing_wavelength_s0_comparison.png`
- `docs/media/aurelia_aurita_turing_wavelength_s0_growth.gif`
- `datasets/aurelia_aurita_turing_wavelength_s0/aurelia_aurita_turing_wavelength_s0_metrics.json`

Lean proofs (no `sorry`) live in `morphoeval/lean/loops/Loop_3129_aurelia_aurita_turing_wavelength_s0.lean`.
