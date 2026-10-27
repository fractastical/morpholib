# Drosophila paulistorum — Turing wavelength, detail rung s0

Slot 3197 of the 10,000-Animal Basic Look. One animal, one scale, one point in the catalog chronoseries.

- **Animal:** *Drosophila paulistorum* Dobzhansky & Pavan (Amazonian incipient semispecies ring complex), II. Anaima → Entoma → Drosophila. Lineage label: Sophophora obscura & willistoni clade.
- **Scale:** **2.4 µm** (2.417×10⁻⁶ m, log10 = **−5.617**, tier 3/8 `3_subcellular_um`, detail variant s0). This is the intra-ocellar pigment-stripe wavelength, not the 91 µm cellular stripe of slot 2941.
- **Age:** **0 Ma**, Holocene (extant). Drosophila chronoseries 26/41. Predecessor in that catalog: *Drosophila orena* (cellular stripe 89.44 µm). Successor: *Drosophila persimilis* (primordium stripe 116.7 µm). All three are extant; the series is a catalog order, not a stratigraphic ancestor chain.

## What changes at this rung

*D. orena* (predecessor) and *D. persimilis* (successor) place the Turing wavenumber on a cellular or primordium domain tens of micrometres across. This s0 reconstruction places \(k_c = 2\pi / 2.417\,\mu\mathrm{m}\) on a periodic ocellar-cortex patch. The 2.417 µm mode packs 37 times into the orena cellular wavelength (remainder 11 nm) and 48 times into the persimilis primordium wavelength (remainder 684 nm). The diffusivity ratio and the domain length are specific to this rung; they are not the power-law coefficients of those neighboring slots.

## Law

Schnakenberg kinetics (Schnakenberg 1979; Turing 1952 dispersion; Maini, Painter & Chau 1997):

\[
\partial_t u = D_u\nabla^2 u + \gamma(a - u + u^2 v), \qquad
\partial_t v = D_v\nabla^2 v + \gamma(b - u^2 v)
\]

\[
k_c^2 = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = 2\pi/k_c
\]

Nominal parameters (slot 3197 only): \(a=0.10\), \(b=0.90\), \(D_v/D_u=36\), \(\gamma=8\), periodic patch \(L=6\lambda=14.502\,\mu\mathrm{m}\). Both diffusivities are then multiplied by 0.478965 so the maximum of the growth rate \(\sigma(k)\), the mode the stripe field selects, lies at 2.417 µm. The classical determinant wavenumber is retained as the threshold condition proved in the Lean module.

## Specimen

`drosophila_paulistorum_male_haars_2018.png` — adult male, Jonathan Haars, 29 Nov 2018, CC BY-SA 4.0, Wikimedia Commons. SHA-256 `e01e60e633f99ef12a7474a79f5afaaa56b1502bc76590c4102723cf5cf7d1ae`. The species-gallery spiral JPEG was not used.

## Sources

Dobzhansky & Spassky 1959, doi:10.1073/pnas.45.3.419, circumscribe the semispecies. They do not report a 2.4 µm wavelength. Turing 1952 supplies \(k_c\). The 2.417 µm length is the assigned subcellular rung, on the order of a Drosophila rhabdomere diameter, reconstructed here as the critical wavelength of an ocellar pigment stripe.

Run:

```bash
python3 models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py
```
