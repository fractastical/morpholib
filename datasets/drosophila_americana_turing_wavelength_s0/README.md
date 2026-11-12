# Drosophila americana — detail-scale Turing wavelength (slot 3175)

Holocene (extant), **0 Ma**. Lineage: Subgenus *Drosophila* virilis–repleta–guttifera radiation.
Chronoseries 4/41. Predecessor: *Drosophila affinis*. Successor: *Drosophila ananassae*.

**Scale:** 2.1 µm (2.117×10⁻⁶ m, log10 = −5.674), tier `3_subcellular_um`, detail variant.
The quantity is the Turing pigment-stripe / ocellus wavelength selected by the critical wavenumber \(k_c\).

This is not the existing 139 µm primordium-scale *D. americana* stripe model (`wittkopp-drosophila-americana-turing-wavelength`). At 2.1 µm the wavelength shift between chronospecies is buffered (body-length exponent β = 0.35) instead of tracking the tergite-stripe ratios of those sibling slots.

## Law

Schnakenberg activator–inhibitor kinetics in the Gierer–Meinhardt class,

\[
f = a - u + u^{2}v, \qquad g = b - u^{2}v,
\]

with the Turing onset wavenumber (Turing 1952; Murray 1981)

\[
k_c^{2} = \frac{D_v f_u + D_u g_v}{2 D_u D_v}, \qquad \lambda_c = \frac{2\pi}{k_c}.
\]

The expressed wavelength is the linear-theory maximum-growth mode \(k_\sigma\). Diffusion ratio \(r = D_v/D_u = 10\) sits just above onset so \(k_\sigma\) stays near \(k_c\). Equal diffusion (\(r = 1\)) is the negative control.

## Specimen

`Drosophila_americana_specimen.jpg` is the repository species card (SHA-256 pinned in the reference JSON). Commons had no CC photograph of this species on 2026-10-27. The card is a brown pigment pattern with a 100 µm scale bar, not an ocellar micrograph.

## Run

```bash
python3 models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py
```

Verified DOIs are in `paper_citations.bib`. The Drosophila-catalog DOI `10.1016/j.cell.2009.09.044` is a Crossref 404; the pigmentation paper used here is Wittkopp et al. 2009, Science, `10.1126/science.1176980`.
