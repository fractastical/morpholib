# Confuciusornis sanctus — 143.4 µm Turing wavelength (slot 3152)

Detail-scale reaction-diffusion wavelength for the Early Cretaceous pygostylian *Confuciusornis sanctus* (Yixian Formation, **125.0 Ma**).

The phenomenon scale is **143.4 µm** (`1.434 × 10⁻⁴ m`, `log10 = -3.843`, tier `5_primordium_100um`, detail variant). That length is one critical Turing wavelength `λ_c = 2π / k_c` of a barb-ridge / within-feather pigment primordium on the ribbon-like rectricial streamer, not the millimetre-scale vane and not the sub-micrometre melanosome.

## What changes at this scale and age

Chronoseries position: **Confuciusornis 1/1**. There is no second *Confuciusornis* chronospecies in this slot. The comparison is to the bounding nodes of the Theropod-Avian Feather Fossil Series, at this same wavelength law:

- **Predecessor (Anchiornis-grade feathered theropod, Late Jurassic, ~160 Ma).** Ribbon-like pygostylian rectrices are absent. A 143.4 µm pigment-band mode has no ribbon-collar domain to occupy. Melanosome-scale pigment (Zhang et al. 2010; Wogelius et al. 2011) sits about two orders of magnitude below 143.4 µm and is not this wavelength.
- **This node (Yixian Formation, 125.0 Ma).** Pygostyle and paired ribbon streamers are present (Hou et al. 1995; O'Connor et al. 2012). A circumferential collar transect of `L = 8 λ_c = 1147.2 µm` admits **8** wavelengths. Kinetics are Schnakenberg parameters fixed for this primordium (`a = 0.08`, `b = 1.12`, `D_v/D_u = 12`), not the organ-scale sibling's parameters.
- **Successor scale (same species-series, 5.4 mm organ-scale vane).** The wavelength stays 143.4 µm. Domain growth raises the mode count from 8 to `round(5400 / 143.4) = 38`. Diffusivities that preserve the same dimensionless wavenumber scale as length squared, by a factor `(5400 / 143.4)² ≈ 1418`. The change is the number of unstable modes that fit, not a new reaction scheme.

## Sources

Crossref-verified DOIs are in `paper_citations.bib`. The specimen photograph is the Senckenberg Naturmuseum slab of *Confuciusornis sanctus*, Wikimedia Commons, **CC BY-SA 4.0**, SHA-256 pinned in the reference JSON.

## Reproduce

```bash
python3 models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-confuciusornis-sanctus-turing-wavelength-s0.py
```
