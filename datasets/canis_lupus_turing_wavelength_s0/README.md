# Canis lupus — Turing wavelength, detail scale (slot 3139)

Holocene extant *Canis lupus* (common catalog name: Canine, MDCK), age **0 Ma**.
Chronoseries length 1/1: there is no predecessor chronospecies in this series.

Phenomenon scale: **1.8 µm** exactly **1.781 × 10⁻⁶ m** (log10 = −5.749), tier `3_subcellular_um`, detail variant `s0`.

The quantitative law is the Turing critical wavenumber

```
k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v),    λ_c = 2π / k_c
```

on the Schnakenberg activator–inhibitor system (Schnakenberg 1979; Turing 1952). Parameters are chosen so λ_c = 1.781 µm on a periodic patch the size of one MDCK apical face (8 wavelengths, 14.248 µm). That is the change at this scale relative to a Murray (1981) coat-scale pre-pattern of order 2 cm: the wavenumber is about 10⁴ times larger, and the domain that holds a handful of modes is a cell, not a flank. k_c itself is fixed by (D, γ). A different cell diameter would change the integer mode count N = round(L / λ_c), not k_c. This series has no earlier Canis chronospecies against which to difference N.

The canonical specimen image is the project gallery micrograph for *Canis lupus* (Canine, MDCK), SHA-256 pinned in the reference JSON. It has no scale bar, so pixel spacings measured on it are not converted to 1.781 µm.

Run:

```bash
python3 models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py
```

Citations: `paper_citations.bib`. Scout notes: `scout_manifest.json`.
