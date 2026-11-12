# Danio nigrofasciatus — cellular Turing wavelength (slot 3163, detail s0)

Holocene (extant) spotted danio at the **27 µm** cellular scale (**2.71×10⁻⁵ m**, log10 = **−4.567**, tier `4_cellular_10um`). Geological age **0 Ma**. Lineage: Danionin Pigment Evolution Series. Danio chronoseries **4/5**, between predecessor *Danio margaritatus* and successor *Danio rerio*.

This slot is the melanophore-contact wavelength, not the ~1 mm organ-scale stripe spacing modelled for the same species in slot 1203.

## Law

Schnakenberg activator–inhibitor kinetics (Schnakenberg 1979) on the Turing dispersion relation (Turing 1952). The critical wavenumber is

```
k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
λ_c  = 2π / k_c = 2.71×10⁻⁵ m
```

An area-preserving anisotropy α = D_AP / D_DV selects stripes when α is large and spots when α → 1. *D. nigrofasciatus* keeps α high on the dorsal flank and drops it ventral to the midlateral line, which is the broken-ventral-stripe phenotype. *D. margaritatus* is nearly isotropic (ocelli). *D. rerio* keeps α high across the flank (continuous stripes). The cellular wavelength itself stays near 27 µm in all three extant species; what changes at this scale is the symmetry of the selected mode.

## Specimen

`danio_nigrofasciatus_specimen.jpg` is the CC BY 4.0 photograph by David M. Parichy (Wikimedia Commons `File:Danio nigrofasciatus_2.jpg`, credit eLife 2015, doi:10.7554/eLife.05635). SHA-256 is pinned in the reference JSON.

## Run

```bash
python3 models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-danio-nigrofasciatus-turing-wavelength-s0.py
```

Outputs:

- `docs/media/danio_nigrofasciatus_turing_wavelength_s0_comparison.png`
- `docs/media/danio_nigrofasciatus_turing_wavelength_s0_growth.gif`

Primary literature is in `paper_citations.bib`. The scout dossier is `scout_manifest.json`.
