# Drosophila erecta — Turing wavelength, detail tier (slot 3183)

Holocene (extant), geological age **0 Ma**. Phenomenon scale **2.3 µm** (2.262 × 10⁻⁶ m, log10 = −5.646), tier `3_subcellular_um` (3/8), detail variant `s0`.

This is the subcellular critical wavelength of a Schnakenberg activator–inhibitor system, not the ~103 µm primordium-tier ocellus spacing used by slot 2751 and not the ~101 µm primordium anchors of the chronoseries neighbours.

## What changes here

| Chronospecies | Role | Age | Tier | Characteristic λ |
| --- | --- | --- | --- | --- |
| *Drosophila elegans* | predecessor | 0 Ma | primordium | 101.1 µm |
| *Drosophila erecta* | this slot | 0 Ma | subcellular detail | **2.262 µm** |
| *Drosophila eugracilis* | successor | 0 Ma | primordium | 100.6 µm |

Age does not change along this Holocene triplet. The wavenumber does: k_erecta / k_elegans ≈ 44.7, because this slot resolves an intra-epidermal melanin spacing instead of the ocellar annulus. *D. erecta* is further distinguished by female dark/light colour dimorphism maintained by balancing selection at the *tan* enhancer (Yassin et al. 2016, doi:10.1038/ncomms10400). The model holds diffusivity fixed and varies activator production `a` between the dark and light alleles, so λ stays near k_c while pigment amplitude moves. Those `a` values are not the power-law coefficients of the elegans, erecta-primordium, or eugracilis stubs.

## Law

Schnakenberg kinetics (doi:10.1016/0022-5193(79)90042-0):

- f(u, v) = a − u + u²v
- g(u, v) = b − u²v

Turing (1952, doi:10.1098/rstb.1952.0012) critical wavenumber:

- k_c² = (D_v f_u + D_u g_v) / (2 D_u D_v)
- λ_c = 2π / k_c = 2.262 µm at a = 0.17, b = 0.93, D_v/D_u = 16

The gallery image’s dark ring is a separate ~100 µm annulus (drawn scale bar 100 µm). A damped Bessel disk mode J0(kr) is fitted to that annulus. The 2.262 µm mode is the fine critical-ring texture grown on top of it.

## Specimen

`drosophila_erecta_specimen.jpg` is the project gallery fallback. On 2026-10-27, Wikimedia Commons and GBIF (usageKey 5074045) returned no CC bitmap of this species. SHA-256 is pinned in the reference JSON.

## Run

```bash
python3 models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-erecta-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_erecta_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_erecta_turing_wavelength_s0_growth.gif`

Formal companion: `morphoeval/lean/loops/Loop_3183_drosophila_erecta_turing_wavelength_s0.lean`.
