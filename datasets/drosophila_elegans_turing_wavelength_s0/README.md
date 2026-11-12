# Drosophila elegans — 1.8 µm Turing wavelength (slot 3182, detail s0)

Holocene (Extant), geological age **0 Ma**. Phenomenon scale **1.8 µm** (1.803×10⁻⁶ m, log10 = −5.744, tier `3_subcellular_um`).

*Drosophila elegans* is the morning-glory flower-breeding fly. Males carry an apical melanic wing patch used as a courtship semaphore (Yeh, Liou & True 2006, Heredity, doi:10.1038/sj.hdy.6800814). This slot is **not** the ~101 µm organ-scale spot (slot 2801). It is the subcellular Turing wavelength of the melanin-granule / ocellus texture inside that patch.

Lineage: Sophophora suzukii–biarmipes wing-spot clade. Drosophila chronoseries position 11/41. Predecessor chronospecies *Drosophila deflecta*. Successor chronospecies *Drosophila erecta*. The genetic papers compare *D. elegans* with *D. gunungcola*; deflecta and erecta are the chronoseries neighbors formalized here, not the sister species in Yeh et al.

## Governing system

Schnakenberg (1979) kinetics with Turing (1952) diffusion:

- ∂u/∂t = a − u + u²v + D_u ∇²u
- ∂v/∂t = b − u²v + D_v ∇²v

Homogeneous state (a, b) = (0.12, 0.88), unique to this slot: u₀ = 1, v₀ = 0.88. Trace f_u + g_v = −0.24 < 0 and det J = 1 > 0, so the epithelium is stable without diffusion. Periodic boundary conditions on a flat apical-wing epithelial patch.

Critical onset ratio r_c = D_v/D_u ≈ 9.372. *D. elegans* sits just above onset (r/r_c = 1.06). Activator diffusivity is solved so the fastest-growing linear wavenumber satisfies λ = 2π/k_max = 1.803 µm. At that fixed D_u the chronoseries changes only the inhibitor range:

| chronospecies | role | r/r_c | selected linear λ |
|---|---|---:|---:|
| *D. deflecta* | predecessor | 2.65 | longer (far above onset, coarse cuticle) |
| *D. elegans* | focal, 0 Ma | 1.06 | 1.803 µm apical ocellus lattice |
| *D. erecta* | successor | 1.48 | intermediate abdominal melanin |

## Specimen

`drosophila_elegans_specimen.jpg` is a byte copy of `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_elegans.jpg` (Wikimedia Commons returned HTTP 403). The photograph is a lateral adult on a leaf with a 1 mm scale bar. It does not resolve 1.8 µm. The growth animation uses it as the adult phenotypic canvas; the subcellular field is the magnified inset.

## Run

```bash
python3 models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-elegans-turing-wavelength-s0.py
```
