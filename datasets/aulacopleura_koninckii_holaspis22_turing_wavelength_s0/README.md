# Aulacopleura koninckii holaspis22 — Turing wavelength, 15 µm (slot 3128)

Silurian (Wenlock) trilobite *Aulacopleura koninckii*, holaspid morphotype II with **22 thoracic segments**, at geological age **428.1 Ma**.

This slot is the detail variant (`s0`) of scale tier `4_cellular_10um`. The phenomenon size is the local Turing wavelength

**λ_c = 15 µm = 1.514×10⁻⁵ m** (log10 = −4.820),

the critical wavelength selected by reaction–diffusion wavenumber k_c on the dorsal cuticle and ocular field. It is not the thoracic-segment spacing (millimetres).

## What changes relative to the predecessor

The predecessor in the Trilobite Segmentation & Eye Fossil Series is *A. koninckii* holaspis18 (18 thoracic segments, catalog age 428.3 Ma). Fusco et al. (2004) and Hughes et al. (2017) treat the 18- and 22-segment animals as contemporaneous intraspecific morphotypes (t18 and t22), not as successive chronospecies. The 0.2 Myr split is catalog bookkeeping for this series.

Hughes et al. (2017) reconstruct:

- thoracic segment release ends at stage s18 for t18 and at stage s22 for t22
- meraspid per-stage linear growth (Dyar) **1.115**
- holaspid per-stage linear growth **1.103**

At a size-invariant cellular wavelength of 15 µm, four extra meraspid molts enlarge linear size by 1.115⁴ ≈ 1.546, which is larger than the segment-count ratio 22/18 ≈ 1.222. Each t22 tergite at holaspid entry is therefore tiled by about 26% more 15 µm wavelengths than a t18 tergite at its earlier entry. The selector wavelength itself stays the local cell-scale quantity fixed for this slot.

## Pinned specimen

`aulacopleura_koninckii_holaspis22_specimen.jpg` — SHA-256 `1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a`, copied from `datasets/species_gallery/II_Anaima_Invertebrata/Aulacopleura_koninckii_holaspis22.jpg`.

## Run

```bash
python3 models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-aulacopleura-koninckii-holaspis22-turing-wavelength-s0.py
```
