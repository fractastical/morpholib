# Aulacopleura koninckii holaspis18 — Turing wavelength at 15 µm

Slot 3127 of the 10,000-Animal Basic Look (Project Delphi / MorphoEval).

- **Animal:** *Aulacopleura koninckii* holaspid morphotype I (18 thoracic segments). Aristotelian place: II. Anaima (Invertebrata) → Entoma (Arthropoda) → *Aulacopleura*.
- **Scale:** 15 µm detail variant, stored as **1.549×10⁻⁵ m** (log10 = −4.810), tier `4_cellular_10um`. This is the Turing pigment-stripe / ocellus wavelength, not the millimetre body length.
- **Age:** Silurian, Wenlock Epoch, **428.3 Ma**. Lineage: Trilobite Segmentation & Eye Fossil Series. Chronoseries position 2/3.
- **Predecessor:** *A. koninckii* meraspis (12 thoracic segments, 4.2 mm, 428.5 Ma).
- **Successor:** *A. koninckii* holaspis22 (22 thoracic segments, 22 mm, 428.1 Ma).

At this cellular scale the critical Turing wavenumber \(k_c\) is fixed by the cuticle reaction–diffusion Jacobian and the two diffusivities. It does **not** retune between the three stages. What changes is the sagittal pitch of one thoracic segment, so the integer number of 15.49 µm repeats that fit on a pleura increases from the meraspid larva through holaspis18 to holaspis22.

The pinned photograph is Wikimedia Commons `File:Aulacopleura konincki.jpg` (CC BY-SA 3.0, Dwergenpaartje), a 14 mm specimen from the Silurian of the Czech Republic. Direct `upload.wikimedia.org` returned HTTP 403; the bytes were fetched via `Special:FilePath`. The three `datasets/species_gallery` JPEGs named meraspis / holaspis18 / holaspis22 are byte-identical, so they are not used as separate chronoseries specimens.

## Reproduce

```bash
python3 models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-aulacopleura-koninckii-holaspis18-turing-wavelength-s0.py
```

Outputs:

- `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_comparison.png`
- `docs/media/aulacopleura_koninckii_holaspis18_turing_wavelength_s0_growth.gif`
