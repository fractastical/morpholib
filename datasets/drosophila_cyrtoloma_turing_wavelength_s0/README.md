# *Drosophila cyrtoloma* — Turing carrier wavelength at 4.3 µm

Slot 3180 of the 10,000-Animal Basic Look (Project Delphi / MorphoEval).

- **Animal:** *Drosophila cyrtoloma* Hardy, Maui giant picture-winged fly (catalog body length 5.8 mm; common-name wingspan 9.5 mm; supernumerary crossvein network).
- **Scale:** **4.3 µm** exactly **4.315×10⁻⁶ m** (log10 = **−5.365**, tier 3/8 `3_subcellular_um`, detail variant). This is the subcellular Turing pigment-stripe / ocellus **carrier** wavelength, not the ~208 µm picture-spot spacing of sibling slot 3061.
- **Age:** **Holocene (extant), 0 Ma.** Lineage label: Hawaiian Picture-Winged *Drosophila* Adaptive Radiation. Delphi chronoseries position **9/41**.
- **Predecessor (chronoseries, not a cladistic ancestor):** *Drosophila buzzatii* (cactophilic repleta-group fly; Loeschcke, Bundgaard & Barker 1999 wing and thorax size reaction norms).
- **Successor (chronoseries, not a cladistic descendant):** *Drosophila deflecta* (quinaria-group fly; longitudinal vein and crossvein melanin shading).
- **Law:** Turing wavenumber \(k_c\) from the two-species dispersion relation (Turing 1952; Murray 1981) on Schnakenberg (1979) kinetics.

## What changes at this scale

Sibling slot 3061 places a primordium-scale pigment spot near **208 µm**. Dividing that envelope by this carrier gives \(207.6\,\mu\mathrm{m} / 4.315\,\mu\mathrm{m} \approx 48\), so each picture-spot holds on the order of forty-eight cycles of the 4.3 µm melanin carrier. Across a 9.5 mm wing the same carrier repeats about 2,200 times.

At 0 Ma all three chronoseries neighbours are extant. The quantity that changes is the selected wavelength of the cuticular melanin carrier:

| chronoseries | species | role | hypothesized carrier |
|---|---|---|---|
| 8/41 | *D. buzzatii* | predecessor | 7.84 µm, coarser, size-plastic wing cuticle |
| 9/41 | *D. cyrtoloma* | this slot | **4.315 µm**, shortest, dense picture-wing lace |
| 10/41 | *D. deflecta* | successor | 6.05 µm, anisotropic, vein-aligned shading |

Church & Extavour (2022) document phylotranscriptomic discordance in Hawaiian *Drosophila*, so this order is a morphospace chronoseries, not a resolved ancestor–descendant chain. The 4.315 µm, 7.84 µm and 6.05 µm values are reconstructions for this slot. They are not measurements reported by Edwards et al. (2007) or by the kinetic papers.

## Specimen plate

Wikimedia Commons returned no bitmap for *Drosophila cyrtoloma* or *Idiomyia cyrtoloma* on 2026-10-27. The file `drosophila_cyrtoloma_specimen.jpg` is a byte copy of `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_cyrtoloma.jpg` (SHA-256 pinned in the reference JSON). It is an 800×600 tan spiral optical-density plate with a 100 µm scale bar and a species caption. The simulation uses that plate as the long-wavelength prepattern and modulates it by the growing 4.3 µm Schnakenberg carrier. It is not a pinned-adult photograph and not a Hardy type image.

## Run

```bash
python3 models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py --self-test
python3 models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py
```

Outputs:

- `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_comparison.png`
- `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_growth.gif`
- `datasets/drosophila_cyrtoloma_turing_wavelength_s0/simulation_metrics.json`

## Sources verified on Crossref or OpenAlex

Turing (1952) 10.1098/rstb.1952.0012; Schnakenberg (1979) 10.1016/0022-5193(79)90042-0; Murray (1981) 10.1016/0022-5193(81)90334-9; Kondo & Miura (2010) 10.1126/science.1179047; Green & Sharpe (2015) 10.1242/dev.114991; Gompel et al. (2005) 10.1038/nature03235; Werner, Koshikawa, Williams & Carroll (2010) 10.1038/nature08896; Edwards, Doescher, Kaneshiro & Yamamoto (2007) 10.1371/journal.pone.0000487; Koshikawa (2020) 10.1111/dgd.12661; Loeschcke, Bundgaard & Barker (1999) 10.1046/j.1420-9101.1999.00060.x; Montgomery (1975) 10.5281/zenodo.10759076; Church & Extavour (2022) 10.1093/molbev/msac012.

Slot 3061 cites 10.11646/zootaxa.2400.1.1 as a Magnacca paper on this fly. Crossref returns that DOI as Zhu, Liu & Lis (2010) on the bug genus *Macroscytus*. This slot does not reuse that citation.
