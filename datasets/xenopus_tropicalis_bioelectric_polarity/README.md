# Dataset: `xenopus_tropicalis_bioelectric_polarity` (Slot #0921 — *Xenopus tropicalis*)

- **Organism (*Ousia*)**: *Xenopus tropicalis* (Western Clawed Frog)
- **Aristotelian Division**: I. Enaima (Vertebrata) → Amphibia & Reptilia (Ootoka)
- **Biophysical Cause (*Poiein*)**: 9. Bioelectric V_mem & Attractors (IX. Bioelectric Regeneration & Neural Maps)
- **Canonical Gallery Copy**: `datasets/species_gallery/I_Enaima_Vertebrata/Xenopus_tropicalis.jpg`

## Data-quality quarantine (fix-001 / BIO-TPL-DQ-01)

`xenopus_tropicalis_bioelectric_polarity_reference.json` → `empirical_series` is a **synthetic slot template** (`morphoeval/run_overnight_1000_loops.py`), not a digitised Keller measurement series. Do not cite `y_empirical` or its Pearson fit to `y_simulated` as biological evidence. See `empirical_series_provenance` in the reference JSON and `morphoeval/audits/2026-10-27-bioelectric-polarity-alpha-template-data-quality.yaml`.
