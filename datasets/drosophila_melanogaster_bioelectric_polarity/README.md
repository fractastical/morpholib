# Dataset: `drosophila_melanogaster_bioelectric_polarity` (Slot #0941 — *Drosophila melanogaster*)

- **Organism (*Ousia*)**: *Drosophila melanogaster* (Fruit Fly)
- **Aristotelian Division**: II. Anaima (Invertebrata) → Entoma (Arthropoda)
- **Biophysical Cause (*Poiein*)**: 9. Bioelectric V_mem & Attractors (IX. Bioelectric Regeneration & Neural Maps)
- **Canonical Gallery Copy**: `datasets/species_gallery/II_Anaima_Invertebrata/Drosophila_melanogaster.jpg`

## Data-quality quarantine (fix-001 / BIO-TPL-DQ-01)

`drosophila_melanogaster_bioelectric_polarity_reference.json` → `empirical_series` is a **synthetic slot template** (`morphoeval/run_overnight_1000_loops.py`), not a digitised Lecuit measurement series. Do not cite `y_empirical` or its Pearson fit to `y_simulated` as biological evidence. See `empirical_series_provenance` in the reference JSON and `morphoeval/audits/2026-10-27-bioelectric-polarity-alpha-template-data-quality.yaml`.
