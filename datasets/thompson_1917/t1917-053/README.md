# T1917-053 — Dimorphodon ↔ Pteranodon skull coordinate transformation

**Claim (Thompson 1917, Ch. XVII, p. 756, Fig. 387):** the skulls of *Dimorphodon* and *Pteranodon* are related by coordinate transformation.

## Primary data

| File | Role |
|------|------|
| `ongrowthform1917thom_p756_leaf0775_fig387.jpg` | Internet Archive scan (public domain); quantitative surface for this loop |
| `pteranodon_longiceps_ypm1177_vendored.jpg` | Wikimedia Commons real specimen (CC BY-SA 3.0 derivative); CV segmentation only |
| `nie1905_dimorphodon_pteranodon_comparison.png` | Historical comparison plate (documentary) |
| `dryad_4nr4cf37_metadata.json` | Pre-scouted clade-frame dataset metadata (file not downloaded) |

## Script

```bash
python3 thompson/claims/t1917-053_on-the-theory-of-transformations.py --self-test
python3 thompson/claims/t1917-053_on-the-theory-of-transformations.py
```

Outputs: `docs/media/t1917_053_on_the_theory_of_transformatio_comparison.png`, `..._growth.gif`, and `result.json`.

## Dryad note

`doi:10.5061/dryad.4nr4cf37` (Foth et al. 2012) is the correct *pterosaur skull morphometrics* clade frame but the MorphoJ file could not be retrieved in this cloud environment (HTTP 401). The verdict is therefore based on Thompson's own Fig. 387, not on an independent landmark corpus.
