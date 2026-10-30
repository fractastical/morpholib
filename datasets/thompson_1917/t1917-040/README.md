# T1917-040 — Dürer/Camper human profile coordinate deformations

Thompson (1917) claim **T1917-040**: human facial profiles in the Dürer/Camper tradition are mutually transformable by whole-grid coordinate deformation (Ch. XVII, Figs. 353–358; profile grids Figs. 364, 366).

## Primary sources (pinned)

| Asset | Role |
|-------|------|
| `thompson1917_leaf748_w1200.jpg` | Figs. 352–354 (stretch / bone panels) |
| `thompson1917_leaf750_w1200.jpg` | Figs. 356–358 (shear / flexure) |
| `thompson1917_leaf759_p740_fig364_durer_profiles.jpg` | Fig. 364 — three Dürer profiles (digitised landmarks) |
| `thompson1917_leaf761_p742_fig366_oblique_profiles.jpg` | Fig. 366 — oblique coordinates + Camper footnote |
| `homo_sapiens_lateral_profile.jpg` | CC BY-SA 3.0 Commons skull for CV segmentation |

## Named Dryad dataset (audit)

- **DOI:** [10.5061/dryad.0zpc86746](https://doi.org/10.5061/dryad.0zpc86746) — cichlid fish facial development (TSA experiment).
- **Verdict:** `DATASET_MISMATCH` for this claim (fish, not human profile grids). Metadata in `dryad_0zpc86746_metadata.json`; binary download blocked (HTTP 403) in the cloud VM.

## Validation script

```bash
python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py --self-test
python3 thompson/claims/t1917-040_on-the-theory-of-transformations.py
```

Outputs: `docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png`, `docs/media/t1917_040_on_the_theory_of_transformatio_growth.gif`, `result.json`.
