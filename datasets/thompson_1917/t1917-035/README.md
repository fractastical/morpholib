# T1917-035 — Sea-urchin test shape (Thompson 1917 Ch. XV, Fig. 328)

**Claim:** Certain hollow structures (e.g. sea-urchin tests in Fig. 328) take forms explicable as membranes/shells under pressure and tension.

## Dataset verification (Step 1)

| Source | Verdict |
|--------|---------|
| Dryad `doi:10.5061/dryad.dn160` | **MISMATCH** — cryptic-species genetics, not test profiles |
| Wikimedia Commons CC BY 2.0 photographs | **MATCH** — empty tests for meridian CV segmentation |

## Files

- `scout_manifest.json` — Step 1 dossier (OpenAlex + Commons + Dryad audit)
- `t1917_035_on_the_shapes_of_eggs_and_of_c_reference.json` — equations, parameters, SHA-256 pins
- `commons_sea_urchin_test_structure_8340336761.jpg` — primary specimen (profile view)
- `commons_sea_urchin_tests_7_77.jpg` — multi-test panel for dispersion statistics
- `echinoid_test_literature_ratios.csv` — published axial:equatorial ratios (sanity check)
- `result.json` — written by `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`

## Reproduction

```bash
python3 thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py --self-test
python3 thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py
```
