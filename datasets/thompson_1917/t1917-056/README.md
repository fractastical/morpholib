# T1917-056 — Eohippus → Equus skull coordinate transformation (Thompson 1917)

**Claim:** Thompson (1917), Ch. XVII, pp. 765–767, Figs. 401–402 — the skull of *Eohippus* (*Hyracotherium*) transforms into *Equus* by coordinate deformation; Heilmann intermediates lie on the transformation.

## Vendored assets

| File | Role |
|------|------|
| `thompson_fig401_402_landmarks.json` | Digitised 18-landmark lateral configurations (endpoints + Heilmann stages) |
| `eohippus_specimen_gallery.jpg` | Specimen photo anchor (gallery plate) |
| `equus_specimen_gallery.jpg` | Extant equid photo anchor (zebra proxy) |
| `scout_manifest.json` | Dryad doi:10.5061/dryad.crjdfn3g3 verification + fetch probe |
| `t1917_056_on_the_theory_of_transformatio_reference.json` | Equations, SHA-256 pins, parameter regime |

## Named public dataset (audit)

**Dryad** [10.5061/dryad.crjdfn3g3](https://doi.org/10.5061/dryad.crjdfn3g3) — Wisniewski & Slater (2026) equid cranial GM (includes *Hyracotherium* and extant equids). Cloud loop Step 3 recorded HTTP 401/403 on bulk download; species/variables match the claim but the archive was not vendored in this run.

## Script

```bash
python3 thompson/claims/t1917-056_on-the-theory-of-transformations.py --self-test
python3 thompson/claims/t1917-056_on-the-theory-of-transformations.py
```

Outputs: `docs/media/t1917_056_on_the_theory_of_transformatio_comparison.png`, `..._growth.gif`, and `result.json`.
