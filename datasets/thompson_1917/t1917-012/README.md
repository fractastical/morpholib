# T1917-012 — Bipolar cleavage field (Thompson 1917, Ch. IV, Figs. 58-59)

**Claim:** Early cleavage/segmentation patterns of eggs conform to the same bipolar (two-like-pole) field configuration.

## Primary specimens

| File | Role |
|------|------|
| `thompson_1917_leaf0209_p189_fig58_59.jp2` | IA scan page 189 (leaf 209) |
| `thompson_1917_fig58_cerebratulus_first_segmentation.png` | Fig. 58 — *Cerebratulus* first cleavage |
| `thompson_1917_fig59_two_similar_poles_field.png` | Fig. 59 — two similar poles field diagram |

## Analogue specimen

`frog_embryo_cleavage_stadia.jpg` — historical frog cleavage plate (Wikimedia Commons, public domain). Used only as a **cross-taxa analogue** for holoblastic furrow orientation; not *Cerebratulus*.

## Reproduction script

```bash
python3 thompson/claims/t1917-012_on-the-internal-form-and-structure-of-the-cell.py --self-test
python3 thompson/claims/t1917-012_on-the-internal-form-and-structure-of-the-cell.py
```

Outputs: `docs/media/t1917_012_*`, `datasets/thompson_1917/t1917-012/result.json`.

## Dryad re-probe (Step 1)

Pre-audit row **UNPROBED** (HTTP 429). This loop found **no** Dryad package matching cleavage-ray geometry; local Drosophila bicoid dataset is **DATASET_MISMATCH**. Verdict uses Thompson's own figures plus the labelled frog analogue.
