# Portfolio sub-prediction: FishBase BodyShapeI categorical body-shape class

Scored 2026-10-23 by `thompson/portfolio/score_fishbase_subjective.py` against the locked predictions (`dac08493`). Registered in `docs/THOMPSON_DATASET_PORTFOLIO.md` (Class F: categorical trait derived from shape) with **registered P = 80%**.

![figure](./fig_fishbase_subjective.png)

## Registered claim vs. realised outcome

| Claim | Registered P | Training Cutpoints (`BDmax/SL`) | `< 50 Myr` Accuracy (`full30` / `T7` vs. Majority `'elongated'`) | `>= 50 Myr` Accuracy (`full30` vs. Majority) | All 8,700 Species (`full30` vs. Majority) | Verdict |
|---|---|---|---|---|---|---|
| Ordinal `BodyShapeI` predicted from locked `BDmax/SL` beats majority-class baseline within 50 Myr | **80%** | `0.066, 0.234, 0.480` (train acc `81.9%`, $n=768$) | **65.6%** / **63.7%** vs. `18.6%` (**+47.1 pp**, $p = 0$) | **35.1%** vs. `49.4%` (**+-14.3 pp**) | **41.3%** vs. `43.2%` (**+-1.9 pp**) | **✓** |

## Accuracy by phylogenetic horizon (`tau` to nearest training species)

| Horizon bin (`tau`, Myr) | `n` | `full 30-D` 4-class accuracy | `Thompson-7` accuracy | `Copy nearest` accuracy | Majority `'elongated'` baseline | Gain over majority (pp) | Ordinal MAE (`full30` vs. baseline) |
|---|---|---|---|---|---|---|---|
| 0–5 | 134 | **73.9%** | 76.1% | 76.1% | 3.0% | **+70.9 pp** | **0.261** vs. 1.358 |
| 5–10 | 133 | **78.2%** | 73.7% | 75.9% | 4.5% | **+73.7 pp** | **0.218** vs. 1.346 |
| 10–25 | 598 | **68.9%** | 67.2% | 67.1% | 9.9% | **+59.0 pp** | **0.311** vs. 1.206 |
| 25–50 | 899 | **60.4%** | 58.1% | 63.0% | 28.8% | **+31.6 pp** | **0.397** vs. 0.923 |
| 50–80 | 1,690 | **37.2%** | 37.5% | 38.9% | 39.5% | **+-2.2 pp** | **0.663** vs. 0.783 |
| 80–120 | 1,788 | **41.0%** | 40.9% | 29.9% | 43.5% | **+-2.5 pp** | **0.610** vs. 0.701 |
| 120–∞ | 3,488 | **31.1%** | 31.1% | 32.2% | 57.3% | **+-26.2 pp** | **0.753** vs. 0.478 |
