# T1917-010 — Thompson (1917) Ch. III: regeneration as a lawful growth process

Loop `thompson1917-t1917-010` (slot 208) validates one dated claim from D'Arcy Thompson,
*On Growth and Form* (1917), Ch. III "The Rate of Growth", section *Regeneration, or growth
and repair* (pp. 138–149, Figs. 35–40):

> Regeneration restores form at a measurable rate related to the amount removed
> (Durbin/Ellis tail-regeneration curves), i.e. restoration of form is itself a lawful
> growth process.

## What is in this directory

| File | Content | Provenance |
| --- | --- | --- |
| `thompson_p139_durbin_expII_regeneration.csv` | Durbin Exp. II raw table: mm regenerated at 3, 7, 10, 14, 18, 24, 30 days (34.2 mm tadpoles, 11.5 mm removed) | Thompson p. 139 = Durbin (1909) Table III |
| `thompson_p141_durbin_expII_interpolated.csv` | Thompson's graphic interpolation of the above, days 3–30, with daily increments and their log10 | Thompson p. 141 (Figs. 35–37) |
| `thompson_p144_durbin_expIV_regeneration.csv` | Durbin Exp. IV (49.1 mm tadpoles, 14.8 mm removed): raw days 3–31 and Thompson's interpolation days 1–17 | Thompson p. 144 = Durbin (1909) Table V |
| `thompson_p148_ellis_tail_regeneration.csv` | Ellis series O/P/R/S: 12.4, 20.2, 40.5, 56.7 % of tail removed; % of the removed amount regenerated at 3–32 days | Thompson p. 148 = Ellis (1909) |
| `thompson_1917_p1NN_leafNNN.jpg` | Greyscale 1200 px copies of IA page scans (pp. 139, 140, 141, 144, 148) used to correct the OCR | `https://archive.org/download/ongrowthform1917thom/page/n{leaf-1}.jpg` |
| `ongrowthform1917thom_ocr_excerpt_pp138-149.txt` | Verbatim OCR excerpt (lines 7744–8250 of the IA djvu text) | IA `ongrowthform1917thom` |
| `lithobates_clamitans_tadpole_specimen.jpg` | CC-BY lateral photograph of a *Rana (Lithobates) clamitans* tadpole — Ellis's species | iNaturalist obs. 137092396, photo 234072694, (c) Evan Grimes, CC BY 4.0 |
| `dryad_n6n6495_fitzpatrick_et_al_data.csv` | Pre-scouted Dryad hit (lizard tail autotomy / telomeres) | Dryad doi:10.5061/dryad.n6n6495 (CC0) — **DATASET_MISMATCH**, analogue only |
| `t1917_010_the_rate_of_growth_reference.json` | Equations, state variables, boundary conditions, parameter regimes, claims tested, SHA-256 pins | this loop |
| `scout_manifest.json` | Step 1 scouting dossier (dataset verification, specimen search, uniqueness check) | this loop |
| `result.json` | Verdict written by `thompson/claims/t1917-010_the-rate-of-growth.py` | this loop |

All CSV tables were transcribed from the page scans; three OCR errors were corrected
(Ellis last column is day **32**, not 21; Durbin Exp. IV raw days are 3, 5, 7, 10, 12, 14,
17, 24, 28, 31; the p. 141 interpolated table had merged rows).

## Why the pre-scouted Dryad dataset is not the test dataset

`thompson/thompson_1917_dataset_audit.json` listed doi:10.5061/dryad.n6n6495 as the top
title-relevant hit. Fetching it shows a single 60-day endpoint of tail regrowth in ten
autotomised *Niveoscincus ocellatus* skinks with "Tail removed" recorded as YES/NO. It has no
time series and no graded amount-removed variable, so it cannot test rate-vs-amount-removed
or the shape of the regeneration curve. The claim override allows Thompson's own 1917
figures from the pinned IA scan instead; these are the very Durbin/Ellis curves the claim
names, so the verdict is computed on them. The Dryad CSV is kept, SHA-256-pinned and
labelled analogue, only as a provenance record.

## Equations formalised (see `reference.json` and the Lean file)

- **E1** `V(t) = dL/dt` — velocity is the derivative of the growth curve (Figs. 35–36).
- **E2** Thompson's power-decay gloss `V·t^n = const`, `n ≈ 2` (p. 143); integrated `L = L∞ − c/t`.
- **E3** Thompson's Fig. 37: `log V` linear in `t`, i.e. `V = V0 e^{−kt}` and
  `L(t) = L∞(1 − e^{−k(t−t0)})` (monomolecular approach).
- **E4** Exp. IV: velocity rises to a maximum "about the seventh day" then falls (sigmoid `L`).
- **E5** Ellis: absolute amount regenerated after a fixed time increases with the amount removed,
  `L_n(R) ≈ p_n · R`.
- **E6** Fig. 40 isochrones: `t_q(0) = 0`, `t_q → ∞` as `R → 75 %`, "cubic parabola" (`t_q ∝ R^{1/3}`).
- **E7** Restoration is incomplete (`p_∞ ≈ 0.44–0.48`).

## Reproduce

```bash
python3 thompson/claims/t1917-010_the-rate-of-growth.py --self-test
python3 thompson/claims/t1917-010_the-rate-of-growth.py          # writes docs/media/t1917_010_* and result.json
```
