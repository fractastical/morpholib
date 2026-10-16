# C. elegans embryo: WormGUIDES atlas + OpenWorm connectome

Inputs for [`2015-santella-wormguides-celegans.py`](../../2015-santella-wormguides-celegans.py).
Nothing is vendored here: the script downloads every file from a pinned git
commit into the git-ignored `.celegans_cache/` and verifies it against
`pinned_inputs.sha256` (429 files, written 2026-10-16). A hash mismatch stops
the script.

## Sources

| layer | file(s) | source | licence |
|---|---|---|---|
| formal geometry (nuclei) | `src/atlas_model/data/nuclei_files/t001-nuclei` … `t360-nuclei` (360 files) | [zhirongbaolab/WormGUIDES](https://github.com/zhirongbaolab/WormGUIDES) @ `17be611f10cdc8ea438e67dc9ca3de859487a05e` | GPL-3.0 |
| cell shapes | `src/atlas_model/configurations/shapes_config/CellShapesConfig.csv` + 51 `src/atlas_model/data/obj_files/<cell>_t330.obj` | same | GPL-3.0 |
| eggshell outline | `src/atlas_model/data/obj_files/embryo_t{1,30,60,100,200,250,280,300,320,330,340,350,359}.obj` (the "Embryo Outline" entry of `CellShapesConfig.csv`; one mesh per frame, 0–359; 13 sampled) | same | GPL-3.0 |
| acquisition metadata | `src/atlas_model/parameters/production_info/ProductionInfo.csv` | same | GPL-3.0 |
| cell identities | `…/CElegansData/PartsList/partslist.txt` (787 rows), `…/CElegansData/CellDeaths/CellDeaths.csv` (113 names) | same | GPL-3.0 |
| formal connectivity | `c302/data/herm_full_edgelist.csv` (7379 rows; Cook et al. 2019) | [openworm/c302](https://github.com/openworm/c302) @ `f54d92d7e9d2410ff6c47d9ae866649f1a419ee8` | MIT |
| photograph / EM | — | [WormAtlas](https://www.wormatlas.org) | cited, not downloaded |

The nuclei files are StarryNite/AceTree tables, read the way WormGUIDES'
`AceTreeTableLineageDataLoader` reads them: column 1 = valid flag, 5–7 = x, y, z,
8 = diameter, 9 = lineage name. `ProductionInfo.csv` states for the
`all-nuclear positions` series: image series `DISPIM_061416`, strain `JIM113`,
60 s per frame, StarryNite segmentation, 360 time points, movie start time
20 min, `X_SCALE = Y_SCALE = Z_SCALE = 1`. Frame *t* is therefore reported
as movie-minute *t* + 19, and the coordinates are unit-less.

## What the script does with them

- **Decoding check.** All 1330 distinct names match the Sulston grammar
  `^(AB|MS|E|C|D|EMS|P[0-4]|Z[23])[aplrdv]*$`; 1326/1326 cells whose parent is in
  the movie first appear after that parent; the remaining four (ABa, ABp, EMS,
  P2) are the four cells present at frame 1, whose parents divided before the
  movie starts.
- **Calibration (one assumption, stated on the figure).** The atlas coordinates
  carry no unit, so the atlas's own eggshell outline mesh at frame 330
  (`embryo_t330.obj`, 303.1 × 177.3 × 175.5 units) is set to the 50 µm egg
  length of Gönczy et al. 1999, giving **0.1649 µm per atlas unit**. The only
  thing assumed is that this egg has the typical length; the length it is
  applied to is the atlas's segmentation of the shell, not the nuclei. Why
  frame 330: the atlas reuses one static shell mesh (285.8 units) for frames
  1–~250, and from frame 30 on the nuclei overrun it at the anterior pole by up
  to 19 units, so it is not the shell of this egg; from ~frame 280 the shell is
  re-segmented per frame (286.8–303.1 units) and encloses the nuclei (overrun
  1.8 units at frame 330). Checks that are not fitted: the frame-330 shell comes
  out 29.2 µm wide and 28.9 µm deep (the egg is ~50 × 30 µm), and the 13 sampled
  shell meshes bracket the scale at 0.1649–0.1750 µm per unit (~6 %). The
  earlier estimate from the longest nuclear cloud (0.1617 µm per unit,
  2026-10-15 checkpoint `2864784`) sits 2 % below the shell value. Under the
  new scale the nuclear cloud is 50.4 × 27.3 µm across at the end (it slightly
  overruns the shell length, within the segmentation slop) and the mean nuclear
  diameter falls from 4.8 µm (4 cells) to 2.2 µm (580 cells). The shell outline
  (convex hull of the nearest sampled mesh in x, y) is drawn on every embryo
  panel.
- **Lineage chart** re-derived from the names alone (depth-first order of the
  tree, parents centred on children, colour by founder).
- **Literature anchors** (printed as `check:` lines):

  | quantity | atlas | literature |
  |---|---|---|
  | cell count at movie-minute 330–350 | 562–571 | ~550 at the end of proliferation (WormAtlas Embryo intro) |
  | names in the atlas's death list | 113, all appear in the movie | 113 embryonic deaths (Sulston et al. 1983) |
  | nuclei at the last frame (movie-minute 379) | 580 (553 with a parts-list identity); maximum 585 at frame 345 | 671 generated, 558 at hatching (Sulston et al. 1983) |
  | 26-cell stage | movie-minute 70 | — |
  | E-cell ingression (gastrulation onset), detector: Ea/Ep at < 0.8 of the median normalised radius for 5 frames | frame 54, 26 cells, movie-minute 73 | ~30-cell stage; Ea/Ep enter at 100 min after first cleavage (WormAtlas) |

  The cell-stage anchors agree; the clock does not: the atlas embryo reaches the
  26-cell stage and gastrulation ~27 min earlier than the WormAtlas/Sulston
  schedule. The script does not resolve this (candidates: recording
  temperature, or a different time origin behind "movie start time 20 min");
  it is reported as a discrepancy.
- **Connectome.** Node names are canonicalised (`DA01` → `DA1`); neurons are the
  300 upper-case node names (muscles and end organs are lowercase-prefixed in
  Cook et al.'s naming; 448 nodes in all), 295 of which are in the parts list
  (AVFL, AVFR, DB1, DB3, PHBR are not). 5084 connected ordered neuron pairs
  from 3638 chemical and 2168 electrical edge rows. This is the connectivity
  layer only; it has no geometry.

## Literature values and where they come from

- Santella A, et al. (2015). WormGUIDES: an interactive single cell
  developmental atlas and tool for collaborative multidimensional data
  exploration. *BMC Bioinformatics* 16:189.
  https://doi.org/10.1186/s12859-015-0627-8 (CC BY 4.0)
- Sarma GP, et al. (2018). OpenWorm: overview and recent advances in
  integrative biological simulation of *Caenorhabditis elegans*. *Phil Trans
  R Soc B* 373:20170382. https://doi.org/10.1098/rstb.2017.0382
- Cook SJ, et al. (2019). Whole-animal connectomes of both *Caenorhabditis
  elegans* sexes. *Nature* 571:63–71. https://doi.org/10.1038/s41586-019-1352-7
  (edge list as redistributed by c302)
- Sulston JE, Schierenberg E, White JG, Thomson JN (1983). The embryonic cell
  lineage of the nematode *Caenorhabditis elegans*. *Dev Biol* 100:64–119.
  https://doi.org/10.1016/0012-1606(83)90201-4 — 671 / 113 / 558 and the
  gastrulation timing are quoted via WormAtlas (Embryo introduction,
  https://www.wormatlas.org/embryo/introduction/EIntroframeset.html, and
  Introduction §3.1), accessed 2026-10-15.
- Gönczy P, Schnabel H, Kaletta T, Amores AD, Hyman T, Schnabel R (1999).
  Dissection of cell division processes in the one cell stage *Caenorhabditis
  elegans* embryo by mutational analysis. *J Cell Biol* 144:927–946.
  https://doi.org/10.1083/jcb.144.5.927 — "50-µm-long cell", used for the
  calibration (the ~30 µm width used as a consistency check is also the
  standard figure quoted there and on WormAtlas).
