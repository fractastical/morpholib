# MorphoEval report — audit `2026-10-16-celegans-audit`

Rubric **0.1.0** · checkpoint `303a87e` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are 0–100 (emitted as `pk:hasProbability` = score/100 on the LLM assessment of the dimension hypothesis); Overall is the rubric-weighted mean and the claim's `pk:weightOfEvidence`. Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Faithfulness | Reproducibility | **Overall** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WormGUIDES C. elegans embryo atlas with the OpenWorm connectome (`2015-santella-wormguides-celegans.py`) | 80 | 62 | 50 | 72 | 85 | 75 | 70 | 67.6 | 76.3 | **70.2** | mostly faithful |

## Findings and remediation

### WormGUIDES C. elegans embryo atlas with the OpenWorm connectome  —  overall 70.2 (mostly faithful)

Downloads the WormGUIDES atlas at a pinned commit and verifies 429 files against a SHA-256 manifest; decodes the 360 StarryNite nuclei tables with the column semantics of the WormGUIDES loader (valid flag, x, y, z, diameter, lineage name) and the movie-start offset of ProductionInfo.csv; checks that all 1330 distinct names follow the Sulston grammar and that every cell whose parent is in the movie appears after it (1326/1326); re-derives the lineage chart from the names; sets the atlas's own eggshell outline mesh at frame 330 to the 50 um egg length (0.1649 um per unit; the shell's unfitted width comes out 29.2 um, the 13 sampled shell meshes bracket the scale at 0.1649-0.1750, the earlier nuclear-cloud estimate was 0.1617); draws the nuclei to scale inside the shell outline at six stages, 51 single-cell meshes at frame 330 and the c302 connectome adjacency matrix (300 neurons after name canonicalisation). Reports that the atlas reuses a static shell mesh for frames 1-~250 that the nuclei overrun by up to 19 units. Against WormAtlas's quotation of Sulston et al. the cell-stage anchors agree (113 listed deaths all present, 562-571 nuclei at 330-350 min against ~550, E-cell ingression at 26 cells against the ~30-cell stage) but the clock runs ~27 min ahead (gastrulation at movie-minute 73 against 100), which the script reports and does not resolve.

- **Model fidelity to source: 80** — Re-scored from 75. The atlas is decoded as the WormGUIDES application decodes it: the column positions follow AceTreeTableLineageDataLoader, the time offset and unit scale are read from ProductionInfo.csv, mesh cells and the eggshell outline are taken from CellShapesConfig.csv. The Sulston naming grammar and parent-of relation are implemented explicitly and tested on every name. The physical scale is now anchored to the atlas's own segmentation of the eggshell rather than to the nuclear cloud, so the one remaining assumption is that this egg has the typical 50 um length; the choice of the frame-330 shell over the static early mesh is justified in the code by the measured nuclear overrun. Not higher because that assumption remains, because the 3D atlas is shown as a 2D projection (back-to-front by z) rather than the application's 3D rendering, and because the connectome layer is a summed adjacency matrix that drops the chemical/electrical distinction and Cook et al.'s muscle and end-organ nodes.  
  evidence: `2015-santella-wormguides-celegans.py` L254–275; `2015-santella-wormguides-celegans.py` L277–319; `2015-santella-wormguides-celegans.py` L102–113; `2015-santella-wormguides-celegans.py` L335–368; `2015-santella-wormguides-celegans.py` L564–595
- **Morphological realism / specimen match: 62** — Re-scored from 60. The re-derived lineage chart has the shape of Sulston's diagram (founder sub-trees in order, AB dominating, P4 ending in two cells); the nuclei are now drawn inside the atlas's eggshell outline, which fills the 50 x 29 um egg and shows the cloud growing out to the shell, and gastrulation shows as the E cells moving inward. Distinct stages are visually distinct. The gain is small because the shell is the atlas's segmentation, not a photograph: no DIC frame or WormGUIDES screenshot is placed beside the output, the WormAtlas images are only cited, and the meshes at frame 330 are drawn as flat projections, so resemblance to the real embryo is still argued from counts and extents rather than shown side by side.  
  evidence: `celegans_four_views.png` L0–0; `2015-santella-wormguides-celegans.py` L433–480; `2015-santella-wormguides-celegans.py` L482–508; `2015-santella-wormguides-celegans.py` L520–541
- **Parameter expressivity: 50** — Not applicable: the script decodes a recorded atlas, it has no morphospace parameters. The only knobs are the calibration length, the frames shown and the gastrulation detector's threshold, all of which are constants stated in the code. Scored neutral so that it neither rewards nor penalises the overall.  
  evidence: `2015-santella-wormguides-celegans.py` L113–122; `2015-santella-wormguides-celegans.py` L390–398
- **Morphometric validation: 72** — Re-scored from 65. Per-frame morphometrics are computed and written to CSV (nucleus count by founder, extents, mean diameter, normalised E-cell radius) and compared in the script's own output against the literature: 113 deaths listed and all present, 562-571 nuclei at 330-350 min against ~550, 580 nuclei at the last frame against 558 at hatching (671 generated), E-cell ingression at 26 cells against the ~30-cell stage, and the 300 connectome neurons against 302. The calibration now has checks of its own: the fitted length leaves the shell width unfitted and it comes out 29.2 um against the ~30 um egg; the 13 sampled shell meshes bound the scale to 0.1649-0.1750 um per unit; the nuclear overrun of each shell mesh is measured (19.0 units for the static early mesh, 1.8 at frame 330); and the previous nuclear-cloud estimate is recomputed as a cross-check (2 % lower). The timing anchor still disagrees (movie-minute 73 against 100 min) and is reported as a discrepancy, not explained, and no nuclear-diameter reference is compared.  
  evidence: `2015-santella-wormguides-celegans.py` L348–368; `2015-santella-wormguides-celegans.py` L370–388; `2015-santella-wormguides-celegans.py` L401–423; `morphoeval/measure_outputs.py` L383–468; `celegans_four_views.png` L0–0
- **Source and data traceability: 85** — Both repositories are pinned to a commit and licence, every input file has a SHA-256 in datasets/celegans/pinned_inputs.sha256 that is checked on every run, the papers are cited with DOIs, the acquisition metadata (image series, strain, frame interval, movie start) is read from the atlas's own ProductionInfo.csv, and the WormAtlas pages are cited with URL and access date. The Sulston counts and timings are quoted second-hand via WormAtlas rather than from the 1983 paper.  
  evidence: `2015-santella-wormguides-celegans.py` L24–49; `2015-santella-wormguides-celegans.py` L92–111; `2015-santella-wormguides-celegans.py` L163–199; `datasets/celegans/pinned_inputs.sha256` L1–3
- **Code-to-model traceability: 75** — The docstring names the loader whose column semantics are followed, the ProductionInfo fields used, the source of every literature constant and the calibration reference; each check prints the value and the literature anchor it is compared with. The nuclei-file format itself (StarryNite/AceTree) is documented only by the WormGUIDES loader source, not by a paper section, so the trace stops at code rather than at a publication.  
  evidence: `2015-santella-wormguides-celegans.py` L1–22; `2015-santella-wormguides-celegans.py` L113–124; `2015-santella-wormguides-celegans.py` L266–275
- **Reproducibility package: 70** — Deterministic, invocation documented in the docstring and README, the inputs are fetched and hash-verified by the script itself and missing mesh files are handled (the shape config's frame ranges are sparse), outputs are identical on re-run. First run needs network access to raw.githubusercontent.com; dependencies are in requirements.txt with lower bounds only, and the known-good environment is recorded only in this audit.  
  evidence: `2015-santella-wormguides-celegans.py` L32–40; `2015-santella-wormguides-celegans.py` L201–227; `2015-santella-wormguides-celegans.py` L674–723; `requirements.txt` L1–7

Remediation:
- Place a WormGUIDES figure panel (Santella et al. 2015 is CC BY) or a DIC frame beside the decoded nuclei at the same stage and measure the agreement.
- Resolve the ~27 min clock offset against Sulston's schedule (recording temperature, or the meaning of "movie start time 20 min") with the Bao lab or the diSPIM metadata.
- Replace the 50 um assumption with the acquisition voxel size of DISPIM_061416 (not in the repository's ProductionInfo.csv; ask the Bao lab), and ask why the shell mesh is static for frames 1-~250 while the nuclei overrun it.
- Draw the connectome with chemical and electrical edges separated, and keep the muscle/end-organ nodes that Cook et al. include.
- Pin the Python 3.14 environment (exact versions) in requirements or a lock file.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### WormGUIDES C. elegans embryo atlas with the OpenWorm connectome
- `celegans_four_views.png` — sha256 `eb4b60086280c9a1…`, 424,197 B, 2240×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| lineage names parsing as Sulston names | 1330 count | 1330 | distinct names in the 360 nuclei files matching ^(AB|MS|E|C|D|EMS|P[0-4]|Z[23])[aplrdv]*$ |
| cells first seen after their parent | 1326 count | 1326 | non-founder cells whose parent name appears in an earlier frame |
| calibration | 0.1649 µm per atlas unit | — | x extent of the atlas's eggshell outline mesh at frame 330 (obj_files/embryo_t330.obj, 'Embryo Outline' in CellShapesConfig.csv; 303.1 units) set to the 50 µm egg length *(assumption: Gönczy et al. 1999 JCB 144:927 ('50-µm-long cell'); the atlas coordinates are unit-less (ProductionInfo X/Y/Z_SCALE = 1); the earlier nuclear-cloud estimate gives 0.1617; the sampled shell meshes give 0.1649-0.1750)* |
| eggshell width under the calibration | 29.238 µm | 30.0 | y extent of the eggshell outline mesh at frame 330 (z extent 28.9 µm) *(a consistency check of the calibration: the width is not fitted)* |
| eggshell length spread across sampled shell meshes | 17.3733 atlas units | — | max - min x extent of the eggshell meshes at frames [1, 30, 60, 100, 200, 250, 280, 300, 320, 330, 340, 350, 359] (285.8-303.1) *(one static mesh is reused for frames 1-~250, re-segmented per frame from ~280; the spread bounds the scale uncertainty (~6 %))* |
| nuclear overrun of the shell in x | 19.0129 atlas units | — | largest distance by which a nucleus (centroid ± radius) lies outside the shell mesh's x range at the sampled frames *(1.8 units at frame 330; the overrun is against the static early mesh, which is why the calibration uses the re-segmented shell)* |
| nuclear cloud width at the last frame | 27.28 µm | — | y span of nuclear centroids + one mean diameter under the calibration *(nuclei do not reach the eggshell; compare the eggshell width above)* |
| nuclei at the last frame | 580 count | 558 | valid rows of t360-nuclei (movie-minute 379) *(the movie ends before hatching; 553 of the 580 nuclei carry a parts-list identity)* |
| maximum nucleus count | 585 count | 550 | maximum over the 360 frames |
| deaths listed by the atlas | 113 count | 113 | names in CellDeaths.csv |
| listed dying cells that occur in the movie | 113 count | 113 | CellDeaths.csv names present in at least one nuclei file |
| gastrulation onset, cell count | 26 count | 30 | first frame from which the E cells sit at < 0.8 of the median normalised radius for 5 frames |
| gastrulation onset, movie-minute | 73 min | 100 | same frame, frame index + movie start offset *(discrepancy: the atlas clock runs ~27 min ahead of the Sulston schedule at the same cell stage; not resolved by the script (temperature or time origin))* |
| nuclei at movie-minute 330-350 | 562-571 count | 550 | nucleus count over that window |
| neurons in the connectome edge list | 300 count | 302 | upper-case node names of c302 herm_full_edgelist.csv after DA01->DA1 canonicalisation *(CANL/CANR are 'no synapses' in the WormGUIDES parts list and so cannot appear in an edge list)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- matplotlib: 3.10.7
- Pillow: 12.0.0

Inputs: 429 files from zhirongbaolab/WormGUIDES at commit 17be611f (GPL-3.0; 360 nuclei tables of image series DISPIM_061416 / strain JIM113, 51 cell meshes at frame 330, 13 eggshell outline meshes embryo_t{1,30,60,100,200,250,280,300,320,330,340,350,359}.obj, CellShapesConfig.csv, ProductionInfo.csv, partslist.txt, CellDeaths.csv) and openworm/c302 at commit f54d92d7 (MIT; c302/data/herm_full_edgelist.csv), downloaded by the script into the git-ignored .celegans_cache/ and verified against datasets/celegans/pinned_inputs.sha256 on every run.
