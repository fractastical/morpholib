# MorphoEval report — audit `2026-10-15-celegans-audit`

Rubric **0.1.0** · checkpoint `2864784` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are 0–100 (emitted as `pk:hasProbability` = score/100 on the LLM assessment of the dimension hypothesis); Overall is the rubric-weighted mean and the claim's `pk:weightOfEvidence`. Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Faithfulness | Reproducibility | **Overall** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WormGUIDES C. elegans embryo atlas with the OpenWorm connectome (`2015-santella-wormguides-celegans.py`) | 75 | 60 | 50 | 65 | 85 | 75 | 70 | 63.9 | 76.3 | **67.7** | mostly faithful |

## Findings and remediation

### WormGUIDES C. elegans embryo atlas with the OpenWorm connectome  —  overall 67.7 (mostly faithful)

Downloads the WormGUIDES atlas at a pinned commit and verifies 416 files against a SHA-256 manifest; decodes the 360 StarryNite nuclei tables with the column semantics of the WormGUIDES loader (valid flag, x, y, z, diameter, lineage name) and the movie-start offset of ProductionInfo.csv; checks that all 1330 distinct names follow the Sulston grammar and that every cell whose parent is in the movie appears after it (1326/1326); re-derives the lineage chart from the names; sets the unit-less coordinates to a 50 um egg (0.1617 um per unit, stated as an assumption); draws the nuclei to scale at six stages, 51 single-cell meshes at frame 330 and the c302 connectome adjacency matrix (300 neurons after name canonicalisation). Against WormAtlas's quotation of Sulston et al. the cell-stage anchors agree (113 listed deaths all present, 562-571 nuclei at 330-350 min against ~550, E-cell ingression at 26 cells against the ~30-cell stage) but the clock runs ~27 min ahead (gastrulation at movie-minute 73 against 100), which the script reports and does not resolve.

- **Model fidelity to source: 75** — The atlas is decoded as the WormGUIDES application decodes it: the column positions follow AceTreeTableLineageDataLoader, the time offset and unit scale are read from ProductionInfo.csv rather than assumed, and mesh cells are taken from CellShapesConfig.csv. The Sulston naming grammar and parent-of relation are implemented explicitly and tested on every name. Not higher because the physical scale is a stated assumption, because the 3D atlas is shown as a 2D projection (back-to-front by z) rather than the application's 3D rendering, and because the connectome layer is a summed adjacency matrix that drops the chemical/electrical distinction and Cook et al.'s muscle and end-organ nodes.  
  evidence: `2015-santella-wormguides-celegans.py` L246–267; `2015-santella-wormguides-celegans.py` L269–308; `2015-santella-wormguides-celegans.py` L313–319; `2015-santella-wormguides-celegans.py` L501–534
- **Morphological realism / specimen match: 60** — The re-derived lineage chart has the shape of Sulston's diagram (founder sub-trees in order, AB dominating, P4 ending in two cells), the nuclear clouds have the egg's aspect under the calibration (49 x 27 x 25 um at the end), and gastrulation shows as the E cells moving inward. Distinct stages are visually distinct. But no photograph, DIC series or WormGUIDES screenshot is placed beside the output, the WormAtlas images are only cited, and the meshes at frame 330 are drawn as flat projections, so resemblance to the real embryo is argued from counts and extents rather than shown side by side.  
  evidence: `celegans_four_views.png` L0–0; `2015-santella-wormguides-celegans.py` L384–428; `2015-santella-wormguides-celegans.py` L433–448; `2015-santella-wormguides-celegans.py` L461–480
- **Parameter expressivity: 50** — Not applicable: the script decodes a recorded atlas, it has no morphospace parameters. The only knobs are the calibration length, the frames shown and the gastrulation detector's threshold, all of which are constants stated in the code. Scored neutral so that it neither rewards nor penalises the overall.  
  evidence: `2015-santella-wormguides-celegans.py` L106–110; `2015-santella-wormguides-celegans.py` L341–349
- **Morphometric validation: 65** — Per-frame morphometrics are computed and written to CSV (nucleus count by founder, extents, mean diameter, normalised E-cell radius) and compared in the script's own output against the literature: 113 deaths listed and all present, 562-571 nuclei at 330-350 min against ~550, 580 nuclei at the last frame against 558 at hatching (671 generated), E-cell ingression at 26 cells against the ~30-cell stage, and the 300 connectome neurons against 302. The timing anchor disagrees (movie-minute 73 against 100 min) and is reported as a discrepancy, not explained. No nuclear-diameter or egg-width reference is compared, and the calibration itself is unverified.  
  evidence: `2015-santella-wormguides-celegans.py` L321–338; `2015-santella-wormguides-celegans.py` L352–370; `morphoeval/measure_outputs.py` L383–450; `celegans_four_views.png` L0–0
- **Source and data traceability: 85** — Both repositories are pinned to a commit and licence, every input file has a SHA-256 in datasets/celegans/pinned_inputs.sha256 that is checked on every run, the papers are cited with DOIs, the acquisition metadata (image series, strain, frame interval, movie start) is read from the atlas's own ProductionInfo.csv, and the WormAtlas pages are cited with URL and access date. The Sulston counts and timings are quoted second-hand via WormAtlas rather than from the 1983 paper.  
  evidence: `2015-santella-wormguides-celegans.py` L24–47; `2015-santella-wormguides-celegans.py` L90–104; `2015-santella-wormguides-celegans.py` L155–190; `datasets/celegans/pinned_inputs.sha256` L1–3
- **Code-to-model traceability: 75** — The docstring names the loader whose column semantics are followed, the ProductionInfo fields used, the source of every literature constant and the calibration reference; each check prints the value and the literature anchor it is compared with. The nuclei-file format itself (StarryNite/AceTree) is documented only by the WormGUIDES loader source, not by a paper section, so the trace stops at code rather than at a publication.  
  evidence: `2015-santella-wormguides-celegans.py` L1–22; `2015-santella-wormguides-celegans.py` L107–117; `2015-santella-wormguides-celegans.py` L258–267
- **Reproducibility package: 70** — Deterministic, invocation documented in the docstring and README, the inputs are fetched and hash-verified by the script itself and missing mesh files are handled (the shape config's frame ranges are sparse), outputs are identical on re-run. First run needs network access to raw.githubusercontent.com; dependencies are in requirements.txt with lower bounds only, and the known-good environment is recorded only in this audit.  
  evidence: `2015-santella-wormguides-celegans.py` L49–56; `2015-santella-wormguides-celegans.py` L193–218; `2015-santella-wormguides-celegans.py` L607–650; `requirements.txt` L1–7

Remediation:
- Place a WormGUIDES figure panel (Santella et al. 2015 is CC BY) or a DIC frame beside the decoded nuclei at the same stage and measure the agreement.
- Resolve the ~27 min clock offset against Sulston's schedule (recording temperature, or the meaning of "movie start time 20 min") with the Bao lab or the diSPIM metadata.
- Verify the 50 um calibration against the atlas's voxel size (the DISPIM_061416 acquisition parameters) instead of assuming it.
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
| calibration | 0.1617 µm per atlas unit | — | longest nuclear cloud (centroid span + one mean diameter) set to the 50 µm egg length *(assumption: Gönczy et al. 1999 JCB 144:927 ('50-µm-long cell'); the atlas coordinates are unit-less (ProductionInfo X/Y/Z_SCALE = 1))* |
| nuclear cloud width at the last frame | 26.75 µm | — | y span of nuclear centroids + one mean diameter under the calibration *(no width reference recorded; nuclei do not reach the eggshell)* |
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

Inputs: 416 files from zhirongbaolab/WormGUIDES at commit 17be611f (GPL-3.0; 360 nuclei tables of image series DISPIM_061416 / strain JIM113, 51 cell meshes at frame 330, CellShapesConfig.csv, ProductionInfo.csv, partslist.txt, CellDeaths.csv) and openworm/c302 at commit f54d92d7 (MIT; c302/data/herm_full_edgelist.csv), downloaded by the script into the git-ignored .celegans_cache/ and verified against datasets/celegans/pinned_inputs.sha256 on every run.
