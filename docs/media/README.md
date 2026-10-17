# README media

Small GIFs embedded in the top-level README, one per reproduction. Each is a
downsized copy of a GIF the script itself writes (max 720 px wide, at most 80
frames, 128 colours); the full-size outputs stay git-ignored and are regenerated
with the commands below. Made 2026-10-15 at commit `5a1b207` plus the `--gif`
options added in the same change; inputs as documented for each script.

| file | script and command | source frames → kept |
|---|---|---|
| `thompson_scarus_pomacanthus.gif` | `python3 thompson/1917-thompson-scarus-pomacanthus.py` → `thompson_scarus_pomacanthus.gif` (2026-10-16) | 70 → 70 |
| `thompson_transformations.gif` | `python3 thompson/1917-thompson.py --gif thompson.gif` (remade 2026-10-16: four families, coaxial circles added) | 196 → 80 (identical hold frames merged) |
| `turing_gray_scott.gif` | `python3 1952-turing-morpho.py` → `turing_patterns.gif` (legacy, tracked at the repo root) | 500 → 60 |
| `raup_shell_coiling.gif` | `python3 create_raup_animations.py` → `raup_shell_animation.gif` (remade 2026-10-17 after the model fix; 660 px, the full-size file is no longer tracked) | 100 → 48 |
| `petroff_stromatolite_dome_to_cone.gif` | `python3 2010-petroff-stromatolite.py --no-show --gif petroff.gif` | 70 → 70 |
| `chindapol_coral_flow_growth.gif` | `python3 2013-chindapol-coral-flow.py --no-show --gif chindapol.gif` | 48 → 48 |
| `planform_morphospace_timeline.gif` | `PLANFORM_DB_PATH=… python3 2013-lobo-planform-morphology.py` → `planform_morphospace_timeline.gif` | 63 → 63 |
| `limbform_morphospace_timeline.gif` | `python3 2014-lobo-limbform-morphology.py` → `limbform_morphospace_timeline.gif` | 66 → 66 |
| `celegans_embryo_timeline.gif` | `python3 2015-santella-wormguides-celegans.py` → `celegans_embryo_timeline.gif` (remade 2026-10-16 at `303a87e`, eggshell outline added) | 120 → 60 |
| `hoffmann_wing_lloyd.gif` | `python3 2018-hoffmann-wing-venation.py --no-show --gif hoffmann.gif` | 35 → 19 (identical hold frames merged) |
| `cervera_levin_mafe_gj_block.gif` | `python3 "2021-Cervera–Levin–Mafe.py" --gif cervera.gif` | 46 → 46 |
| `kahng_coral_geometry_growth.gif` | `python3 2024-kahng-coral-geometry.py --no-show --gif kahng.gif` | 24 → 16 (identical hold frames merged) |
| `odonata_wing_cells_deep_time.gif` | `python3 2026-eshghi-odonata-wing-cells.py --no-resegment --no-crosscheck` → `odonata_wing_cells_deep_time.gif` | 28 → 26 |

Downsizing: frames sampled evenly, resized with Lanczos, quantised with median
cut and Floyd–Steinberg dithering, frame duration scaled to keep the original
running time (Pillow 12.0.0). Total ≈ 15 MB.

`morphoeval_pipeline.png` and `morphoeval_pipeline.gif` are not script outputs but documentation: the source papers (illustrated from the GIFs in this folder; in the GIF version each thumbnail plays its animation once per loop), how a reproduction is made, and the MorphoEval cycle, drawn by `morphoeval/report/draw_pipeline.py --gif` (deterministic; re-run it after editing the stages or regenerating a GIF).

`thompson_scarus_pomacanthus_outlines.png` is a 1400-px, 256-colour copy of the still figure `thompson/1917-thompson-scarus-pomacanthus.py` writes as `thompson_scarus_pomacanthus_outlines.png` (2026-10-17): the real-outline test on two Wikimedia Commons photographs, whose credits (CC BY 4.0 and CC BY 2.0) are printed in the figure and listed in `datasets/thompson/README.md`.

`all_forms_montage.gif` tiles the thirteen animations above (each resampled to 40 frames across its own loop, fitted whole into a 216-px tile, shared 128-colour palette) with a title tile and the MorphoEval overall scores read from the audit YAMLs; `python3 docs/media/make_montage.py` remakes it after any animation or score changes.
