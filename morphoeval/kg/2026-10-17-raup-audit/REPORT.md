# MorphoEval report — audit `2026-10-17-raup-audit`

Rubric **0.1.0** · checkpoint `ba6fd20` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are 0–100 (emitted as `pk:hasProbability` = score/100 on the LLM assessment of the dimension hypothesis); Overall is the rubric-weighted mean and the claim's `pk:weightOfEvidence`. Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Faithfulness | Reproducibility | **Overall** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Raup shell coiling morphospace (`create_raup_animations.py`) | 80 | 45 | 70 | 45 | 75 | 75 | 80 | 58.6 | 77.0 | **64.1** | mostly faithful |

## Findings and remediation

### Raup shell coiling morphospace  —  overall 64.1 (mostly faithful)

raup_shell(W, D, T, S) sweeps an ellipse of radial semi-axis a = a0 W^(theta/2pi) and axial semi-axis a/S, centred at r_c = a(1+D)/(1-D) from the axis and z_c = T r_c along it, in the plane through the axis. self_test() recovers W, D, T from meshes on a 3x3x4 grid to 4.9e-15 and checks the spire's axial travel rises strictly with T (the outer height/width may stay flat while the last whorl still envelops the spire, which is correct behaviour for involute forms); a T=0 shell is checked to be exactly one aperture thick. Six illustrative panels (W, D, T from variables.json) rendered rotating in side and overhead view; W x D and W x T sheets; a six-panel still with height/width; a CSV of nominal and recovered W, D, T per panel with height/width and spire/width. The measurements sidecar recovers W, D, T again, independently, from the script's generating function.

- **Model fidelity to source: 80** — The three parameters now mean what Raup's model says they mean, and the script proves it on its own output: W is the per-revolution magnification of every dimension, D the inner/outer-margin ratio of the generating curve, T the axial over radial movement of its centre, and the generating curve lies in the plane through the axis, so the shells are self-similar tubes rather than flat ribbons. The three defects of the first version are each named in the docstring and each is excluded by the self-test. Not higher because the primary paper was not consulted: the definitions are taken from two secondary restatements (which agree with each other on W, D and T up to Noshita's 2/(1+D) scaling of T, which is noted), the direction of the S ratio is the repository's convention rather than Raup's, and Raup's generating curve is an arbitrary aperture outline, here always an ellipse.  
  evidence: `create_raup_animations.py` L20–36; `create_raup_animations.py` L96–104; `create_raup_animations.py` L38–45; `create_raup_animations.py` L134–157; `1966-raup.py` L28–37; `datasets/raup/variables.json` L134–140
- **Morphological realism / specimen match: 45** — The six forms are now distinct and each is what its name says (spire/width 0.00 for the two planispirals, 0.86 for the turreted form, 0.74 high-spired, 0.48 conical, 0.34 globose), and the panel titles and a figure footnote say that the taxon names are resemblances, not specimens. But there is still no real shell in the figure and no measured specimen the panels are tested against, which is what the dimension asks for; the labels are honest, not validated. Scored above the 30 of the first audit because the forms are no longer wrong, below the midpoint because nothing checks that a Turritella looks like the turreted panel.  
  evidence: `raup_empirical_comparison.png` L0–0; `raup_shell_animation.gif` L0–0; `create_raup_animations.py` L174–176; `datasets/raup/variables.json` L66–133
- **Parameter expressivity: 70** — Each of W, D, T now has its own visible monotone effect, shown on two sheets (W x D at fixed T, W x T at fixed D) and on the 3 x 3 W x T slice of 1966-raup.py, and the self-test asserts the T effect. S is exposed in the function but not swept in any figure, and D is swept only together with W rather than on its own; the ranges on the sheets are the dataset's indicative bounds, not Raup's.  
  evidence: `create_raup_animations.py` L236–259; `1966-raup.py` L40–71; `create_raup_animations.py` L147–151
- **Morphometric validation: 45** — Shape quantities are now measured on the rendered geometry - W, D and T recovered from each mesh, height/width and spire/width per panel - written to a CSV, printed, shown on the still, and re-derived independently in the measurements sidecar; recovered values equal the inputs to 5e-15 in the script's own check and, in the sidecar's independent derivation, exactly for W and T and to 3e-4 for D (the aperture's inner margin is read off a 100-point grid). That is validation of the implementation against the definitions, and it is complete. What the dimension also asks for - a comparison with specimens or with the paper's own figures - is absent: no published W, D, T of a real shell, and none of Raup's block-diagram placements, is compared with anything here.  
  evidence: `create_raup_animations.py` L110–131; `create_raup_animations.py` L189–204; `raup_shell_metrics.csv` L0–0
- **Source and data traceability: 75** — Raup 1962, 1965 and 1966 are cited with DOIs (the 1966 JSTOR id, wrong in the dataset since 2025, is corrected), the two secondary sources the definitions actually come from are cited with DOIs and the specific statement used from each is given, and the dataset says plainly which of its earlier contents could not be verified and why they were replaced. Not higher because the primary paper was not read, so nothing here is pinned to a page or equation of Raup 1966, and because the parameter sets and ranges are the author's illustrative choices with no external source.  
  evidence: `create_raup_animations.py` L6–18; `datasets/raup/README.md` L1–30; `datasets/raup/variables.json` L10–10; `datasets/raup/paper_citations.bib` L1–70
- **Code-to-model traceability: 75** — The docstring maps each parameter to its definition and its line of code, the generating function is commented line by line with the identity each line enforces, the self-test names what it excludes, and the parameter sets carry a note on why each value. Not higher because the equations are tied to the secondary sources' notation rather than to numbered equations of Raup 1966.  
  evidence: `create_raup_animations.py` L81–107; `1966-raup.py` L1–18; `datasets/raup/README.md` L32–52
- **Reproducibility package: 80** — Deterministic (no RNG), no external data, a CLI with --frames, --fps, --no-gif, --self-test-only and --out-dir, a self-test that runs before any drawing and fails loudly, outputs that fill their panels, and the minimal script runs headless with --no-show/--out. numpy, matplotlib and Pillow are in requirements.txt and pinned in requirements.lock.txt. Not higher because the GIF is 16 MB and 100 frames for a 7 s loop, and because Pillow silently merges 20 of the 120 rendered frames, which the script does not report.  
  evidence: `create_raup_animations.py` L280–300; `create_raup_animations.py` L178–183; `1966-raup.py` L40–44

Remediation:
- Obtain Raup 1966 and pin the definitions of W, D, T, S to its equations and figures; confirm the direction of the S ratio.
- Put a real shell beside at least one panel - a photograph, or a specimen's measured (W, D, T) from Raup 1967's ammonoid table or a later gastropod dataset - and measure the agreement.
- Sweep S and D on their own in the parameter sheet; mark Raup's block-diagram regions (gastropods, ammonoids, bivalves, brachiopods) on the W x T and W x D sheets once the paper is in hand.
- Report the stored frame count after saving the GIF, or write it with a writer that does not merge frames.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Raup shell coiling morphospace
- `raup_empirical_comparison.png` — sha256 `c50dbee57d0a6495…`, 562,658 B, 1630×1070 px
- `raup_shell_animation.gif` — sha256 `986555a0032be189…`, 15,189,434 B, 1200×800 px, 100 frames
- `raup_shell_metrics.csv` — sha256 `384d298d81f5373f…`, 483 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| recovered translation rate T, 'Planispiral' (W=3.0, D=0.0, T=0.0) | 0.0 ratio | 0.0 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.629, spire/width = 0.000)* |
| recovered distance from axis D, 'Planispiral' (W=3.0, D=0.0, T=0.0) | 0.0003 ratio | 0.0 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Planispiral' (W=3.0, D=0.0, T=0.0) | 3.0 ratio | 3.0 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Turreted' (W=1.3, D=0.05, T=3.5) | 3.5 ratio | 3.5 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 1.142, spire/width = 0.858)* |
| recovered distance from axis D, 'Turreted' (W=1.3, D=0.05, T=3.5) | 0.0502 ratio | 0.05 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Turreted' (W=1.3, D=0.05, T=3.5) | 1.3 ratio | 1.3 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Evolute disc' (W=2.0, D=0.45, T=0.0) | 0.0 ratio | 0.0 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.321, spire/width = 0.000)* |
| recovered distance from axis D, 'Evolute disc' (W=2.0, D=0.45, T=0.0) | 0.4501 ratio | 0.45 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Evolute disc' (W=2.0, D=0.45, T=0.0) | 2.0 ratio | 2.0 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Conical' (W=3.5, D=0.0, T=1.5) | 1.5 ratio | 1.5 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.805, spire/width = 0.481)* |
| recovered distance from axis D, 'Conical' (W=3.5, D=0.0, T=1.5) | 0.0003 ratio | 0.0 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Conical' (W=3.5, D=0.0, T=1.5) | 3.5 ratio | 3.5 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'High-spired' (W=1.6, D=0.1, T=2.5) | 2.5 ratio | 2.5 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.998, spire/width = 0.738)* |
| recovered distance from axis D, 'High-spired' (W=1.6, D=0.1, T=2.5) | 0.1002 ratio | 0.1 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'High-spired' (W=1.6, D=0.1, T=2.5) | 1.6 ratio | 1.6 | centre distance from the axis at the aperture / one revolution earlier |
| recovered translation rate T, 'Globose' (W=3.0, D=0.1, T=1.0) | 1.0 ratio | 1.0 | Δz_c / Δr_c of the generating-curve centre over the last revolution of raup_shell() *(height/width = 0.628, spire/width = 0.341)* |
| recovered distance from axis D, 'Globose' (W=3.0, D=0.1, T=1.0) | 0.1002 ratio | 0.1 | inner margin / outer margin of the aperture (min/max radial distance of the last generating curve) |
| recovered whorl expansion W, 'Globose' (W=3.0, D=0.1, T=1.0) | 3.0 ratio | 3.0 | centre distance from the axis at the aperture / one revolution earlier |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- matplotlib: 3.10.7
- Pillow: 12.0.0

No external inputs: the six parameter sets and the parameter ranges are read from datasets/raup/variables.json in the repository. Outputs regenerated at this commit with python3 create_raup_animations.py (raup_shell_animation.gif 1200x800, 120 frames rendered / 100 stored; raup_parameter_space.png; raup_empirical_comparison.png; raup_shell_metrics.csv).
