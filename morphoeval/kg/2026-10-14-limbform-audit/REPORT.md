# MorphoEval report — audit `2026-10-14-limbform-audit`

Rubric **0.1.0** · checkpoint `747c57f` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LimbformDB limb morphologies (`2014-lobo-limbform-morphology.py`) | 85 | 65 | 75 | 55 | 85 | 75 | 65 | 81.7 | 61.2 | 74.3 | 71.3 | **71.3** | mostly faithful |

## Findings and remediation

### LimbformDB limb morphologies  —  overall 71.3 (mostly faithful)

Decodes the Limbform formalism (segments as graph edges with length and direction, joints as vertices with border radii, sub-joints for grafts, digits with phalanx counts) for 555 of the 565 morphology rows of LimbformDB 2.1.0 — the other 10 are outcome labels without geometry — and draws each limb as a smooth outline through the joint control points with the paper's proximodistal red-to-blue gradient, green coxa, yellow basis/trochanter, black long bones and blue digits. Against the paper's own cartoons of the five Fig. 1 wild types the decoded silhouettes have the same topology, pose and segment/digit counts; bounding-box-aligned IoU is 0.41-0.74 (best for the cricket leg, worst for the frog hindlimb, where the thigh-shank angle differs). The joint-radius rule of the formalism holds for 3137 of 3146 joints when a sub-joint counts its host segment twice.

- **Model fidelity to source: 85** — The encoding is read as the formalism defines it: Link rows are segments with Type, Dist, Ang, Side/Reversed/Inverted and NumBones; JointParam radii are one per bisector of consecutive segments at the joint or +90/+180/+270 deg for a terminal joint; LinkSubJoint places grafts at (Dist, Ang) on their host segment; Digit rows hang off the hand/foot segment. The rule is checked against every joint at load time (3137/3146) and the 9 under-parameterised joints are padded with the joint's mean radius, which is stated in the code. Short of full marks because the papers describe the formalism in prose, so the bisector-order convention and the 'host segment counts twice' rule for sub-joints were established by checking the database, and because the residual silhouette differences against Fig. 1 (below) leave the Ang convention not fully confirmed.  
  evidence: `2014-lobo-limbform-morphology.py` L26–47; `2014-lobo-limbform-morphology.py` L222–260; `2014-lobo-limbform-morphology.py` L314–352; `2014-lobo-limbform-morphology.py` L606–620
- **Morphological realism / specimen match: 65** — The only published renderings of these database entries are the cartoons in Fig. 1 of the Regeneration paper; side by side, the decoded axolotl forelimb, Xenopus hindlimb, fiddler-crab walking leg, cricket T3 leg and huntsman-spider leg have the same segments, digits, pose and colour scheme (gradient, coxa green, basis yellow, bones, digits). Measured on the figure, bounding-box-aligned silhouette IoU is 0.64, 0.41, 0.43, 0.74 and 0.49 for panels A''-E'': the cricket leg is essentially identical, the others differ in joint angles and segment width (the frog thigh-shank angle most of all), which may reflect a database revision after the figure was drawn or an angle convention not captured. Distinct morphologies (double limbs, grafts, branched chelipeds, missing segments) render distinctly. No photographs of real limbs are compared.  
  evidence: `limbform_wildtype_gallery.png` L0–0; `2014-lobo-limbform-morphology.py` L419–453; `2014-lobo-limbform-morphology.py` L632–666; `2014-lobo-limbform-morphology.py` L678–704
- **Parameter expressivity: 75** — Every encoded number has a visible effect: each radius moves one control point of the outline, each segment's Dist/Ang moves its distal joint and everything beyond it, each sub-joint places a graft, each digit's Dist/Ang/NumBones draws one digit. Morphologies in one frame share one scale, so size differences between phenotypes are preserved. There is no parameter sweep in the output and Side/Reversed/Inverted are only reported (no 2.1.0 entry uses them), so their effect is untested.  
  evidence: `2014-lobo-limbform-morphology.py` L286–312; `2014-lobo-limbform-morphology.py` L377–400; `2014-lobo-limbform-morphology.py` L788–862
- **Morphometric validation: 55** — The script computes per-morphology morphometrics (segment and digit counts, proximodistal path length, bounding box, silhouette area) and writes them as CSV, and the measurements sidecar compares the five Fig. 1 wild types quantitatively with the paper's own figure (segment and digit counts against the labelled graphs A'-E', all equal; silhouette IoU and bounding-box aspect against the cartoons A''-E''). Those are comparisons with the source authors' rendering of the same data, not with measurements of real limbs, and the IoU values themselves show the agreement is partial.  
  evidence: `2014-lobo-limbform-morphology.py` L587–604; `2014-lobo-limbform-morphology.py` L569–585; `2014-lobo-limbform-morphology.py` L869–883; `limbform_wildtype_gallery.png` L0–0
- **Source and data traceability: 85** — Both papers are cited with DOIs, the database version, download URL, byte size and SHA-256 are recorded and verified on every run, the schema tables used are listed, and the reference figure used by the measurements is pinned by hash. The database's licence is not stated on the Lobo-lab page or in the file, which is why it is cached rather than vendored; that gap is the source's, but it is recorded here.  
  evidence: `2014-lobo-limbform-morphology.py` L14–24; `2014-lobo-limbform-morphology.py` L89–93; `2014-lobo-limbform-morphology.py` L200–219; `2014-lobo-limbform-morphology.py` L263–284
- **Code-to-model traceability: 75** — The module docstring states the formalism table by table and names the paper section it follows; the radius rule, the sub-joint rule, the Fig. 1 colour conventions and the bones-for-tetrapods-only convention are documented where they are implemented. As with the Planform sibling, the papers give no numbered equations, so the rules were confirmed against the database rather than quoted.  
  evidence: `2014-lobo-limbform-morphology.py` L1–13; `2014-lobo-limbform-morphology.py` L113–125; `2014-lobo-limbform-morphology.py` L330–352
- **Reproducibility package: 65** — Deterministic (no random state), invocation documented in the docstring and README, the input is fetched and hash-verified by the script itself, rows without geometry are skipped with a message, outputs are identical on re-run. Dependencies are in requirements.txt with lower bounds only and the known-good environment (Python 3.14) is recorded only in this audit.  
  evidence: `2014-lobo-limbform-morphology.py` L49–60; `2014-lobo-limbform-morphology.py` L885–921; `requirements.txt` L1–7

Remediation:
- Confirm the Ang convention and the joint-radius order against the Limbform software source or the authors, and explain the Fig. 1 B''/C'' silhouette residuals.
- Compare at least the five wild types with photographs of the real limbs (Fig. 1 A-E), e.g. segment-length ratios, not only with the paper's cartoons.
- Pin the Python 3.14 environment (exact versions) in requirements or a lock file.
- Ask the Lobo lab for the LimbformDB licence so the file can be vendored with its hash.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### LimbformDB limb morphologies
- `limbform_wildtype_gallery.png` — artifact_sha256 `d561f4aab1075697…`, 620,043 B, 2400×1875 px

| measurement | value | reference | method |
|---|---:|---|---|
| morphologies decoded | 555 count | 565 | Morphology rows with at least one Link (segment) row *(the rest are outcome labels without geometry (Lethal, No limb, ...))* |
| joints obeying the radius rule | 3137 count | 3146 | joints whose JointParam count equals the number of incident segments (3 for a terminal joint; a sub-joint counts its host segment twice) |
| segments, #1 Wild type, right forelimb | 3 count | 3 | Link rows of the morphology |
| digits, #1 Wild type, right forelimb | 4 count | 4 | Digit rows of the morphology |
| segments, #280 Wild type, right hindlimb | 3 count | 3 | Link rows of the morphology |
| digits, #280 Wild type, right hindlimb | 6 count | 6 | Digit rows of the morphology |
| segments, #314 Wild type, right walking leg, fiddler crab | 7 count | 7 | Link rows of the morphology |
| segments, #490 Wild type, right metathoracic (T3) leg, cricket nymph | 8 count | 8 | Link rows of the morphology |
| segments, #491 Wild type, right leg I, huntsman spider | 7 count | 7 | Link rows of the morphology |
| silhouette IoU vs Fig. 1 A'', #1 Wild type, right forelimb | 0.6448 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #1 Wild type, right forelimb | 1.5738 ratio | 1.463 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 B'', #280 Wild type, right hindlimb | 0.4105 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot0+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #280 Wild type, right hindlimb | 1.4696 ratio | 1.832 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 C'', #314 Wild type, right walking leg, fiddler crab | 0.4313 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #314 Wild type, right walking leg, fiddler crab | 1.2414 ratio | 1.051 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 D'', #490 Wild type, right metathoracic (T3) leg, cricket nymph | 0.7432 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #490 Wild type, right metathoracic (T3) leg, cricket nymph | 1.1346 ratio | 1.084 | longer / shorter side of the decoded silhouette's bounding box |
| silhouette IoU vs Fig. 1 E'', #491 Wild type, right leg I, huntsman spider | 0.4851 ratio | 1.0 | largest ink component of the cartoon panel (holes filled) and the decoded silhouette, each cropped to its bounding box and scaled to a 200-px canvas (aspect kept); intersection over union, best of 8 rotations/reflections (rot180+mirror) *(bones, digits and gradient are inside the outline, so this compares outer shape only; the cartoon is a 560-px JPEG, which bounds the achievable IoU below 1)* |
| bounding-box aspect, #491 Wild type, right leg I, huntsman spider | 1.5391 ratio | 1.455 | longer / shorter side of the decoded silhouette's bounding box |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- pandas: 2.3.3
- scikit-image: 0.26.0
- matplotlib: 3.10.7
- Pillow: 12.0.0

Input: LimbformDB 2.1.0 (limbformDB_2.1.0.ldb, SQLite, 897,024 bytes, SHA-256 9c26bf39…099f0), downloaded by the script into the git-ignored .limbform_cache/ and hash-checked on every run. The reference figure (Regeneration 1:37 Fig. 1, PMC4339036, CC BY, SHA-256 b26e25c1…f1ba2) is fetched and pinned by morphoeval/measure_outputs.py the same way.
