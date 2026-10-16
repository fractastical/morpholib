# MorphoEval report — audit `2026-10-14-morphology-audit`

Rubric **0.1.0** · checkpoint `161f269` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are 0–100 (emitted as `pk:hasProbability` = score/100 on the LLM assessment of the dimension hypothesis); Overall is the rubric-weighted mean and the claim's `pk:weightOfEvidence`. Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Faithfulness | Reproducibility | **Overall** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PlanformDB planarian morphology animation (`2013-lobo-planform-morphology.py`) | 85 | 60 | 75 | 45 | 70 | 75 | 60 | 66.1 | 67.3 | **66.5** | mostly faithful |
| Raup shell coiling morphospace (`create_raup_animations.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Hoffmann-Donoughe insect wing venation (`2018-hoffmann-wing-venation.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Chindapol flow-driven coral growth (`2013-chindapol-coral-flow.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Gray-Scott reaction-diffusion patterns (`1952-turing-morpho.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Petroff stromatolite morphospace (`2010-petroff-stromatolite.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Kahng coral geometry morphospace (`2024-kahng-coral-geometry.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |
| Cervera-Levin-Mafe bioelectric morphospace (`2021-Cervera–Levin–Mafe.py`) | — | — | — | — | — | — | — | — | — | *unevaluated* | — |

## Findings and remediation

### PlanformDB planarian morphology animation  —  overall 66.5 (mostly faithful)

Rewritten. The region graph of every PlanformDB morphology is now decoded exactly (polar radii per region, centres and borders from the RegionsLink labels, organs from the organ table) and drawn in Planform's own cartoon convention; wild type, double head, headless, cyclops and the other canonical phenotypes are visually distinct and recognisable as Planform drawings. Remaining gap: the formalism stores three radii per head, so species features such as auricles or a triangular head are not in the data and the renders still read as cartoons, not specimens.

- **Model fidelity to source: 85** — Region shape parameters are interpreted with the formalism's own rule (0 links -> 4 radii right/anterior/left/posterior; 1 link -> +90/+180/+270 deg from the edge; k >= 2 links -> one radius per bisector of consecutive edges), region centres are placed from the RegionsLink Dist/Ang labels and borders at Ratio*Dist, and the rule is checked at load time against all 1181 regions in the database. The only element not taken from the source is how consecutive control radii are joined: a superellipse arc (n = 2.2 free, n = 7 at a border) chosen to reproduce Planform's Fig. 1D; the paper does not publish Planform's interpolant.  
  evidence: `2013-lobo-planform-morphology.py` L20–34; `2013-lobo-planform-morphology.py` L286–314; `2013-lobo-planform-morphology.py` L248–272; `2013-lobo-planform-morphology.py` L701–715; `2013-lobo-planform-morphology.py` L317–355
- **Morphological realism / specimen match: 60** — Side by side with Planform's Fig. 1D (Biology Open) the wild type now has the same structure: a straight-sided body with flat head/trunk and trunk/tail borders, rounded caps, red head fading to pink, blue tail, oval eyes with pupils, green brain lobes and nerve cords, a tube pharynx. Double head (#24), headless (#20), cyclops, side heads and polarity-reversed morphologies are told apart at a glance; 167 distinct body silhouettes among 369 morphologies, the rest (115 of them wild-type bodied) differing only in organs, which is how the database encodes them. Against a photographed S. mediterranea the match is weaker: no auricles, no triangular head, no pigmentation; these are not encoded in the database.  
  evidence: `planform_morphology_gallery.png` L0–0; `2013-lobo-planform-morphology.py` L78–87; `2013-lobo-planform-morphology.py` L460–510
- **Parameter expressivity: 75** — Every encoded quantity now has a visible effect: each radius moves one control point of its region, each link moves a region centre and a border, each organ vector/rotation places one organ. All morphologies in a frame share one scale (frame_bounds), so the size and proportion differences between phenotypes are preserved. The final morphospace grid of the timeline fits each morphology to its cell, which deliberately trades scale for legibility.  
  evidence: `2013-lobo-planform-morphology.py` L525–531; `2013-lobo-planform-morphology.py` L546–565; `2013-lobo-planform-morphology.py` L572–681; `planform_morphospace_timeline.gif` L0–0
- **Morphometric validation: 45** — The script computes silhouette morphometrics (AP length, ML width, aspect, area, region and organ counts, length relative to wild type) for every morphology and can write them as CSV, and the measurements sidecar checks the decoded wild type against the radii encoded in the database (560 x 120 units). Nothing is yet compared with measurements of real animals or with values reported in the literature; the reference is the formalism, not a specimen.  
  evidence: `2013-lobo-planform-morphology.py` L386–412; `2013-lobo-planform-morphology.py` L684–698; `2013-lobo-planform-morphology.py` L713–714
- **Source and data traceability: 70** — Both source papers are cited with DOIs in the module docstring, the database version is named and the schema tables used are listed. The database itself is still an external file located through PLANFORM_DB_PATH; no hash or download URL is recorded, so the exact input cannot be verified from the repository alone.  
  evidence: `2013-lobo-planform-morphology.py` L10–16; `2013-lobo-planform-morphology.py` L138–146
- **Code-to-model traceability: 75** — The decoding rule, the coordinate convention (+y anterior, wild-type head->trunk angle 3pi/2) and the cartoon convention are documented where they are implemented, and region_outline() names the figure it reproduces. The radii-count rule was established by checking the database rather than quoted from a numbered equation, because the papers describe the formalism in prose.  
  evidence: `2013-lobo-planform-morphology.py` L35–38; `2013-lobo-planform-morphology.py` L321–331; `2013-lobo-planform-morphology.py` L211–244
- **Reproducibility package: 60** — Deterministic (no random state), invocation documented in the docstring and README, outputs reproduced identically on re-run, and morphologies with no encoded regions are skipped with a message rather than failing. Dependencies (numpy, scipy, matplotlib, Pillow) are in requirements.txt with lower bounds only, and the run depends on an unversioned external database.  
  evidence: `2013-lobo-planform-morphology.py` L40–43; `2013-lobo-planform-morphology.py` L196–202; `requirements.txt` L1–7

Remediation:
- Record the sha256 and version of the PlanformDB file used, and its download location.
- Add per-species outline templates (auricles, head shape) layered on the decoded regions, clearly marked as not from the database.
- Compare the decoded aspect ratio and organ positions with measurements on photographed S. mediterranea / D. japonica.
- Pin dependency versions for the Python 3.14 environment used here.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### PlanformDB planarian morphology animation
- `planform_morphology_gallery.png` — sha256 `77a7dfa087bf8944…`, 443,915 B, 1920×2160 px
- `planform_morphospace_timeline.gif` — sha256 `f7c8cadf3c06c835…`, 3,265,519 B, 1500×780 px, 63 frames

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type silhouette AP length | 557.6 Planform units | 560 | union of decoded region outlines of Morphology 1, rasterised; bbox extent along the AP axis |
| wild-type silhouette ML width | 119.8 Planform units | 120 | as above, across the ML axis |
| wild-type AP/ML aspect | 4.653 ratio | 4.667 | AP length / ML width of the decoded silhouette *(S. mediterranea in Lobo et al. 2013 Fig. 1A is of this order; the formalism, not a specimen, is the reference)* |
| morphologies rendered | 369 count | — | Morphology rows with at least one encoded region *(372 rows in the database; 3 have no regions (Lethal, Midbody Buldge, Double head, side tail))* |
| distinct body silhouettes | 167 count | 369 | unique (AP length, ML width, area to 100 units, #heads, #tails) tuples over all rendered morphologies *(115 morphologies share the wild-type body and differ only in organs (ectopic eyes, doubled CNS, pharynx position, ...), which is how the database encodes them)* |
| distinct encodings (regions + links + organs) | 369 count | 369 | unique tuples of every encoded number read from the database |
| IoU wild type vs double head (#24) | 0.8884 ratio | < 1; the two bodies differ only at the posterior cap (tail -> second head), the rest of the difference is organs | intersection-over-union of the two decoded silhouettes on a shared canvas (region centres aligned by layout) |
| IoU wild type vs headless (#20) | 0.6116 ratio | ≈ (trunk+tail area)/(whole-body area) if the head is simply absent | as above |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- matplotlib: 3.10.7
- Pillow: 12.0.0

Input database: planformDB_2.5.0.edb as distributed with Planform (Lobo et al. 2013); the script reads it read-only through sqlite3. Previous checkpoint for this simulation: f3fe478 (audit 2026-10-13).
