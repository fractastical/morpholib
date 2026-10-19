# MorphoEval report — audit `2026-10-19-petroff-kahng-second-assessor-audit`

Rubric **0.1.0** · checkpoint `5564750` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasProbability` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for ProbKnow template conformance). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Petroff et al. (2010) Conical Stromatolite Spacing Law & Dome-to-Cone Morphospace (`2010-petroff-stromatolite.py`) | 30 | 22 | 55 | 15 | 45 | 35 | 62 | 38.3 | 19.4 | 49.1 | 34.0 | **34.0** | unfaithful |
| Kahng, Odle & Wakeman (2024) Reef-Coral Colony Geometry & Surface-Area Productivity (`2024-kahng-coral-geometry.py`) | 62 | 30 | 68 | 22 | 68 | 58 | 64 | 64.0 | 27.0 | 63.7 | 49.1 | **49.1** | partially faithful |

## Findings and remediation

### Petroff et al. (2010) Conical Stromatolite Spacing Law & Dome-to-Cone Morphospace  —  overall 34.0 (unfaithful)

Blinded second-assessor pass. The script reproduces one scalar relation from Petroff et al. (2010), ell = alpha*sqrt(D*tau), plots it against tau with two script-chosen alpha values, draws a hexagonal lattice of harvest disks at ell(tau=12 h) ~ 0.66 cm, and prints a verbal 'reading the fossil' panel. The lower half of the figure and the GIF are not Petroff's model: they integrate a Lagrangian front advanced by v*z_hat + lambda*n_hat (attributed in the code to Batchelor et al. 2004), multiplied by an ad hoc crest-weighting w_photic = 0.12 + 0.88*(z/z_max)^2, with Laplacian smoothing, a pinned base, and a triangular seed. Petroff et al.'s actual shape argument is nutrient-flux limited (quasi-steady diffusion field around neighbouring cones, mat growth set by local diffusive flux), and the paper's evidence is measured cone-spacing distributions from modern Yellowstone mats, laboratory cultures, and Proterozoic/Archean fossils; none of those data or that model appear here.

- **Model fidelity to source: 30** — Spacing law diffusion_length (L64-67) is the correct form, but alpha and tau are free knobs set by the script (alpha_field = 1.0 at L179, alpha_lab = 0.30 at L180 described as a least-squares fit with no data or fit in the file), and the text panel states that 0.66 cm records a ~20 h rhythmic metabolism (L238-239) whereas 0.66 cm was computed at tau = 12 h (L193, L207; tau = 20 h gives 0.85 cm). Panels (d-f) and the GIF do not implement Petroff's diffusion-limited growth: they credit Batchelor et al. 2004 (L31-34, L86-98), modify it with an unsourced quadratic crest weight w_photic = 0.12 + 0.88*(z/z_max)^2 (L126-130) and 3-point smoothing (L141-142), and never integrate the docstring's own KPZ equation dh/dt = v + lambda*(1 + 1/2 (dh/dr)^2) (L34, L85, L296).  
  evidence: `2010-petroff-stromatolite.py` L64–67; `2010-petroff-stromatolite.py` L126–142; `2010-petroff-stromatolite.py` L193–240
- **Morphological realism / specimen match: 22** — No specimen photograph, thin-section, or Petroff figure is rendered or overlaid anywhere. The 'DOME' at beta = 0.3 (panel d, GIF frame 0) is a flat-topped trapezoid with vertical walls and a residual central kink inherited from the triangular seed (L152); it does not resemble a domal stromatolite. The 'CONE' and 'SHARP CONE' (panels e, f) are parabolic bullets sitting on a fixed-width cylinder because the feet are pinned at +-1 (L136-137), with a rounded apex rather than straight-flanked Conophyton cones. Partial credit for axial crest lamina thickening at large beta and visual distinction across the three regimes.  
  evidence: `petroff_stromatolite_morphospace.png` L0–0; `docs/media/petroff_stromatolite_dome_to_cone.gif` L0–0; `2010-petroff-stromatolite.py` L136–154
- **Parameter expressivity: 55** — beta monotonically controls height/width (0.63 to 3.03 across beta = 0.2 to 6.0 in the GIF sweep at L300-334 and regimes at L249-253), and tau monotonically controls ell (L187-192). However, apex sharpness and flank slope are constrained by the pinned footprint (L136-137) and smooth (L141-142, L146); lambda is fixed at 1 throughout (L248, L305); and seed_height, base_half, smooth, and t_total (L146-147) are hidden parameters with no CLI exposure.  
  evidence: `2010-petroff-stromatolite.py` L187–192; `2010-petroff-stromatolite.py` L249–253; `2010-petroff-stromatolite.py` L300–334
- **Morphometric validation: 15** — Only one morphometric is computed, aspect_ratio (L165-171), which the docstring notes is used only to annotate the figure, and it is compared with nothing measured. The shaded 0.5-1.5 cm band (L198) stands in for the paper's measured spacing distributions without data, and no lamina-thickness, apex-angle, or crest-zone-width metric is computed.  
  evidence: `2010-petroff-stromatolite.py` L165–171; `2010-petroff-stromatolite.py` L198–198
- **Source and data traceability: 45** — Paper cited with journal, volume, pages, and PMC ID (L5-8), but no DOI is given in the file. Batchelor et al. 2004 (L32) and Fejes Toth 1948 (L27) are cited by author/year only without title, venue, or DOI; 'Bakal Formation' and 'eddy diffusivity D_eff ~ Re*D' (L241-243) are asserted without reference; and 'alpha_lab = 0.30 least-squares fit' (L180) cites no data table.  
  evidence: `2010-petroff-stromatolite.py` L5–8; `2010-petroff-stromatolite.py` L27–35; `2010-petroff-stromatolite.py` L180–243
- **Code-to-model traceability: 35** — Equation and figure tags ('Petroff Eq. 2' at L21, L66; 'Petroff, Fig. 4' at L73) are present, but the growth model (L86-163) is traced to a different paper than the one in the filename/title, w_photic (L130) and smooth (L141-142) trace to no equation at all, and the equation printed to the console (L296) differs from the Lagrangian rule actually integrated.  
  evidence: `2010-petroff-stromatolite.py` L85–143; `2010-petroff-stromatolite.py` L296–297
- **Reproducibility package: 62** — Deterministic (no RNG), backed by requirements.txt and requirements.lock.txt, with CLI flags documented at L336-348. However, --gif is omitted from the 'Run it' docstring block (L45-48), show=True is the default (L344), GIF capture via fig.canvas.buffer_rgba() (L329) is font/backend dependent, and there is no self-test.  
  evidence: `2010-petroff-stromatolite.py` L45–48; `2010-petroff-stromatolite.py` L328–348

Remediation:
- 2010-petroff-stromatolite.py:86-163 — Solve nabla^2 c = 0 around a periodic array of absorbing mats and grow the surface at a rate proportional to local diffusive flux dc/dn, rather than substituting a modified Batchelor front with ad-hoc w_photic (L126-130).
- 2010-petroff-stromatolite.py:193,207,238-239 — Make tau = 12 h (ell = 0.66 cm) and the '~20 h rhythmic metabolism' text panel consistent, and vendor measured modern/lab/fossil cone-spacing distributions and a Conophyton thin-section outline in datasets/petroff/.
- 2010-petroff-stromatolite.py:136-137 — Release the pinned basal footprint so beta < 1 produces a convex domal stromatolite rather than a vertical-walled trapezoid.

### Kahng, Odle & Wakeman (2024) Reef-Coral Colony Geometry & Surface-Area Productivity  —  overall 49.1 (partially faithful)

Blinded second-assessor pass. The script encodes Kahng, Odle & Wakeman's (2024) accounting rule — skeleton volume grows by G*S/rho per step — and derives, for four idealised shapes (hemisphere, fixed-aspect cone, fixed-thickness disc, fixed-radius cylinder), the one-step linear extension and surface-area productivity (dS/S). The algebra for V(t+1) -> size(t+1) is correct for every shape, and the qualitative conclusions match the paper. However, the script's annual forward-Euler step makes hemisphere extension look size-dependent (0.52 cm/yr at r = 0.5 cm rising to 1.25 cm/yr), whereas the continuous geometry gives C = G/rho at every size; the 'equal G, rho' comparison applies G = 1.5 g cm^-2 yr^-1 to a 0.5 cm plate (yielding 26 cm/yr extension at r = 30 cm and S ~ 10^9 cm^2 after 15 years in the GIF); and no measured coral growth data are compared with the model output.

- **Model fidelity to source: 62** — The master rule V(t+1) = V(t) + G*S(t)/rho (L15-16, L64-67) and the four shape derivations (hemisphere L68-78, disc L81-89, cone L92-104, cylinder L107-110) are algebraically exact for a 1-year discrete step. However, the discrete annual step introduces a time-step artifact presented as geometry: hemisphere_extension (L70) and panel (c) (L173-182) show C rising from 0.52 cm/yr at r = 0.5 cm toward a plateau G/rho = 1.25 cm/yr, whereas in the continuous model dV/dt = 2*pi*r^2*dr/dt = G*(2*pi*r^2)/rho implies dr/dt = G/rho identically for all r (and the discrete disc extension is ~30% below continuum at G/(rho*h) = 2.5). In addition, upper-surface-only disc S (L82), baseless cone S (L93-94), and single-cylinder branching (L107-110) are not explicitly stated as simplifying assumptions.  
  evidence: `2024-kahng-coral-geometry.py` L68–110; `2024-kahng-coral-geometry.py` L172–185
- **Morphological realism / specimen match: 30** — Output consists of analytical line plots plus four schematic 2D matplotlib patches (_draw_glyphs at L116-147; Wedge and Polygon in the GIF at L294-296), with no coral colony photograph, 3D mesh, or measured colony outline. The glyphs are shape-distinct and labelled with V and S expressions (L135-140), though three rectangles (L128-130) are a schematic stand-in for branching Acropora.  
  evidence: `kahng_coral_geometry_morphospace.png` L0–0; `docs/media/kahng_coral_geometry_growth.gif` L0–0; `2024-kahng-coral-geometry.py` L116–147
- **Parameter expressivity: 68** — Cone aspect ratio alpha monotonically raises cone SAP (panel d, L189-196), branch diameter and plate thickness monotonically lower SAP (panel e, L199-211), and colony size monotonically lowers hemisphere/cone SAP (panel b). However, applying the COMMON parameter set (G=1.50, rho=1.20, L60) to a 0.5 cm disc gives SAP = 2.5 yr^-1 and extension 26 cm/yr at r = 30 cm (panel c), which dominates the vertical axis and compresses the hemisphere and cone curves; panel (e) conflates branch diameter and plate thickness on a single x-axis (L201-207); and step size dt is not exposed.  
  evidence: `2024-kahng-coral-geometry.py` L60–60; `2024-kahng-coral-geometry.py` L172–211
- **Morphometric validation: 22** — The script prints G/rho = 1.28 cm/yr for massive Porites (L252), the ~19x microfragmentation SAP ratio between r = 1 cm and r = 25 cm (L231, L263), and Leptoseris SAP = 0.114 yr^-1 (L258), but never compares them against observed coral extension or fragment growth tables (Lough 2008, Hughes 1987, Kahng 2023), and computes no fit, residual, or observed-vs-model validation metric.  
  evidence: `2024-kahng-coral-geometry.py` L215–236; `2024-kahng-coral-geometry.py` L250–264
- **Source and data traceability: 68** — DOI (10.7717/peerj.17037), PMC ID (PMC10909345), journal, and the authors' GitHub repository are all given in the header (L5-9). However, the empirical parameter sources ('Lough 2008', 'Hughes 1987', 'Kahng 2023' at L57-59) are cited by author-year only without DOIs, table numbers, or page numbers, and no dataset files or SHA-256 digests are present.  
  evidence: `2024-kahng-coral-geometry.py` L5–9; `2024-kahng-coral-geometry.py` L57–60
- **Code-to-model traceability: 58** — Every function docstring states V, S, and the resulting formula (L68-110), and the module header table (L25-30) maps shapes to scaling predictions and cites Appendix A / Results (L66), Fig. 6 (L60, L156), and Fig. 3 (L187). Deduct points because no specific equation numbers from Appendix A are given and the annual forward-Euler discretization is not traced to a specific equation in the paper.  
  evidence: `2024-kahng-coral-geometry.py` L25–31; `2024-kahng-coral-geometry.py` L64–110
- **Reproducibility package: 64** — Deterministic (no RNG), backed by requirements.lock.txt, with CLI flags at L320-331. However, --gif (L326) is omitted from the documented 'Run it' block (L38-41), show=True is the default (L327), GIF rendering via buffer_rgba (L312) is backend/font dependent, Pillow deduplicates the trailing hold frames (L314) from 24 to 16 frames on disk, and there is no self-test.  
  evidence: `2024-kahng-coral-geometry.py` L38–41; `2024-kahng-coral-geometry.py` L312–331

Remediation:
- 2024-kahng-coral-geometry.py:68-110,173-182 — Separate continuous differential rates (hemisphere dr/dt = G/rho identically; cone dh/dt = G*sqrt(1+alpha^2)/rho; disc dr/dt = G*r/(2*rho*h)) from the discrete 1-yr Euler step, and expose dt as a parameter.
- 2024-kahng-coral-geometry.py:60,160-178 — Use biologically realistic DISC parameters in panel (c) so a 0.5 cm plate does not extend at 26 cm/yr and obscure the hemisphere and cone curves.
- 2024-kahng-coral-geometry.py:57-59 — Vendor empirical coral extension/calcification tables from Lough (2008), Hughes (1987), and Kahng (2023) in datasets/kahng/ with full DOIs and SHA-256 digests, and compare model C and SAP against observed colony growth.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Petroff et al. (2010) Conical Stromatolite Spacing Law & Dome-to-Cone Morphospace
- **spec_fingerprint** (semantic model + inputs): `2b3289aae8f3466486ffbc5f503dd7585e6e1c2863ac78c108d1488932cf713f`
- `docs/media/petroff_stromatolite_dome_to_cone.gif` — artifact_sha256 `b7f825720b31f42f…`, 920,055 B, 560×416 px, 70 frames
- `petroff_stromatolite_morphospace.png` — artifact_sha256 `8e00cc6fabe5cf24…`, 328,721 B, 1736×1244 px

| measurement | value | reference | method |
|---|---:|---|---|
| field cone spacing ell at tau = 12 h (D = 1e-5 cm^2/s, alpha = 1.0) | 0.6573 cm | 1.0 | diffusion_length(12 * 3600, D=1e-5, alpha=1.0) = alpha * sqrt(D * tau) *(lab clumps (alpha=0.30) ell(12 h) = 0.1972 cm; hex_lattice min NN distance = 0.6573 cm)* |
| dome regime (beta = 0.3) final lamina height/width aspect ratio | 0.6692 height/width | — | aspect_ratio(grow_stromatolite(v=0.3, lam=1.0)[-1]) |
| cone regime (beta = 1.5) final lamina height/width aspect ratio | 1.1696 height/width | — | aspect_ratio(grow_stromatolite(v=1.5, lam=1.0)[-1]) |
| sharp cone regime (beta = 5.0) final lamina height/width aspect ratio | 2.6191 height/width | — | aspect_ratio(grow_stromatolite(v=5.0, lam=1.0)[-1]) *(monotonic aspect-ratio progression across beta=(0.3, 1.5, 5.0): 0.669 < 1.170 < 2.619)* |

### Kahng, Odle & Wakeman (2024) Reef-Coral Colony Geometry & Surface-Area Productivity
- **spec_fingerprint** (semantic model + inputs): `833cc6f3d6f3988fa15a852dd8366b15c68ef56d6fd98635b3250ff241f4746a`
- `docs/media/kahng_coral_geometry_growth.gif` — artifact_sha256 `6d2629a5bb1c8f33…`, 248,080 B, 720×300 px, 16 frames
- `kahng_coral_geometry_morphospace.png` — artifact_sha256 `fa8ce6cd95b493ee…`, 282,553 B, 1774×1267 px

| measurement | value | reference | method |
|---|---:|---|---|
| massive Porites hemisphere linear extension C at r = 30 cm (discrete 1-yr step) | 1.2301 cm/yr | 1.2812 | hemisphere_extension(r=30 cm, G=1.64, rho=1.28) = (r^3 + 3*G*r^2/rho)^(1/3) - r *(at r = 1 cm discrete step gives C = 0.6920 cm/yr due to 1-yr forward-Euler discretization)* |
| massive Porites microfragmentation SAP speedup ratio (r = 1 cm vs r = 25 cm) | 18.6207 SAP(1 cm) / SAP(25 cm) | 18.92 | hemisphere_sap(r=1 cm) / hemisphere_sap(r=25 cm) for G=1.64, rho=1.28 |
| deep-water Leptoseris flat-disc size-independent SAP (h = 0.20 cm) | 0.1142 yr^-1 | 0.1142 | disc_sap(G=0.061, rho=2.67, h=0.20) = G / (rho * h) |
| branching Acropora cylinder size-independent SAP (r = 0.50 cm) | 7.1429 yr^-1 | 7.1429 | cylinder_sap(G=2.50, rho=1.40, r=0.50) = 2 * G / (rho * r) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- matplotlib: 3.10.7
- Pillow: 12.0.0

Strictly blinded second-assessor evaluation of `2010-petroff-stromatolite.py` and `2024-kahng-coral-geometry.py` at commit 5564750 by Claude Fable 5.1 (Anthropic), paired with primary assessor Gemini 4 Argon (Google) in `2026-10-19-petroff-kahng-audit.yaml`.
