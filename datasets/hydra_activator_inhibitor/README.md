# Gierer & Meinhardt (1972) *Hydra vulgaris* Activator-Inhibitor Reference Dataset

Primary reference dataset and biological specimen metadata for `1972-gierer-hydra-activator-inhibitor.py` (Slot 001 of the Infinite Morphospace & MorphoEval Campaign).

## Governing Equations — Gierer & Meinhardt (1972) *Kybernetik* 12:30–39, Eqs. (15a)–(15b)

$$\frac{\partial a}{\partial t} = \rho_0 \rho(x) + \frac{c \rho(x) a^2}{h} - \mu_a a + D_a \frac{\partial^2 a}{\partial x^2}$$

$$\frac{\partial h}{\partial t} = c' \rho(x) a^2 - \mu_h h + D_h \frac{\partial^2 h}{\partial x^2}$$

where:
- $a(x,t)$: short-range autocatalytic head/tentacle activator concentration
- $h(x,t)$: long-range lateral inhibitor concentration
- $\rho(x)$: morphogen source density along the *Hydra* apico-basal body column
- $\mu_a, \mu_h$: degradation rates ($\mu_h > \mu_a$ required for homogeneous stability)
- $D_a, D_h$: diffusion coefficients ($D_h / D_a > d_c = (\sqrt{2}+1)^2 \approx 5.8284$ required for Turing-Gierer-Meinhardt instability)

## Files & SHA-256 Provenance

| File | Description | SHA-256 |
|---|---|---|
| `hydra_vulgaris_specimen.jpg` | Dark-field whole-mount photograph of *Hydra vulgaris* (Wikimedia Commons `File:Hydravulgaris.jpg`, CC BY-SA 3.0, 640×614 px) showing basal disc, tubular body column, apical hypostome, and 6 radial tentacles | `007d127bcfa8501d3b9737ffaf7a512efec37db8a0c83a5334568b513da214f9` |
| `scout_manifest.json` | Step 1 OpenAlex & Wikimedia Commons scout dossier (`morphoeval/scout_literature.py --slot 1`) | `b57d0d923283daaf442c82adfb0b709dc507e23ec0d05ae8ea4e9926782e6478` |
| `paper_citations.bib` | BibTeX entries for Gierer & Meinhardt (1972), Meinhardt & Gierer (1974), Meinhardt (1993), and Meinhardt (2012) | `49b5c0a5531a00fe76fed6d969e9e0af0c9c4ec23499de11893523405f080da6` |
| `hydra_activator_inhibitor_reference.json` | Canonical parameters, exact Jacobian & critical diffusivity ratio invariants, regeneration/tentacle ring benchmarks (code-parity values, Pass 3), the `three_system_cylindrical_shell` block, and SHA-256 locks | verified at runtime (digest recorded in the audit measurements) |

## Pass 3 (2026-10-19) — Coupled Three-System Model on a Specimen-Shaped 3D Shell

The Pass 2 audit asked for the 1D column and ring to be extended to a coupled 3D shell with separate head,
tentacle and foot activator-inhibitor loops (Meinhardt 1993, 2012). The reduced version implemented in
`1972-gierer-hydra-activator-inhibitor.py` (§2b of its docstring) and pinned in the
`three_system_cylindrical_shell` block of the reference JSON is:

- **Geometry from the specimen.** `measure_specimen_body_profile()` walks the PCA body axis of the segmented
  photograph to the hypostome tip and the basal disc, samples the Euclidean distance transform to obtain the body
  radius profile $r(s)$, and locates the tentacle attachment ring (where the continuous hypostome disc fragments
  into separate arms). Specimen references used as bands: ring offset $0.070$ of body length, half-thickness
  $0.081$, inter-tentacle spacing $0.084$ of body length (all measured on a 2D projection of a tilted animal).
- **Operator.** Laplace–Beltrami operator of the surface of revolution on a $90 \times 64$ finite-difference grid
  (exact Fourier step in $\theta$, backward-Euler tridiagonal step in $s$). This is a finite-difference shell,
  not a finite-element mesh; FEM and tissue remodelling remain open items.
- **Three thresholded loops + source density.** Head $(a,h)$, tentacle $(b,c)$ and foot $(f,g)$
  Gierer–Meinhardt loops with basal inhibitor production (so each has a firing threshold
  $w_c = (\mu/2)\sqrt{\rho_{inh}/(c\,\mu_{inh}\,\rho_{act})} = 0.1633$), tentacle drive
  $H_t(\sigma)/(1+\kappa_a a)$, foot drive $1/(1+(\sigma/\sigma_f)^2)$, and a source density
  $\sigma$ initialised to $\exp(-4s/L)$ and raised slowly by the head activator. Functional forms and
  coefficients are this implementation's reduced choices, not digitised from the papers.
- **Scale calibration (declared).** $L = 90$ cells was chosen so that the model's intrinsic tentacle spacing
  ($\approx 8.1$ cells) matches the specimen's spacing / body length; the tentacle *count* (7 vs 6) is therefore a
  geometric consequence, while the ring axial offset ($0.111$ vs $0.070 \pm 0.081$), the azimuthally uniform
  hypostome cap, the basal foot and the head-before-ring regeneration order are uncalibrated predictions.
- **Lean 4.** Four new theorems in `morphoeval/lean/loops/Loop_001_hydra_activator_inhibitor.lean`
  (firing-threshold discriminant antitone in the drive + integer instance, tentacle-drive monotonicity,
  head–foot source-density exclusion, shell-Laplacian conservation of constants) are registered in
  `morphoeval/math_kg_loops/gierer-hydra-activator-inhibitor.yaml`.
