# A "Big Pixel" Walkthrough of the Ca²⁺ Wave-Vector Approach

This document is a deliberately tiny, fully transparent version of our
Ca²⁺ imaging analysis. The real pipeline runs on millions of native camera
pixels per frame, which makes the intermediate numbers impossible to read off
a figure. Here we coarsen each frame into **"big pixels"** (here 30 px blocks,
giving a 90 × 13 grid so the embryos are ~13 big pixels tall) so that *every
step is legible* while the grid is still fine enough to show the wave's shape.

The figures below are computed on a **real embryo video**, `WGD84`
(`Layer 4 – signaling patterns in neighboring embryo / No physical contact /
Tail-head`): two *Xenopus* embryos side by side, the left one wounded. The
wound flares with Ca²⁺ and, a couple of seconds later, the **neighbor embryo
lights up** — an inter-embryo injury wave (cf. Tung et al. 2024).

Every calculation is the **same** one used in production, just at coarse
resolution:

| Step | What we compute | Production script |
|------|-----------------|-------------------|
| 0 | Imaging frames `F(x, y, t)` | (raw TIFF stacks) |
| 1 | Coarsen to big pixels | (this demo) |
| 2 | Linear brightness scale `[0,1]` | `generate_pixel_brightness_vectors.py` |
| 3 | Baseline `F₀` per location | `score_landmark_responses.py`, `generate_xy_dff_report.py` |
| 4 | Activation `ΔF/F₀` | same |
| 5 | Motion vectors (t−1 → t) + propagation field | `generate_pixel_brightness_vectors.py` |
| 6 | Wave fronts + propagation speed | `rollup_pixel_vectors_to_waves.py` |

### Reproducing the figures

The real-embryo figures (in `output_wgd84/`) were produced with:

```bash
python wave-vector-analysis/bigpixel_illustration/bigpixel_walkthrough.py \
    --tiff "/path/to/WGD84_C-oHTHT-pT-Zoom5.60.tif" \
    --big-px 30 --frame-start 0 --frame-stride 6 --max-frames 18 \
    --poke-frame 1 --fps 2.0 --dff-thresh 0.3 --arrow-scale 3.0 \
    --output-dir wave-vector-analysis/bigpixel_illustration/output_wgd84
```

A **self-contained synthetic** version (no Box dependency, written to
`output/`) runs with no arguments and is handy as a sanity check:

```bash
python wave-vector-analysis/bigpixel_illustration/bigpixel_walkthrough.py
```

> The real videos live in Box (`Calcium videos/…`), so only the synthetic run
> is reproducible without that data. `--big-px` controls the grid density;
> smaller blocks = more big pixels = finer vectors.

---

## Step 0 — What the microscope gives us

A time-lapse stack is just a stack of grayscale frames:

- `F(x, y, t)` = fluorescence intensity at native pixel `(x, y)` in frame `t`.
- `t = 0` is pinned to the **poke / injury** frame, so `time_s = (t − poke_frame) / fps`.

For a big multipage stack we read only a sparse temporal window
(`--frame-start / --frame-stride / --max-frames`) so the wave actually moves
between sampled frames, and we auto-crop the dark border to the tissue.

---

## Step 1 — Big pixels (coarsening)

We tile each frame into `B × B` blocks (here `B = 30` px) and replace each
block by the **mean** intensity of its native pixels:

```
bigpixel[J, I] = mean of all native F(x, y) inside block (J, I)
```

After auto-cropping to the tissue, the WGD84 frame (2723 × 404 px) becomes a
90 × 13 grid — fine enough that each embryo is ~13 big pixels tall, so a wave
front is a visible 2-D shape rather than a single row. This is the only thing
that differs from production — it just lowers the resolution so we can see what
is happening. Each big pixel still has its own intensity time course `F_bp(t)`.

![Raw frame to big pixels](output_wgd84/fig1_raw_to_bigpixels.png)

*Left: one raw frame — the bright blob on the left embryo is the wound site.
Right: the same frame as 30 px big pixels.*

---

## Step 2 — One linear brightness scale

So that brightness is comparable **across pixels and across frames**, raw
intensity is mapped onto a single global linear scale using one stack-wide
range `[g_min, g_max]`:

```
linear(F) = clip( (F − g_min) / (g_max − g_min), 0, 1 )
```

A dim frame stays dim and a bright frame stays bright — we do *not* re-stretch
each frame independently. (This is `linearize()` in
`generate_pixel_brightness_vectors.py`.)

---

## Step 3 — Baseline F₀ (per big pixel)

Each big pixel has a resting fluorescence. We define its baseline `F₀` as a
**low percentile of its own time course** (the 20th percentile, matching
`score_landmark_responses.py`):

```
F₀[J, I] = percentile_20( F_bp(t) over all t )
```

Using a low percentile (rather than the minimum) makes `F₀` robust to noise.

![F0 baseline](output_wgd84/fig2_f0_baseline.png)

*Left: the F₀ map — both embryos are clearly resolved. The red square marks a
probe big pixel (auto-selected as the clearest rise-and-fall). Right: that
probe's intensity time course `F(t)`; the dashed line is its `F₀`, the dotted
line is the poke. The cell sits near rest, then **rises as the wave arrives**
(peaking ~1.5 s after the poke) and decays — exactly the response `F₀`/`ΔF/F₀`
is built to quantify.*

---

## Step 4 — Activation ΔF/F₀

The normalized **activation** of a big pixel is the fractional rise above its
own baseline:

```
ΔF/F₀ (J, I, t) = ( F_bp(J, I, t) − F₀[J, I] ) / F₀[J, I]
```

`ΔF/F₀ = 0` means "at rest"; a big pixel counts as **active** when
`ΔF/F₀ ≥ threshold` (here `0.3`).

![dF/F0 maps](output_wgd84/fig3_dff_maps.png)

*Three snapshots. At +1.0 s the wounded (left) embryo is active. By +2.5 s the
**neighbor (right) embryo is lighting up**, and by +4.0 s the activation has
spread further into it. Green outlines = above threshold.*

---

## Step 5 — Motion vectors and the propagation field

To turn "activation that moves" into **vectors** we use the production rule:
for each bright big pixel in frame `t`, link it to the **nearest bright big
pixel in the previous frame** `t−1` (within a max link distance; brightest
matched first; each previous pixel used once). For a matched pair:

```
Δx = x_curr − x_last ,  Δy = y_curr − y_last
displacement = √(Δx² + Δy²) ,  speed = displacement / Δt
angle = atan2(−Δy, Δx)   (0° = →, 90° = ↑)
```

**Two honest views of the same data, one per slide:**

- **(a) per-frame nearest-match vectors.** At native resolution these are the
  short per-frame steps the production tool draws (and magnifies with
  `--arrow-scale`). At big-pixel resolution most bright cells brighten *in
  place* and match themselves (zero-length), so only a few cells move per
  frame — which is genuinely what the raw data does.

![Per-frame motion vectors](output_wgd84/fig4a_vectors_perframe.png)

- **(b) the propagation-direction field.** The motion accumulates into a
  coherent wave. Coloring each big pixel by its **arrival time** (first frame
  it crosses the threshold) and taking the spatial gradient gives, at every
  cell, the direction the wave travelled (early → late) — arrows flowing from
  the wound out across the neighbor embryo.

![Propagation-direction field](output_wgd84/fig4b_propagation_field.png)

---

## Step 6 — Grouping bright pixels into fronts, then a tracked wave

Finally we go from individual vectors to a **wave**, in two sub-steps that
mirror `rollup_pixel_vectors_to_waves.py`:

1. **Group** the bright big pixels of each frame into **connected components** —
   each connected blob of active cells is one wave *front*.
2. **Track** each front's **brightness-weighted centroid** across frames:

```
x_front(t) = Σ (ΔF/F₀ · x) / Σ ΔF/F₀
y_front(t) = Σ (ΔF/F₀ · y) / Σ ΔF/F₀
```

Consecutive centroids give the **propagation speed and direction**:

```
propagation_speed(t) = distance( front(t), front(t−1) ) / Δt
direction(t)         = atan2( −Δy_front, Δx_front )
```

*Three slides:*

*(a) **Grouping**: in one frame the active big pixels are clustered into
connected fronts (each color a separate front; ★ = its centroid) — this is how
a scatter of bright pixels becomes a discrete "wave".*

![Grouping into fronts](output_wgd84/fig5a_grouping.png)

*(b) The front-centroid path over time, colored by time: it starts at the left
embryo's wound (x ≈ 900) and sweeps **rightward** across the field (to
x ≈ 1740) as the neighbor responds.*

![Front-centroid path](output_wgd84/fig5b_centroid_path.png)

*(c) Propagation speed over time — the front accelerates to ~350–420 px/s as it
crosses toward the neighbor.*

![Propagation speed](output_wgd84/fig5c_prop_speed.png)

The pipeline keeps two distinct speeds:

- **front speed** = how fast individual pixels move frame-to-frame (Step 5).
- **propagation speed** = how fast the *front centroid* advances (Step 6).

---

## Worked numbers

The run also writes [`output_wgd84/bigpixel_numbers.txt`](output_wgd84/bigpixel_numbers.txt).
For the probe big pixel in Step 3:

```
F₀ (20th percentile of F(t)) ≈ 1322
peak F                        ≈ 2775   (at +1.5 s, i.e. wave arrival)
peak ΔF/F₀ = (peak F − F₀)/F₀ = (2775 − 1322) / 1322 ≈ 1.10
```

and the wave-front centroid sweeps from x ≈ 900 px (wound) to x ≈ 1740 px at a
mean propagation speed of ≈ 175 px/s — quantifying the injury-wave's direction
and how fast it travels toward the neighbor embryo.

---

## Why this mirrors the real analysis

| Concept here | Where it lives in production |
|--------------|------------------------------|
| big-pixel mean | (pedagogical only; production uses native pixels) |
| `linear(F)` global scale | `linearize()` in `generate_pixel_brightness_vectors.py` |
| `F₀` = low percentile of the time course | `score_landmark_responses.py` (`np.nanpercentile(c, 20)`) |
| `ΔF/F₀ = (F − F₀)/F₀` | `score_landmark_responses.py`, `generate_xy_dff_report.py` |
| nearest-neighbour t→t−1 vectors | `match_current_to_last_frame()` |
| arrival-time / propagation field | the wave view behind `analyze_pixel_waves.py` |
| connected-component front grouping | `build_frame_fronts()` in `rollup_pixel_vectors_to_waves.py` |
| front centroid + propagation speed | `summarize_waves()` in `rollup_pixel_vectors_to_waves.py` |

The big-pixel version is for explaining and sanity-checking the method; the
production scripts remain the source of truth for real measurements.

---

## Files in this folder

- `bigpixel_walkthrough.py` — the script that generates everything here.
- `output_wgd84/` — the **real embryo** figures (`fig1…fig5`), `bigpixel_walkthrough.pdf`, `bigpixel_numbers.txt`.
- `output/` — the self-contained **synthetic** version (no Box dependency).

Each page of `bigpixel_walkthrough.pdf` carries a **production code
cross-reference** panel under the figure, citing the precise line(s) in the
production pipeline that implement that step (the vector page breaks the
construction into its individual steps). The line numbers are read live at
build time, so they stay accurate even if the source files are edited, making
each figure easy to verify against the real code.

### Useful options

- `--big-px` — block size; smaller = more big pixels = finer vectors.
- `--frame-start / --frame-stride / --max-frames` — pick the active temporal window.
- `--poke-frame / --fps` — define `t = 0` and the time axis.
- `--dff-thresh` — activation cutoff for a big pixel to count as "bright".
- `--arrow-scale` — magnify the small per-frame vectors (Step 5a).
- `--no-autocrop` — keep the full frame instead of cropping to tissue.
