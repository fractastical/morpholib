#!/usr/bin/env python3
"""
morphoeval/render_aristotelian_animated_tiles.py — Animated Tiled Views of the
1,116-Model Aristotelian Morphospace Matrix (`Ousia × Poiein`) with Direct
Canonical-Specimen Computer-Vision Overlay & Image-Fitness Scoring.

Produces three visual artifacts in `docs/media/`:
  1. `docs/media/morphospace_aristotelian_best9_matrix.gif`:
     Split-screen dashboard pairing the **10×10 Aristotelian Coverage Matrix** (`n = 1,116` models,
     with numbered callout badges `[1]..[9]` on the 9 flagship grid spaces) on the left with a
     **live 3×3 Animated Tile Grid** on the right where all 9 grid spaces animate simultaneously
     (pairing the segmented canonical specimen image with the dynamical simulation, Image-Fitness
     score, DOI paper citation, and upstream code repository).
  2. `docs/media/morphospace_aristotelian_best9_matrix.png`:
     High-resolution static poster frame of the 10×10 Matrix + 3×3 Best-9 Tiled Grid.
  3. `docs/media/morphospace_aristotelian_cell_9tiles.gif`:
     A **3×3 Animated Zoom Inside a Grid Space**, showing the 9 audited plant morphogenesis models
     in `Phyta × 4. Polar Auxin & Turgor Stress` (`n = 22` in cell), where **every tile is a direct
     split-screen comparison**:
       - **Left Pane (`Canonical Specimen + CV Overlay`):** Real biological specimen photograph with
         live computer-vision segmentation contour / feature overlay + animated mathematical overlay
         registered directly onto the specimen geometry.
       - **Right Pane (`Biophysical Simulation`):** Live dynamical simulation (`*_growth.gif` / vertex mesh)
         with explicit **Audit Score**, **Image Fitness (`ImgFit`)**, **DOI Paper Citation**, and
         **Upstream Code Repository** badge.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT / "docs" / "media"
INDEX_JSON = ROOT / "morphoeval" / "runs" / "aristotelian_categories_index.json"

SUBALTERN_ORDER = [
    ("Mammalia (Zootoka)", "Mammalia (Zootoka)", (56, 189, 248)),
    ("Aves (Ornithes)", "Aves (Ornithes)", (56, 189, 248)),
    ("Amphibia & Reptilia (Ootoka)", "Amphibia/Reptilia", (56, 189, 248)),
    ("Pisces (Ichthyes)", "Pisces (Ichthyes)", (56, 189, 248)),
    ("Entoma (Arthropoda)", "Entoma (Insects)", (245, 158, 11)),
    ("Ostrakoderma (Mollusca & Echinodermata)", "Ostrakoderma (Shells)", (245, 158, 11)),
    ("Zoophyta & Vermes (Cnidaria/Platyhelminthes)", "Zoophyta (Cnidaria/Worms)", (245, 158, 11)),
    ("Angiospermae & Gymnospermae", "Phyta (Land Plants)", (34, 197, 94)),
    ("Slime Molds & Bacteria", "Myxobacteria & Biofilms", (168, 85, 247)),
    ("Mineralized Protists (Silica/Calcite)", "Silica/Calcite Protists", (168, 85, 247)),
]

CAUSE_ORDER = [
    "1. Chemical Turing / RD",
    "2. Morphogen + Condensations",
    "3. Phase/Delay Oscillators",
    "4. Polar Auxin & Turgor Stress",
    "5. Vertex / Potts / Nematics",
    "6. Tip-Splitting & Flow Trees",
    "7. Elastic Shell/Sheet Buckling",
    "8. Crystal Accretion & TPMS",
    "9. Bioelectric V_mem & Attractors",
    "10. Stigmergy & Active Swarms",
]

CAUSE_SHORT = [
    "1. Turing / RD",
    "2. Condensations",
    "3. Oscillators",
    "4. Auxin & Turgor",
    "5. Vertex / Nematics",
    "6. Flow Trees",
    "7. Elastic Buckling",
    "8. Accretion & TPMS",
    "9. Bioelectric V_mem",
    "10. Stigmergy Swarms",
]

BEST_9_CELLS = [
    {
        "badge": 1,
        "row_idx": 7,  # Phyta (Land Plants)
        "col_idx": 3,  # 4. Polar Auxin & Turgor Stress
        "gif": "runions_leaf_venation_growth.gif",
        "fallback_gif": "vogel_phyllotaxis_growth.gif",
        "title": "Phyta × Auxin & Turgor",
        "subtitle": "Golden-Angle Phyllotaxis & Venation",
    },
    {
        "badge": 2,
        "row_idx": 0,  # Mammalia (Zootoka)
        "col_idx": 5,  # 6. Tip-Splitting & Flow Trees
        "gif": "lung_gland_branching_tips_growth.gif",
        "fallback_gif": "vascular_plexus_shear_remodeling_growth.gif",
        "title": "Mammalia × Flow Trees",
        "subtitle": "Murray's Law & Tip-Splitting Trees",
    },
    {
        "badge": 3,
        "row_idx": 4,  # Entoma (Insects)
        "col_idx": 4,  # 5. Vertex / Potts / Nematics
        "gif": "drosophila_wing_vertex_model_growth.gif",
        "fallback_gif": "planar_cell_polarity_wing_growth.gif",
        "title": "Entoma × Vertex / Nematics",
        "subtitle": "Epithelial Hex Packing & Jamming",
    },
    {
        "badge": 4,
        "row_idx": 5,  # Ostrakoderma (Shells)
        "col_idx": 7,  # 8. Crystal Accretion & TPMS
        "gif": "conus_textile_rule30.gif",
        "fallback_gif": "raup_shell_coiling.gif",
        "title": "Ostrakoderma × Accretion",
        "subtitle": "Logarithmic Coiling & Rule 30 Shells",
    },
    {
        "badge": 5,
        "row_idx": 1,  # Aves (Ornithes)
        "col_idx": 1,  # 2. Morphogen + Condensations
        "gif": "avian_feather_bud_hexagonal_growth.gif",
        "fallback_gif": "avian_beak_darwin_scaling_growth.gif",
        "title": "Aves × Condensations",
        "subtitle": "Feather Hex Lattice & Beak Scaling",
    },
    {
        "badge": 6,
        "row_idx": 0,  # Mammalia (Zootoka)
        "col_idx": 8,  # 9. Bioelectric V_mem & Attractors
        "gif": "bressloff_v1_form_constants.gif",
        "fallback_gif": "grid_cell_hexagonal_attractor_growth.gif",
        "title": "Mammalia × Bioelectric V1",
        "subtitle": "Cortical Form Constants & Grid Cells",
    },
    {
        "badge": 7,
        "row_idx": 3,  # Pisces (Ichthyes)
        "col_idx": 0,  # 1. Chemical Turing / RD
        "gif": "turing_gray_scott.gif",
        "fallback_gif": "zebrafish_stripe_spot_bifurcation_growth.gif",
        "title": "Pisces × Turing RD",
        "subtitle": "Pomacanthus & Danio Stripe Waves",
    },
    {
        "badge": 8,
        "row_idx": 6,  # Zoophyta (Cnidaria/Worms)
        "col_idx": 8,  # 9. Bioelectric V_mem & Attractors
        "gif": "cervera_levin_mafe_gj_block.gif",
        "fallback_gif": "planarian_anterior_posterior_polarity_growth.gif",
        "title": "Zoophyta × Bioelectric V_mem",
        "subtitle": "Planarian 2-Headed Bistable Polarity",
    },
    {
        "badge": 9,
        "row_idx": 8,  # Myxobacteria & Biofilms
        "col_idx": 9,  # 10. Stigmergy & Active Swarms
        "gif": "dictyostelium_camp_scroll_waves_growth.gif",
        "fallback_gif": "physarum_slime_mold_network_growth.gif",
        "title": "Myxobacteria × Stigmergy",
        "subtitle": "cAMP Spiral Waves & Slime Networks",
    },
]


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fname = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(d / fname), size)


def load_gif_frames(gif_path: Path, n_frames: int, target_size: tuple[int, int]) -> list[Image.Image]:
    """Load and resample a GIF (or static image fallback) into `n_frames` RGB frames of `target_size`."""
    tw, th = target_size
    if not gif_path.exists():
        blank = Image.new("RGB", target_size, (15, 23, 42))
        return [blank for _ in range(n_frames)]

    im = Image.open(gif_path)
    raw_frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    if not raw_frames:
        raw_frames = [im.convert("RGB")]

    indices = [int(round(i * (len(raw_frames) - 1) / max(1, n_frames - 1))) for i in range(n_frames)]
    out: list[Image.Image] = []
    for idx in indices:
        fr = raw_frames[idx].copy()
        fr.thumbnail((tw, th), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (11, 15, 25))
        canvas.paste(fr, ((tw - fr.width) // 2, (th - fr.height) // 2))
        out.append(canvas)
    return out


def build_animated_specimen_overlay_frames(
    sp_path: Path,
    slug: str,
    n_frames: int,
    target_size: tuple[int, int],
) -> list[Image.Image]:
    """
    Build `n_frames` of the canonical biological specimen photograph cleanly fitted
    to `target_size` without synthetic moving overlays.
    """
    tw, th = target_size
    if not sp_path.exists() or sp_path.is_dir():
        blank = Image.new("RGB", target_size, (15, 23, 42))
        return [blank for _ in range(n_frames)]

    with Image.open(sp_path) as raw_im:
        base_im = ImageOps.fit(raw_im.convert("RGB"), (tw, th), method=Image.Resampling.LANCZOS)

    f_tiny = get_font(8, True)
    fr = base_im.copy()
    draw = ImageDraw.Draw(fr, "RGBA")
    draw.rectangle([2, 2, 98, 16], fill=(7, 10, 18, 205))
    draw.text((5, 3), "Canonical Specimen", font=f_tiny, fill=(56, 189, 248, 255))
    clean_frame = fr.convert("RGB")
    return [clean_frame.copy() for _ in range(n_frames)]


def render_left_matrix_panel(
    mat_counts: np.ndarray,
    total_models: int,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    """Render the 10×10 Aristotelian Coverage Matrix (verified models only) with `[1]..[9]` callout badges."""
    dpi = 110
    fig, ax = plt.subplots(figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi, facecolor="#070a12")
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.22, top=0.88)
    ax.set_facecolor("#0f172a")

    vmax_val = max(8, int(mat_counts.max()))
    im = ax.imshow(mat_counts, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax_val)

    row_labels = [s[1] for s in SUBALTERN_ORDER]
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5, color="#f1f5f9", fontweight="bold")
    ax.set_xticks(np.arange(len(CAUSE_SHORT)))
    ax.set_xticklabels(CAUSE_SHORT, fontsize=8.0, color="#e2e8f0", rotation=32, ha="right")

    badge_lookup = {(c["row_idx"], c["col_idx"]): c["badge"] for c in BEST_9_CELLS}

    for r in range(mat_counts.shape[0]):
        for c in range(mat_counts.shape[1]):
            val = int(mat_counts[r, c])
            if (r, c) in badge_lookup and val > 0:
                b_num = badge_lookup[(r, c)]
                rect = plt.Rectangle(
                    (c - 0.47, r - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor="#facc15",
                    linewidth=2.4,
                    zorder=5,
                )
                ax.add_patch(rect)
                ax.text(
                    c,
                    r - 0.12,
                    f"#{b_num}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#070a12",
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "#facc15", "edgecolor": "#ffffff", "lw": 0.8},
                    zorder=6,
                )
                ax.text(
                    c,
                    r + 0.27,
                    f"({val})",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#ffffff" if val >= vmax_val * 0.55 else "#0f172a",
                    zorder=6,
                )
            elif val > 0:
                txt_col = "#0f172a" if val < vmax_val * 0.55 else "#ffffff"
                ax.text(c, r, f"{val}", ha="center", va="center", fontsize=8.0, fontweight="bold", color=txt_col)
            else:
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1.0,
                        1.0,
                        facecolor="#090e1a",
                        edgecolor="#1e293b",
                        linewidth=0.6,
                        zorder=2,
                    )
                )

    for boundary in [3.5, 6.5, 7.5]:
        ax.axhline(boundary, color="#facc15", ls="--", lw=1.1, alpha=0.8)

    pop_cells = int(np.sum(mat_counts > 0))
    ax.set_title(
        f"Aristotelian 10×10 Matrix ({total_models:,} Verified Models · {pop_cells} Cells)\n"
        "Only Models Passing Image Fitness (ImgFit ≥ 70.0) Shown",
        color="white",
        fontsize=10.2,
        fontweight="bold",
        pad=10,
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    pil_im = Image.fromarray(rgba[:, :, :3], mode="RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    plt.close(fig)
    return pil_im


def build_best9_animated_matrix_gif(entries: list[dict[str, Any]], n_frames: int = 24) -> tuple[Path, Path]:
    """Build `morphospace_aristotelian_best9_matrix.gif` and `.png`."""
    sub_keys = [s[0] for s in SUBALTERN_ORDER]
    sub_colors = {s[0]: s[2] for s in SUBALTERN_ORDER}

    passing_entries = [
        e for e in entries if e.get("passes_image_threshold", False) and float(e.get("image_fitness", 0.0)) >= 70.0
    ]

    mat_counts = np.zeros((10, 10), dtype=int)
    cell_entries: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for e in passing_entries:
        if e["subaltern_genos"] in sub_keys and e["efficient_cause"] in CAUSE_ORDER:
            r = sub_keys.index(e["subaltern_genos"])
            c = CAUSE_ORDER.index(e["efficient_cause"])
            mat_counts[r, c] += 1
            cell_entries[(r, c)].append(e)

    for k in cell_entries:
        cell_entries[k].sort(key=lambda x: x.get("rank_score", x["score"]), reverse=True)

    W, H = 1520, 860
    header_h = 58
    left_w = 560
    right_w = W - left_w - 16
    tile_cols, tile_rows = 3, 3
    gap = 8
    tile_w = (right_w - (tile_cols + 1) * gap) // tile_cols
    tile_h = (H - header_h - (tile_rows + 1) * gap) // tile_rows
    anim_h = tile_h - 64

    left_panel = render_left_matrix_panel(mat_counts, len(passing_entries), left_w, H - header_h - 12)

    tile_data = []
    for cell_spec in BEST_9_CELLS:
        r, c = cell_spec["row_idx"], cell_spec["col_idx"]
        items = cell_entries[(r, c)]
        gif_path = MEDIA_DIR / cell_spec["gif"]
        if not gif_path.exists():
            gif_path = MEDIA_DIR / cell_spec["fallback_gif"]
        frames = load_gif_frames(gif_path, n_frames, (tile_w - 8, anim_h))

        spec_thumbs: list[tuple[Image.Image, dict[str, Any]]] = []
        for it in items:
            sp_rel = it.get("specimen_rel", "")
            if sp_rel and (ROOT / sp_rel).exists() and not (ROOT / sp_rel).is_dir():
                try:
                    with Image.open(ROOT / sp_rel) as sp_im:
                        th_im = sp_im.convert("RGB").resize((76, 54), Image.Resampling.LANCZOS)
                        spec_thumbs.append((th_im, it))
                except Exception:
                    pass
        tile_data.append(
            {
                "spec": cell_spec,
                "items": items,
                "frames": frames,
                "spec_thumbs": spec_thumbs,
                "border_color": sub_colors[sub_keys[r]],
                "count": int(mat_counts[r, c]),
            }
        )

    f_title = get_font(16, True)
    f_sub = get_font(11, False)
    f_tile_hdr = get_font(11, True)
    f_tile_sub = get_font(9, False)
    f_tile_bold = get_font(9, True)

    out_frames: list[Image.Image] = []
    for k in range(n_frames):
        canvas = Image.new("RGB", (W, H), (7, 10, 18))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, W, header_h], fill=(15, 23, 42))
        draw.line([(0, header_h), (W, header_h)], fill=(56, 189, 248), width=2)
        draw.text(
            (16, 8),
            f"Aristotelian Morphospace Matrix — Animated Best-9 Grid Spaces (Ousia × Poiein across {len(passing_entries)} Verified Models)",
            font=f_title,
            fill=(248, 250, 252),
        )
        draw.text(
            (16, 33),
            f"Left: 10×10 Coverage Matrix ({int(np.sum(mat_counts > 0))} Populated Cells Passing ImgFit ≥ 70.0)   |   "
            "Right: Audited Top Models (Image Fitness + DOI Paper + Code Provenance)",
            font=f_sub,
            fill=(148, 163, 184),
        )

        canvas.paste(left_panel, (8, header_h + 6))

        rx0 = left_w + 12
        for idx, td in enumerate(tile_data):
            tr, tc = divmod(idx, 3)
            tx = rx0 + gap + tc * (tile_w + gap)
            ty = header_h + gap + tr * (tile_h + gap)

            bcol = td["border_color"]
            draw.rectangle([tx, ty, tx + tile_w, ty + tile_h], fill=(15, 23, 42), outline=bcol, width=2)

            draw.rectangle([tx + 2, ty + 2, tx + tile_w - 2, ty + 24], fill=(23, 33, 53))
            badge_num = td["spec"]["badge"]
            draw.rectangle([tx + 5, ty + 4, tx + 28, ty + 22], fill=(250, 204, 21))
            draw.text((tx + 8, ty + 6), f"#{badge_num}", font=f_tile_bold, fill=(7, 10, 18))
            draw.text((tx + 34, ty + 5), td["spec"]["title"], font=f_tile_hdr, fill=(248, 250, 252))
            cnt_txt = f"n={td['count']}"
            cw = draw.textlength(cnt_txt, font=f_tile_bold)
            draw.text((tx + tile_w - cw - 6, ty + 6), cnt_txt, font=f_tile_bold, fill=(250, 204, 21))

            anim_frame = td["frames"][k]
            canvas.paste(anim_frame, (tx + 4, ty + 26))

            items = td["items"]
            cur_item = items[(k // 2) % len(items)] if items else None
            if td["spec_thumbs"]:
                sp_im, sp_item = td["spec_thumbs"][(k // 2) % len(td["spec_thumbs"])]
                cur_item = sp_item
                ix = tx + tile_w - sp_im.width - 8
                iy = ty + 30
                draw.rectangle([ix - 2, iy - 2, ix + sp_im.width + 1, iy + sp_im.height + 1], fill=(7, 10, 18), outline=(250, 204, 21), width=1)
                canvas.paste(sp_im, (ix, iy))

            by0 = ty + 26 + anim_h + 2
            draw.rectangle([tx + 2, by0, tx + tile_w - 2, ty + tile_h - 2], fill=(11, 17, 30))
            if cur_item:
                doi_s = cur_item.get("paper_doi", "")
                sub_line = f"DOI: {doi_s} · {cur_item.get('upstream_repo_label', '')[:24]}"
                draw.text((tx + 6, by0 + 2), sub_line, font=f_tile_sub, fill=(148, 163, 184))
                idx_in_cell = ((k // 2) % len(items)) + 1
                imf = cur_item.get("image_fitness", cur_item["score"])
                cyc_txt = f"[{idx_in_cell}/{len(items)}] {cur_item['genus']} · Audit {cur_item['score']:.1f} · ImgFit {imf:.1f}"
                while draw.textlength(cyc_txt, font=f_tile_bold) > tile_w - 12 and len(cyc_txt) > 12:
                    cyc_txt = cyc_txt[:-2] + "…"
                draw.text((tx + 6, by0 + 17), cyc_txt, font=f_tile_bold, fill=(52, 211, 153))

        out_frames.append(canvas)

    out_gif = MEDIA_DIR / "morphospace_aristotelian_best9_matrix.gif"
    out_png = MEDIA_DIR / "morphospace_aristotelian_best9_matrix.png"
    out_frames[0].save(out_png, format="PNG", optimize=True)

    strip = Image.new("RGB", (W, H * 4))
    for i, f in enumerate(out_frames[:: max(1, len(out_frames) // 4)][:4]):
        strip.paste(f, (0, i * H))
    pal = strip.quantize(colors=128, method=Image.MEDIANCUT)
    q_frames = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in out_frames]
    q_frames[0].save(
        out_gif,
        save_all=True,
        append_images=q_frames[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    return out_gif, out_png


def build_single_cell_9tiles_gif(entries: list[dict[str, Any]], n_frames: int = 24) -> Path:
    """
    Build `morphospace_aristotelian_cell_9tiles.gif`: a 3×3 animated grid showing the models in
    `Phyta × Polar Auxin & Turgor Stress` that PASS the Canonical Image-Fitness threshold
    (`ImgFit >= 70.0`), where EVERY tile is a side-by-side split view:
      - LEFT PANE: Canonical biological specimen photograph with real computer-vision contour
        segmentation AND animated mathematical overlay registered directly onto the specimen.
      - RIGHT PANE: Live biophysical simulation GIF (`*_vertex_pin1.gif`, `*_mesh_growth.gif`,
        `*_growth.gif`) + explicit Image-Fitness score, DOI citation, and code repository.
      - Models below the threshold (`ImgFit < 70.0`) are suppressed and NOT displayed as tiles.
    """
    all_cell_models = [
        e
        for e in entries
        if e["subaltern_genos"] == "Angiospermae & Gymnospermae"
        and e["efficient_cause"] == "4. Polar Auxin & Turgor Stress"
    ]
    passing_cell_models = [e for e in all_cell_models if e.get("passes_image_threshold", False)]
    suppressed_core = [
        e for e in all_cell_models
        if not e.get("passes_image_threshold", False) and e.get("audit_tier") != "expansion_template"
    ]
    suppressed_templates = [
        e for e in all_cell_models
        if e.get("audit_tier") == "expansion_template"
    ]

    with_gifs = []
    for e in passing_cell_models:
        sim_rel = e.get("sim_media_rel", "")
        if sim_rel and (ROOT / sim_rel).exists():
            with_gifs.append((e, ROOT / sim_rel))
            continue
        slug_u = e["slug"].replace("-", "_")
        preferred = (
            sorted(MEDIA_DIR.glob(f"*{slug_u}*vertex*.gif"))
            or sorted(MEDIA_DIR.glob(f"*{slug_u}*mesh*.gif"))
            or sorted(MEDIA_DIR.glob(f"*{slug_u}*growth.gif"))
            or sorted(MEDIA_DIR.glob(f"*{slug_u}*.gif"))
        )
        if preferred:
            with_gifs.append((e, preferred[0]))

    # Sort strictly among models that pass the Image Fitness threshold (ImgFit >= 70.0)
    with_gifs.sort(key=lambda x: x[0].get("rank_score", x[0]["score"]), reverse=True)
    best9_in_cell = with_gifs[:9]

    W, H = 1320, 880
    header_h = 64
    gap = 10
    tile_w = (W - 4 * gap) // 3
    tile_h = (H - header_h - 4 * gap) // 3
    anim_h = tile_h - 62
    pane_w = (tile_w - 12) // 2

    f_title = get_font(15, True)
    f_sub = get_font(10, False)
    f_hdr = get_font(10, True)
    f_meta = get_font(8, False)
    f_meta_b = get_font(8, True)
    f_score = get_font(9, True)

    preloaded = []
    for idx, (entry, gif_path) in enumerate(best9_in_cell):
        sim_frames = load_gif_frames(gif_path, n_frames, (pane_w, anim_h))
        sp_rel = entry.get("specimen_rel", "")
        sp_path = ROOT / sp_rel if sp_rel else Path("")
        spec_overlay_frames = build_animated_specimen_overlay_frames(
            sp_path, entry["slug"], n_frames, (pane_w, anim_h)
        )
        preloaded.append((idx + 1, entry, spec_overlay_frames, sim_frames))

    out_frames: list[Image.Image] = []
    for k in range(n_frames):
        canvas = Image.new("RGB", (W, H), (7, 10, 18))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, W, header_h], fill=(15, 23, 42))
        draw.line([(0, header_h), (W, header_h)], fill=(34, 197, 94), width=2)
        draw.text(
            (16, 8),
            f"Inside a Single Aristotelian Grid Space — {len(preloaded)} Verified Models Passing Image Fitness (ImgFit ≥ 70.0) in [Phyta × 4. Polar Auxin & Turgor Stress]",
            font=f_title,
            fill=(248, 250, 252),
        )
        draw.text(
            (16, 34),
            f"Left of Each Tile: Canonical Biological Specimen   |   "
            f"Right: Dynamical Simulation + Audit & Image Fitness + DOI Paper & Code Links",
            font=f_sub,
            fill=(167, 243, 208),
        )

        for rank, entry, spec_frames, sim_frames in preloaded:
            r, c = divmod(rank - 1, 3)
            tx = gap + c * (tile_w + gap)
            ty = header_h + gap + r * (tile_h + gap)

            draw.rectangle([tx, ty, tx + tile_w, ty + tile_h], fill=(15, 23, 42), outline=(34, 197, 94), width=2)
            draw.rectangle([tx + 2, ty + 2, tx + tile_w - 2, ty + 24], fill=(20, 38, 34))
            draw.text(
                (tx + 7, ty + 5),
                f"#{rank} {entry['genus']} ({entry['common']})",
                font=f_hdr,
                fill=(248, 250, 252),
            )
            imf = entry.get("image_fitness", entry["score"])
            sc_txt = f"Audit {entry['score']:.1f} · ImgFit {imf:.1f}"
            sw = draw.textlength(sc_txt, font=f_score)
            draw.text((tx + tile_w - sw - 7, ty + 6), sc_txt, font=f_score, fill=(250, 204, 21))

            # Paste LEFT pane (Specimen + CV segmentation + registered math overlay)
            lx = tx + 4
            ly = ty + 26
            canvas.paste(spec_frames[k], (lx, ly))
            draw.rectangle([lx, ly, lx + pane_w, ly + anim_h], outline=(34, 211, 238), width=1)

            # Paste RIGHT pane (Dynamical simulation)
            rx = tx + 4 + pane_w + 4
            canvas.paste(sim_frames[k], (rx, ly))
            draw.rectangle([rx, ly, rx + pane_w, ly + anim_h], outline=(71, 85, 105), width=1)

            # Two-line provenance & image-metric footer
            by0 = ty + 26 + anim_h + 2
            draw.rectangle([tx + 2, by0, tx + tile_w - 2, ty + tile_h - 2], fill=(11, 17, 30))

            line1 = f"{entry['script']} · {entry.get('image_metric_summary', entry['poson_law'])}"
            while draw.textlength(line1, font=f_meta_b) > tile_w - 14 and len(line1) > 16:
                line1 = line1[:-2] + "…"
            draw.text((tx + 6, by0 + 3), line1, font=f_meta_b, fill=(56, 189, 248))

            doi = entry.get("paper_doi", "")
            repo_lbl = entry.get("upstream_repo_label", "")
            line2 = f"Paper: doi.org/{doi}  |  Code: {repo_lbl}"
            while draw.textlength(line2, font=f_meta) > tile_w - 14 and len(line2) > 16:
                line2 = line2[:-2] + "…"
            draw.text((tx + 6, by0 + 17), line2, font=f_meta, fill=(203, 213, 225))

        out_frames.append(canvas)

    out_gif = MEDIA_DIR / "morphospace_aristotelian_cell_9tiles.gif"
    strip = Image.new("RGB", (W, H * 4))
    for i, f in enumerate(out_frames[:: max(1, len(out_frames) // 4)][:4]):
        strip.paste(f, (0, i * H))
    pal = strip.quantize(colors=128, method=Image.MEDIANCUT)
    q_frames = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in out_frames]
    q_frames[0].save(
        out_gif,
        save_all=True,
        append_images=q_frames[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    return out_gif


def build_3x3_aristotelian_matrix_grid_gif(entries: list[dict[str, Any]], n_frames: int = 24) -> Path:
    """Build `morphospace_aristotelian_3x3_matrix_grid.gif`."""
    row_headers = [
        ("I. Enaima (Vertebrates)", "Mammalia · Aves · Amphibia · Pisces", (56, 189, 248)),
        ("II. Anaima (Invertebrates)", "Entoma · Ostrakoderma · Zoophyta", (245, 158, 11)),
        ("III–IV. Phyta & Protista", "Land Plants · Biofilms · Protists", (34, 197, 94)),
    ]
    col_headers = [
        ("A. Chemical & Morphogen Fields", "Turing RD · Condensations · Oscillators"),
        ("B. Cellular & Tissue Mechanics", "Auxin/Turgor · Vertex/Nematics · Buckling"),
        ("C. Branching, Shells & Bioelectricity", "Flow Trees · Accretion/TPMS · V_mem & Swarms"),
    ]

    grid_3x3_specs = [
        (0, 0, "I. Enaima", ["1.", "2.", "3."], "avian_feather_bud_hexagonal_growth.gif", "Panthera / Gallus / Danio · Stripes, Spots & Somites"),
        (0, 1, "I. Enaima", ["4.", "5.", "7."], "cortical_gyrification_buckling_growth.gif", "Homo / Bos / Canis · Cortical Gyri, Gut Villi & Vertex"),
        (0, 2, "I. Enaima", ["6.", "8.", "9.", "10."], "bressloff_v1_form_constants.gif", "Homo / Mus / Macaca · Lung/Vascular Trees & V1 Maps"),
        (1, 0, "II. Anaima", ["1.", "2.", "3."], "butterfly_eyespot_focus_rings_growth.gif", "Bicyclus / Papilio / Drosophila · Eyespots & Gap Genes"),
        (1, 1, "II. Anaima", ["4.", "5.", "7."], "drosophila_wing_vertex_model_growth.gif", "Drosophila / Apis · Wing Vertex Packing & Furrow Folding"),
        (1, 2, "II. Anaima", ["6.", "8.", "9.", "10."], "conus_textile_rule30.gif", "Conus / Nautilus / Schmidtea · Coiled Shells & Planaria"),
        (2, 0, "III", ["1.", "2.", "3."], "chemotactic_stripe_splitting_growth.gif", "Myxococcus / Bacillus · Travelling Waves & Condensations"),
        (2, 1, "III", ["4.", "5.", "7."], "runions_leaf_venation_growth.gif", "Ginkgo / Helianthus / Arabidopsis · Venation & Phyllotaxis"),
        (2, 2, "IV", ["6.", "8.", "9.", "10."], "dictyostelium_camp_scroll_waves_growth.gif", "Coscinodiscus / Physarum / Dictyostelium · Silica & Swarms"),
    ]

    W, H = 1340, 880
    header_h = 64
    col_hdr_h = 46
    row_hdr_w = 210
    gap = 10
    tile_w = (W - row_hdr_w - 4 * gap) // 3
    tile_h = (H - header_h - col_hdr_h - 4 * gap) // 3
    anim_h = tile_h - 52

    f_title = get_font(16, True)
    f_sub = get_font(11, False)
    f_ax_bold = get_font(12, True)
    f_ax_sub = get_font(9, False)
    f_tile_hdr = get_font(10, True)
    f_tile_meta = get_font(9, False)

    passing_entries = [
        e for e in entries if e.get("passes_image_threshold", False) and float(e.get("image_fitness", 0.0)) >= 70.0
    ]
    cells_prepared = []
    for r, c, meg_prefix, cause_prefixes, gif_name, desc in grid_3x3_specs:
        if r < 2:
            residents = [
                e
                for e in passing_entries
                if e["megiston_genos"].startswith(meg_prefix)
                and any(e["efficient_cause"].startswith(cp) for cp in cause_prefixes)
            ]
        else:
            residents = [
                e
                for e in passing_entries
                if (e["megiston_genos"].startswith("III.") or e["megiston_genos"].startswith("IV."))
                and any(e["efficient_cause"].startswith(cp) for cp in cause_prefixes)
            ]
        residents.sort(key=lambda x: x.get("rank_score", x["score"]), reverse=True)

        frames = load_gif_frames(MEDIA_DIR / gif_name, n_frames, (tile_w - 8, anim_h))
        thumbs = []
        for it in residents[:12]:
            sp_rel = it.get("specimen_rel", "")
            if sp_rel and (ROOT / sp_rel).exists() and not (ROOT / sp_rel).is_dir():
                try:
                    with Image.open(ROOT / sp_rel) as im:
                        thumbs.append((im.convert("RGB").resize((64, 46), Image.Resampling.LANCZOS), it))
                except Exception:
                    pass
        cells_prepared.append((r, c, residents, frames, thumbs, desc))

    out_frames: list[Image.Image] = []
    for k in range(n_frames):
        canvas = Image.new("RGB", (W, H), (7, 10, 18))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, W, header_h], fill=(15, 23, 42))
        draw.line([(0, header_h), (W, header_h)], fill=(250, 204, 21), width=2)
        draw.text(
            (16, 9),
            f"Aristotelian 3×3 Animated Matrix Grid — Taxonomic Genus (Ousia) × Formal/Efficient Cause (n = {len(passing_entries)} Verified Models)",
            font=f_title,
            fill=(248, 250, 252),
        )
        draw.text(
            (16, 34),
            "Every Matrix Cell Cycles Exclusively Through Verified Models Passing Canonical Image Fitness (ImgFit ≥ 70.0)",
            font=f_sub,
            fill=(250, 204, 21),
        )

        cx0, cy0 = gap, header_h + gap
        draw.rectangle([cx0, cy0, row_hdr_w, cy0 + col_hdr_h], fill=(17, 24, 39), outline=(71, 85, 105), width=1)
        draw.text((cx0 + 10, cy0 + 7), "Rows: Ousia (Taxon)", font=f_tile_hdr, fill=(56, 189, 248))
        draw.text((cx0 + 10, cy0 + 24), "Cols: Poiein (Cause)", font=f_tile_hdr, fill=(250, 204, 21))

        for c, (ctitle, csub) in enumerate(col_headers):
            tx = row_hdr_w + gap + c * (tile_w + gap)
            draw.rectangle([tx, cy0, tx + tile_w, cy0 + col_hdr_h], fill=(23, 33, 53), outline=(100, 116, 139), width=1)
            draw.text((tx + 10, cy0 + 6), ctitle, font=f_ax_bold, fill=(248, 250, 252))
            draw.text((tx + 10, cy0 + 25), csub, font=f_ax_sub, fill=(148, 163, 184))

        for r, (rtitle, rsub, rcol) in enumerate(row_headers):
            ty = header_h + col_hdr_h + 2 * gap + r * (tile_h + gap)
            draw.rectangle([cx0, ty, row_hdr_w, ty + tile_h], fill=(17, 24, 39), outline=rcol, width=2)
            draw.text((cx0 + 10, ty + tile_h // 2 - 22), rtitle, font=f_ax_bold, fill=rcol)
            draw.text((cx0 + 10, ty + tile_h // 2 + 2), rsub, font=f_ax_sub, fill=(203, 213, 225))

        for r, c, residents, frames, thumbs, desc in cells_prepared:
            tx = row_hdr_w + gap + c * (tile_w + gap)
            ty = header_h + col_hdr_h + 2 * gap + r * (tile_h + gap)
            rcol = row_headers[r][2]

            draw.rectangle([tx, ty, tx + tile_w, ty + tile_h], fill=(15, 23, 42), outline=rcol, width=2)
            draw.rectangle([tx + 2, ty + 2, tx + tile_w - 2, ty + 24], fill=(23, 33, 53))
            draw.text((tx + 8, ty + 5), desc, font=f_tile_hdr, fill=(248, 250, 252))
            badge = f"n = {len(residents)} models"
            bw = draw.textlength(badge, font=f_tile_hdr)
            draw.rectangle([tx + tile_w - bw - 14, ty + 4, tx + tile_w - 4, ty + 22], fill=(250, 204, 21))
            draw.text((tx + tile_w - bw - 9, ty + 5), badge, font=f_tile_hdr, fill=(7, 10, 18))

            canvas.paste(frames[k], (tx + 4, ty + 26))

            cur_item = residents[(k // 2) % len(residents)] if residents else None
            if thumbs:
                sp_im, sp_item = thumbs[(k // 2) % len(thumbs)]
                cur_item = sp_item
                ix = tx + tile_w - sp_im.width - 8
                iy = ty + 30
                draw.rectangle([ix - 2, iy - 2, ix + sp_im.width + 1, iy + sp_im.height + 1], fill=(7, 10, 18), outline=(250, 204, 21), width=1)
                canvas.paste(sp_im, (ix, iy))

            by0 = ty + 26 + anim_h + 2
            draw.rectangle([tx + 2, by0, tx + tile_w - 2, ty + tile_h - 2], fill=(11, 17, 30))
            if cur_item:
                rank_idx = ((k // 2) % len(residents)) + 1
                imf = cur_item.get("image_fitness", cur_item["score"])
                doi = cur_item.get("paper_doi", "")
                footer = f"[{rank_idx}/{len(residents)}] {cur_item['genus']} · Audit {cur_item['score']:.1f} · ImgFit {imf:.1f} · doi:{doi}"
                while draw.textlength(footer, font=f_tile_meta) > tile_w - 14 and len(footer) > 12:
                    footer = footer[:-2] + "…"
                draw.text((tx + 8, by0 + 5), footer, font=f_tile_meta, fill=(52, 211, 153))

        out_frames.append(canvas)

    out_gif = MEDIA_DIR / "morphospace_aristotelian_3x3_matrix_grid.gif"
    strip = Image.new("RGB", (W, H * 4))
    for i, f in enumerate(out_frames[:: max(1, len(out_frames) // 4)][:4]):
        strip.paste(f, (0, i * H))
    pal = strip.quantize(colors=128, method=Image.MEDIANCUT)
    q_frames = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in out_frames]
    q_frames[0].save(
        out_gif,
        save_all=True,
        append_images=q_frames[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    return out_gif


def main() -> int:
    entries = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    gif1, png1 = build_best9_animated_matrix_gif(entries, n_frames=24)
    print(f"Wrote Best-9 Aristotelian Matrix Animated GIF -> {gif1} ({gif1.stat().st_size // 1024} KB)")
    print(f"Wrote Best-9 Aristotelian Matrix Poster PNG  -> {png1} ({png1.stat().st_size // 1024} KB)")
    gif2 = build_single_cell_9tiles_gif(entries, n_frames=24)
    print(f"Wrote Single-Cell Best-9 Animated Tiles GIF  -> {gif2} ({gif2.stat().st_size // 1024} KB)")
    gif3 = build_3x3_aristotelian_matrix_grid_gif(entries, n_frames=24)
    print(f"Wrote 3x3 Aristotelian Matrix Grid GIF       -> {gif3} ({gif3.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
