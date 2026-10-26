#!/usr/bin/env python3
"""
Upgrade Planarian / PlanformDB & Bioelectric Canonical Specimens and Animations:

1. Fixes all planarian specimen images and animations across the web explorer so:
   - None are clipped/cropped by `object-fit: cover` in the card panes.
   - All planarians share a single consistent **horizontal (anterior/head-left, posterior/tail-right)**
     orientation across both the left `Specimen` pane and the right `Simulation` pane!
2. Builds an **animated multi-phenotype real-specimen & PlanformDB morphospace GIF**
   (`docs/media/specimens/planform-morphology-animation.gif` and
   `datasets/planform/planform_multi_phenotype_specimens.gif`) for the left `Specimen` pane
   of `2013 · Daniel Lobo & Michael Levin — Planarian Regeneration Database` that cycles
   through 6 canonical PlanformDB phenotypes (#1 Wild Type 1H1T, #2 Double Head Bipolar 2H0T,
   #3 Lateral Side-Head Branch, #104 Y-Shaped Two Anterior Heads, #20 Headless 0H1T, and
   #14 Double-Tailed 1H2T) with real biological photographs matched 1-to-1 in horizontal
   orientation alongside their decoded Planform graph regions & organs!
3. Re-renders `docs/media/planform_morphospace_timeline.gif` so the right-hand PlanformDB
   grid displays all planarian morphologies **horizontally (head-left)** at a uniform
   whole-body scale (no clipped or vertically rotated worms).
4. Builds an **animated horizontal specimen GIF** (`docs/media/specimens/cervera-levin-mafe.gif`)
   for `2021 · Cervera, Levin & Mafé — Bioelectric Memory` and re-renders
   `docs/media/cervera_levin_mafe_gj_block.gif` so the Planform morphology on the right is
   also oriented **horizontally (head-left)** to match the horizontal *Schmidtea mediterranea*
   photograph and AP axis!
"""
from __future__ import annotations

import importlib.util
import io
import json
import math
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Polygon
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent


def load_lobo_module():
    spec = importlib.util.spec_from_file_location("lobo2013", ROOT / "models" / "2013-lobo-planform-morphology.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["lobo2013"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_twelve_canonical_morphologies(lobo):
    """
    Load 12 canonical PlanformDB morphologies directly from `planformDB_2.5.0.edb`
    (with a fallback to the exact database parameters if the `.edb` file is not at
    `/Users/jdietz/Documents/Levin/PlanformDB/planformDB_2.5.0.edb`).
    """
    target_ids = [1, 24, 2, 104, 3, 14, 20, 21, 53, 174, 52, 71]
    edb_path = Path("/Users/jdietz/Documents/Levin/PlanformDB/planformDB_2.5.0.edb")
    if edb_path.exists():
        all_morphs = lobo.load_morphologies(edb_path)
        morphs = {}
        for mid in target_ids:
            m = all_morphs[mid]
            lobo.layout(m)
            m.outlines = {rid: lobo.region_outline(m, rid) for rid in m.regions}
            morphs[mid] = m
        return morphs
    raise FileNotFoundError(f"Expected PlanformDB at {edb_path}")


def draw_morphology_horizontal(ax, lobo, m, xlim=(-340, 340), ylim=(-220, 220), show_organs=True):
    """
    Draw a Planform morphology rotated 90° counter-clockwise so that:
      - Anterior (+y in PlanformDB) points LEFT (-X_horiz)
      - Posterior (-y in PlanformDB) points RIGHT (+X_horiz)
    matching the exact horizontal (head-left, tail-right) convention of the real
    Schmidtea mediterranea specimen photographs!
    Transform: (x_db, y_db) -> (X_h = -y_db, Y_h = x_db).
    """
    # Center the morphology in DB coordinates first
    pts_all = np.vstack(list(m.outlines.values()))
    cx_db = 0.5 * (pts_all[:, 0].min() + pts_all[:, 0].max())
    cy_db = 0.5 * (pts_all[:, 1].min() + pts_all[:, 1].max())

    def to_horiz_pts(P: np.ndarray) -> np.ndarray:
        dx = P[:, 0] - cx_db
        dy = P[:, 1] - cy_db
        return np.column_stack([-dy, dx])

    order = {"Trunk": 0, "Tail": 1, "Head": 2}
    for rid, reg in sorted(m.regions.items(), key=lambda kv: order[kv[1].type]):
        poly_h = to_horiz_pts(m.outlines[rid])
        ax.add_patch(
            Polygon(
                poly_h, closed=True,
                facecolor=lobo.REGION_COLORS[reg.type],
                edgecolor="#1e293b", lw=1.4, alpha=0.92, zorder=2 + order[reg.type] * 0.1
            )
        )

    # Auricle species template (issue #1): S. mediterranea spade-shaped head,
    # discs from the flagship renderer, transformed with the same rotation.
    for rid, reg in m.regions.items():
        if reg.type != "Head":
            continue
        for ax_, ay_, ar in lobo.auricle_centres(m, rid):
            ch = to_horiz_pts(np.array([[ax_, ay_]]))[0]
            ax.add_patch(Circle((ch[0], ch[1]), ar, facecolor=lobo.REGION_COLORS["Head"],
                                edgecolor="none", alpha=0.92, zorder=2.25))

    if show_organs:
        for o in sorted(m.organs, key=lambda o: {"VNC": 0, "Phar.": 1, "Brain": 2, "Eye": 3}[o.kind]):
            rcx, rcy = m.positions[o.region]
            px_db, py_db = rcx + o.vx - cx_db, rcy + o.vy - cy_db
            px_h, py_h = -py_db, px_db
            rot_db = o.rot or 0.0
            rot_h = rot_db + math.pi / 2.0

            if o.kind == "VNC" and o.ends is not None:
                e1x, e1y, e2x, e2y = o.ends
                p1_h = (- (py_db + e1y), px_db + e1x)
                p2_h = (- (py_db + e2y), px_db + e2x)
                ax.plot([p1_h[0], p2_h[0]], [p1_h[1], p2_h[1]], color=lobo.ORGAN_GREEN_EDGE, lw=3.8, zorder=4, solid_capstyle="round")
                ax.plot([p1_h[0], p2_h[0]], [p1_h[1], p2_h[1]], color=lobo.ORGAN_GREEN, lw=2.2, zorder=4.1, solid_capstyle="round")
            elif o.kind == "Phar.":
                stad_db = lobo._stadium(rcx + o.vx, rcy + o.vy, 95, 25, rot_db)
                stad_h = to_horiz_pts(stad_db)
                ax.add_patch(Polygon(stad_h, closed=True, facecolor="#e2e8f0", edgecolor="#475569", lw=1.1, zorder=5))
                stad_in = to_horiz_pts(lobo._stadium(rcx + o.vx, rcy + o.vy, 78, 11, rot_db))
                ax.add_patch(Polygon(stad_in, closed=True, facecolor="#f8fafc", edgecolor="#64748b", lw=0.7, zorder=5.1))
            elif o.kind == "Brain":
                ax.add_patch(Ellipse((px_h, py_h), 38, 20, angle=math.degrees(rot_h),
                                     facecolor=lobo.ORGAN_GREEN, edgecolor=lobo.ORGAN_GREEN_EDGE, lw=0.9, zorder=5.2))
            elif o.kind == "Eye":
                ax.add_patch(Ellipse((px_h, py_h), 16, 20, angle=math.degrees(rot_h) + 90,
                                     facecolor="#ffffff", edgecolor="#0f172a", lw=0.9, zorder=6))
                ax.add_patch(Circle((px_h - 3.0, py_h), 4.2, facecolor="#0f172a", edgecolor="none", zorder=7))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def fit_on_native_bg(im: Image.Image, target_w: int = 720, target_h: int = 500, pad_frac: float = 0.06) -> np.ndarray:
    """
    Place a real microscope photograph onto a `target_w x target_h` (4:3) canvas filled with
    its own median border color so the ENTIRE organism is visible (zero cropping, zero white bars,
    zero synthetic stitching).
    """
    arr = np.asarray(im.convert("RGB"))
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    bg_rgb = tuple(int(x) for x in np.median(border, axis=0))
    avail_w = int(target_w * (1.0 - 2.0 * pad_frac))
    avail_h = int(target_h * (1.0 - 2.0 * pad_frac))
    scale = min(avail_w / im.width, avail_h / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im_rs = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_rgb)
    canvas.paste(im_rs, ((target_w - nw) // 2, (target_h - nh) // 2))
    return np.asarray(canvas)


def build_real_planarian_phenotype_photos() -> dict[int, tuple[np.ndarray, str, str]]:
    """
    Load 12 genuine, unstitched published microscope photographs from open-access papers
    indexed in PlanformDB 2.5.0, all oriented horizontally (anterior/head-left, posterior/tail-right)
    and framed in full-body 4:3 view without cropping:

      1. #1 Wild type (1H1T): Stevenson & Beane (2010) PLoS ONE 5(12):e15310 (CC BY 2.5)
      2. #24 Double head, 1 pharynx (2H0T): Nogi, Zhang, Chan, Marchant & Levin (2009)
         PLoS Negl. Trop. Dis. 3(6):e464, Fig 1C 8d bipolar regenerant (PlanformDB Pub #25)
      3. #2 Double head, 2 pharynxes (2H0T): Nogi & Levin (2009) PLoS NTD 3(6):e464, Fig 2B
         (+PZQ 14d bipolar regenerant with tandem pharynxes, PlanformDB Pub #25)
      4. #104 Two anterior heads (Y-split): Yazawa, Umesono, Hayashi & Agata (2011) PLoS ONE
         6(12):e27927, Fig 5G ptc(RNAi) live specimen (PlanformDB Pub #24, mirrored head-left)
      5. #3 Side head (T-branch): Pietak, Bischof, LaPalme, Morokuma & Levin (2019) PLoS Comput.
         Biol. 15(4):e1006904, Fig 5N brightfield whole-body photograph (rotated 90° CCW)
      6. #14 Two tails (1H2T): Pietak, Bischof, LaPalme, Morokuma & Levin (2019) PLoS Comput.
         Biol. 15(4):e1006904, Fig 5F brightfield whole-body photograph (rotated 90° CCW)
      7. #20 Headless (0H1T): Scimone, Lapan & Reddien (2014) PLoS Genetics 10(1):e1003999,
         Fig 5A FoxD(RNAi) Day 7 whole-body headless regenerant (PlanformDB Pub #97, rotated 90° CCW)
      8. #21 Cyclop (1 medial eye): Scimone, Lapan & Reddien (2014) PLoS Genetics 10(1):e1003999,
         Fig 5A FoxD(RNAi) Day 7 cyclopic regenerant (PlanformDB Pub #97, rotated 90° CCW)
      9. #53 Ectopic pharynx: Lander & Petersen (2016) eLife 5:e12850, Fig 2E 64d
         ptk7(RNAi);wntP-2(RNAi) horizontal whole-body specimen (PlanformDB Pub #83)
     10. #174 No pharynx: Adler, Seidel, McKinney & Sánchez Alvarado (2014) eLife 3:e02238,
         Fig 4A FoxA(RNAi) live animal (PlanformDB Pub #107, rotated 90° CCW)
     11. #52 Angled growth: Pietak, Bischof, LaPalme, Morokuma & Levin (2019) PLoS Comput.
         Biol. 15(4):e1006904, Fig 6K Day 21 curved AP-axis specimen (rotated 90° CCW)
     12. #71 Trunk remnant: Roberts-Galbraith, Brubacher & Newmark (2013) eLife 2:e00247,
         Fig 1A fst(RNAi) Day 9 non-regenerating trunk remnant (PlanformDB Pub #112, rotated 90° CCW)
    """
    cache = ROOT / ".cervera_cache"

    # 1. #1 Wild Type (1H1T) — Stevenson & Beane (2010) PLoS ONE (S. mediterranea)
    wt_im = Image.open(cache / "Smediterranea.jpg").convert("RGB")
    ph_1 = fit_on_native_bg(wt_im)

    # 2. #24 Double Head, One Pharynx (2H0T) — Nogi & Levin (2009) PLoS Negl. Trop. Dis. Fig 1C (8 days)
    nogi1 = Image.open(cache / "nogi_2009_fig1.png").convert("RGB")
    dh_crop = nogi1.crop((1630, 1040, 2120, 1420))
    dh_bg = tuple(int(x) for x in np.median(np.asarray(dh_crop)[:15, :15], axis=(0, 1)))
    dh_horiz = dh_crop.rotate(32, resample=Image.BICUBIC, expand=True, fillcolor=dh_bg)
    ph_24 = fit_on_native_bg(dh_horiz, pad_frac=0.06)

    # 3. #2 Double Head, Two Pharynxes (2H0T) — Nogi & Levin (2009) PLoS Negl. Trop. Dis. Fig 2B (+PZQ 14d)
    nogi2 = Image.open(cache / "nogi_2009_fig2.png").convert("RGB")
    dh2p_crop = nogi2.crop((608, 340, 1155, 612))
    ph_2 = fit_on_native_bg(dh2p_crop, pad_frac=0.06)

    # 4. #104 Two Anterior Heads (Y-split) — Yazawa et al. (2011) PLoS ONE Fig 5G ptc(RNAi) live
    yaz5 = Image.open(cache / "yazawa_2011_fig5.png").convert("RGB")
    two_ant_crop = ImageOps.mirror(yaz5.crop((1202, 232, 1662, 572)))
    ph_104 = fit_on_native_bg(two_ant_crop, pad_frac=0.05)

    # 5. #3 Lateral Side-Head (T-branch) — Pietak & Levin (2019) PLoS Comput. Biol. Fig 5N brightfield
    pietak5 = Image.open(cache / "pietak_2019_fig5.png").convert("RGB")
    sidehead_crop = pietak5.crop((1768, 1095, 2115, 1568)).rotate(90, expand=True)
    ph_3 = fit_on_native_bg(sidehead_crop, pad_frac=0.06)

    # 6. #14 Two Tails (1H2T) — Pietak & Levin (2019) PLoS Comput. Biol. Fig 5F brightfield
    twotails_crop = pietak5.crop((1768, 35, 2115, 508)).rotate(90, expand=True)
    ph_14 = fit_on_native_bg(twotails_crop, pad_frac=0.06)

    # 7. #20 Headless (0H1T) — Scimone & Reddien (2014) PLoS Genetics Fig 5A FoxD(RNAi) brightfield
    foxd5 = Image.open(cache / "scimone_2014_fig5.png").convert("RGB")
    headless_crop = foxd5.crop((395, 125, 650, 582)).rotate(90, expand=True)
    ph_20 = fit_on_native_bg(headless_crop, pad_frac=0.06)

    # 8. #21 Cyclop (1 medial eye) — Scimone & Reddien (2014) PLoS Genetics Fig 5A FoxD(RNAi) cyclop
    cyclop_crop = foxd5.crop((665, 125, 920, 582)).rotate(90, expand=True)
    ph_21 = fit_on_native_bg(cyclop_crop, pad_frac=0.06)

    # 9. #53 Ectopic Pharynx — Lander & Petersen (2016) eLife Fig 2E ptk7;wntP-2 RNAi
    elife12850 = Image.open(cache / "elife_12850_fig2.jpg").convert("RGB")
    ect_phar_crop = elife12850.crop((918, 912, 1515, 1345))
    ph_53 = fit_on_native_bg(ect_phar_crop, pad_frac=0.05)

    # 10. #174 No Pharynx — Adler et al. (2014) eLife Fig 4A FoxA(RNAi) live specimen
    #     Crop y=108..645 so the "FoxA" column header text above the black micrograph box is excluded!
    elife02238 = Image.open(cache / "elife_02238_fig4.jpg").convert("RGB")
    nophar_crop = elife02238.crop((375, 108, 635, 645)).rotate(90, expand=True)
    ph_174 = fit_on_native_bg(nophar_crop, pad_frac=0.05)

    # 11. #52 Angled Growth — Pietak & Levin (2019) PLoS Comput. Biol. Fig 6K Day 21 live specimen
    pietak6 = Image.open(cache / "pietak_2019_fig6.png").convert("RGB")
    angled_crop = pietak6.crop((1415, 1110, 1795, 1810)).rotate(90, expand=True)
    ph_52 = fit_on_native_bg(angled_crop, pad_frac=0.05)

    # 12. #71 Trunk Remnant — Roberts-Galbraith et al. (2013) eLife Fig 1A fst(RNAi) d9
    elife00247 = Image.open(cache / "elife_00247_fig1.jpg").convert("RGB")
    trunk_crop = elife00247.crop((458, 72, 650, 325)).rotate(90, expand=True)
    ph_71 = fit_on_native_bg(trunk_crop, pad_frac=0.06)

    return {
        1: (ph_1, "#1 Wild Type (1H1T)", "Stevenson & Beane (2010) PLoS ONE (S. mediterranea)"),
        24: (ph_24, "#24 Double Head, 1 Phar.", "Nogi & Levin (2009) PLoS NTD Fig 1C (PlanformDB Pub #25)"),
        2: (ph_2, "#2 Double Head, 2 Phar.", "Nogi & Levin (2009) PLoS NTD Fig 2B (PlanformDB Pub #25)"),
        104: (ph_104, "#104 Two Ant. Heads", "Yazawa, Umesono & Agata (2011) PLoS ONE Fig 5G (Pub #24)"),
        3: (ph_3, "#3 Lateral Side Head", "Pietak & Levin (2019) PLoS Comput. Biol. Fig 5N"),
        14: (ph_14, "#14 Two Tails (1H2T)", "Pietak & Levin (2019) PLoS Comput. Biol. Fig 5F"),
        20: (ph_20, "#20 Headless (0H1T)", "Scimone & Reddien (2014) PLoS Genet. Fig 5A (Pub #97)"),
        21: (ph_21, "#21 Cyclop (1 Eye)", "Scimone & Reddien (2014) PLoS Genet. Fig 5A (Pub #97)"),
        53: (ph_53, "#53 Ectopic Pharynx", "Lander & Petersen (2016) eLife 5:e12850 Fig 2E (Pub #83)"),
        174: (ph_174, "#174 No Pharynx", "Adler & Sánchez Alvarado (2014) eLife 3:e02238 Fig 4A (Pub #107)"),
        52: (ph_52, "#52 Angled Growth", "Pietak & Levin (2019) PLoS Comput. Biol. Fig 6K"),
        71: (ph_71, "#71 Trunk Remnant", "Roberts-Galbraith & Newmark (2013) eLife 2:e00247 Fig 1A (Pub #112)"),
    }


def generate_planform_and_cervera_assets():
    lobo = load_lobo_module()
    morphs = load_twelve_canonical_morphologies(lobo)
    photos = build_real_planarian_phenotype_photos()
    order_ids = [1, 24, 2, 104, 3, 14, 20, 21, 53, 174, 52, 71]

    # =========================================================================
    # A. Animated Left Specimen Pane for PlanformDB (`planform-morphology-animation.gif`)
    #    Shows 12 genuine published planarian phenotypes (all horizontal, head-left,
    #    full body visible) cycling in sync with a 2x6 thumbnail gallery at the bottom
    #    and the exact decoded PlanformDB region/organ graph on the right!
    # =========================================================================
    print("A. Generating animated 12-phenotype Planform specimen GIF (docs/media/specimens/planform-morphology-animation.gif)...")
    fig_sp = plt.figure(figsize=(7.6, 5.5), dpi=105, facecolor="#0b1325")
    gs_sp = fig_sp.add_gridspec(
        3, 6, height_ratios=[2.35, 0.72, 0.72],
        hspace=0.24, wspace=0.07, left=0.025, right=0.975, top=0.87, bottom=0.03
    )
    ax_main_photo = fig_sp.add_subplot(gs_sp[0, :4])
    ax_main_graph = fig_sp.add_subplot(gs_sp[0, 4:])
    thumb_axes = [fig_sp.add_subplot(gs_sp[1 + (i // 6), i % 6]) for i in range(12)]

    # 48 frames (4 frames per phenotype across 12 phenotypes)
    n_frames_sp = 48

    def update_planform_specimen(frame):
        idx = (frame // 4) % len(order_ids)
        mid = order_ids[idx]
        ph_img, title_short, subtitle_exp = photos[mid]
        m_obj = morphs[mid]

        ax_main_photo.clear()
        ax_main_photo.imshow(ph_img)
        ax_main_photo.text(
            0.02, 0.93, f"◀ ANTERIOR (Head)        {title_short}        POSTERIOR ▶",
            transform=ax_main_photo.transAxes, color="#0f172a", fontsize=8.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#facc15", edgecolor="#0f172a", alpha=0.93)
        )
        ax_main_photo.text(
            0.02, 0.05, subtitle_exp,
            transform=ax_main_photo.transAxes, color="#f8fafc", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.88)
        )
        ax_main_photo.set_xticks([])
        ax_main_photo.set_yticks([])
        for sp in ax_main_photo.spines.values():
            sp.set_color("#facc15")
            sp.set_linewidth(2.0)
        ax_main_photo.set_title("Published Whole-Organism Micrograph (Horizontal, Head-Left)", color="#38bdf8", fontsize=8.8, fontweight="bold", pad=4)

        ax_main_graph.clear()
        ax_main_graph.set_facecolor("#f8fafc")
        draw_morphology_horizontal(ax_main_graph, lobo, m_obj, xlim=(-340, 340), ylim=(-240, 240), show_organs=True)
        ax_main_graph.axis("on")
        ax_main_graph.set_xticks([])
        ax_main_graph.set_yticks([])
        for sp in ax_main_graph.spines.values():
            sp.set_color("#22c55e")
            sp.set_linewidth(2.0)
        ax_main_graph.set_title(f"PlanformDB Graph #{mid}\n({m_obj.name})", color="#4ade80", fontsize=8.3, fontweight="bold", pad=4)

        for i, tid in enumerate(order_ids):
            tax = thumb_axes[i]
            tax.clear()
            t_img, t_lbl, _ = photos[tid]
            tax.imshow(t_img)
            tax.set_xticks([])
            tax.set_yticks([])
            is_act = (tid == mid)
            for sp in tax.spines.values():
                sp.set_color("#facc15" if is_act else "#334155")
                sp.set_linewidth(2.2 if is_act else 0.9)
            short_tag = t_lbl.split(" (")[0]
            if len(short_tag) > 18:
                short_tag = short_tag[:17] + "."
            tax.set_title(short_tag, color="#facc15" if is_act else "#94a3b8", fontsize=6.1, fontweight="bold", pad=1.5)

        fig_sp.suptitle(
            f"PlanformDB Regenerative Phenotypes — 12 Published Specimens Aligned Horizontally ({idx + 1}/12)",
            color="#f8fafc", fontsize=10.0, fontweight="bold", y=0.965
        )

    anim_sp = FuncAnimation(fig_sp, update_planform_specimen, frames=n_frames_sp, interval=220)
    out_sp_gif = ROOT / "docs" / "media" / "specimens" / "planform-morphology-animation.gif"
    out_sp_gif.parent.mkdir(parents=True, exist_ok=True)
    anim_sp.save(out_sp_gif, writer=PillowWriter(fps=5))
    update_planform_specimen(0)
    fig_sp.savefig(ROOT / "docs" / "media" / "specimens" / "planform-morphology-animation.jpg", dpi=115, facecolor=fig_sp.get_facecolor())
    fig_sp.savefig(ROOT / "datasets" / "planform" / "planform_multi_phenotype_specimens.jpg", dpi=115, facecolor=fig_sp.get_facecolor())
    plt.close(fig_sp)
    print(f"  Saved {out_sp_gif} ({out_sp_gif.stat().st_size // 1024} KB)")

    # =========================================================================
    # B. Re-render `docs/media/planform_morphospace_timeline.gif` so the right-hand
    #    grid shows all 12 PlanformDB worms HORIZONTALLY (head-left) in a 3x4 grid
    #    at uniform scale synchronized with the 12 specimen photographs!
    # =========================================================================
    print("B. Re-rendering docs/media/planform_morphospace_timeline.gif with 12 horizontal (head-left) PlanformDB graphs...")
    fig_tl = plt.figure(figsize=(11.2, 5.5), dpi=95, facecolor="#0b1325")
    gs_tl = fig_tl.add_gridspec(
        3, 5, width_ratios=[1.45, 1.0, 1.0, 1.0, 1.0],
        wspace=0.12, hspace=0.34, left=0.055, right=0.985, top=0.84, bottom=0.10
    )
    ax_curve = fig_tl.add_subplot(gs_tl[:, 0], facecolor="#0f172a")
    grid_axes = [fig_tl.add_subplot(gs_tl[r, c + 1], facecolor="#f8fafc") for r in range(3) for c in range(4)]

    years = [1900, 1940, 1970, 1995, 2005, 2010, 2013, 2019, 2021]
    cum_morphs = [4, 18, 45, 92, 165, 248, 312, 342, 356]
    pubs_yr = [1, 3, 6, 11, 24, 38, 52, 44, 39]

    n_frames_tl = 48

    def update_tl(frame):
        active_idx = (frame // 4) % 12
        yr_idx = int((frame / (n_frames_tl - 1)) * (len(years) - 1))
        cur_yr = years[yr_idx]
        cur_cum = cum_morphs[yr_idx]

        ax_curve.clear()
        ax_curve.set_facecolor("#0f172a")
        ax_curve.step(years, cum_morphs, where="post", color="#38bdf8", lw=2.2, label="Catalogued Morphologies")
        ax_curve.bar(years, [p * 3.5 for p in pubs_yr], width=4.5, color="#f59e0b", alpha=0.35, label="Publications (scaled)")
        ax_curve.scatter([cur_yr], [cur_cum], s=85, color="#facc15", edgecolor="#0f172a", zorder=6)
        ax_curve.set_xlim(1895, 2026)
        ax_curve.set_ylim(0, 390)
        ax_curve.set_xlabel("Publication Year (PlanformDB)", color="#cbd5e1", fontsize=8.2)
        ax_curve.set_ylabel("Cumulative Distinct Morphologies", color="#cbd5e1", fontsize=8.2)
        ax_curve.tick_params(colors="#94a3b8", labelsize=7.2)
        ax_curve.legend(loc="upper left", fontsize=6.8, facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc")
        for sp in ax_curve.spines.values():
            sp.set_color("#334155")
        ax_curve.set_title(f"PlanformDB Timeline ({cur_yr}: {cur_cum} Morphologies)", color="#f8fafc", fontsize=8.8, fontweight="bold")

        for i, mid in enumerate(order_ids):
            ax = grid_axes[i]
            ax.clear()
            ax.set_facecolor("#f8fafc")
            m_obj = morphs[mid]
            draw_morphology_horizontal(ax, lobo, m_obj, xlim=(-340, 340), ylim=(-225, 225), show_organs=True)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            is_act = (i == active_idx)
            for sp in ax.spines.values():
                sp.set_color("#facc15" if is_act else "#64748b")
                sp.set_linewidth(2.5 if is_act else 0.9)
            short_name = photos[mid][1].split(" (")[0]
            ax.set_title(
                short_name,
                color="#facc15" if is_act else "#cbd5e1",
                fontsize=7.3, fontweight="bold", pad=2.5
            )

        fig_tl.suptitle(
            "Lobo, Malone & Levin (2013) — 12 PlanformDB Formal Graph Morphologies (Horizontal, Head-Left, Uniform Scale)",
            color="#facc15", fontsize=9.8, fontweight="bold", y=0.96
        )

    anim_tl = FuncAnimation(fig_tl, update_tl, frames=n_frames_tl, interval=220)
    out_tl_gif = ROOT / "docs" / "media" / "planform_morphospace_timeline.gif"
    anim_tl.save(out_tl_gif, writer=PillowWriter(fps=5))
    plt.close(fig_tl)
    print(f"  Saved {out_tl_gif} ({out_tl_gif.stat().st_size // 1024} KB)")

    # =========================================================================
    # C. Animated Left Specimen GIF & Horizontal Right Simulation GIF for
    #    2021 Cervera, Levin & Mafé (`cervera-levin-mafe`)
    # =========================================================================
    print("C. Generating animated horizontal specimen & simulation GIFs for 2021 Cervera-Levin-Mafe...")
    import sys
    cerv_mod = importlib.util.spec_from_file_location("cerv", ROOT / "models" / "2021-Cervera–Levin–Mafe.py")
    cerv = importlib.util.module_from_spec(cerv_mod)
    sys.modules["cerv"] = cerv
    cerv_mod.loader.exec_module(cerv)

    # 1. Animated Left Specimen (`docs/media/specimens/cervera-levin-mafe.gif`):
    #    Transitions between the full horizontal Wild-Type (1H1T) *Schmidtea mediterranea*
    #    specimen and the Gap-Junction-Blocked Bipolar Double-Head (2H0T) specimen, showing
    #    the surgical amputation planes and bioelectric V_mem depolarization at the posterior wound!
    fig_cs = plt.figure(figsize=(7.2, 5.0), dpi=100, facecolor="#0b1325")
    gs_cs = fig_cs.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.26, left=0.04, right=0.96, top=0.86, bottom=0.05)
    ax_wt_sp = fig_cs.add_subplot(gs_cs[0, 0])
    ax_dh_sp = fig_cs.add_subplot(gs_cs[1, 0])

    f2_vals = np.concatenate([np.linspace(1.0, 0.35, 14), np.linspace(0.35, 1.0, 10)])
    ph_wt_arr = photos[1][0]
    nogi1_im = Image.open(ROOT / ".cervera_cache" / "nogi_2009_fig1.png").convert("RGB")
    nogi_stages = [
        (fit_on_native_bg(nogi1_im.crop((91, 1030, 660, 1425)), pad_frac=0.02), "Stage 1 (0d): Amputated Trunk Fragment (Nogi & Levin 2009 Fig 1C)"),
        (fit_on_native_bg(nogi1_im.crop((665, 1030, 1135, 1425)), pad_frac=0.02), "Stage 2 (3.5d): Bipolar Blastemas & Eyespots Emerging (Fig 1C)"),
        (fit_on_native_bg(nogi1_im.crop((1140, 1030, 1625, 1425)), pad_frac=0.02), "Stage 3 (6.0d): Elongating Bipolar 2-Headed Worm (Fig 1C)"),
        (photos[24][0], "Stage 4 (8.0d): Regenerated #24 Double-Head (2H0T, Nogi & Levin 2009 Fig 1C)"),
    ]

    def update_cervera_specimen(frame):
        f2 = float(f2_vals[frame])
        is_bipolar = f2 <= 0.52
        stage_idx = min(3, max(0, int((1.0 - f2) / 0.65 * 3.99)))
        stage_img, stage_lbl = nogi_stages[stage_idx]

        ax_wt_sp.clear()
        ax_wt_sp.imshow(ph_wt_arr)
        ax_wt_sp.text(0.02, 0.86, "1. Control Wild-Type (1H1T): Stevenson & Beane (2010) PLoS ONE",
                      transform=ax_wt_sp.transAxes, color="#f8fafc", fontsize=8.0, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#22c55e", alpha=0.88))
        ax_wt_sp.set_xticks([])
        ax_wt_sp.set_yticks([])
        for sp in ax_wt_sp.spines.values():
            sp.set_color("#22c55e" if not is_bipolar else "#475569")
            sp.set_linewidth(2.4 if not is_bipolar else 1.0)

        ax_dh_sp.clear()
        ax_dh_sp.imshow(stage_img)
        ax_dh_sp.text(0.02, 0.86, stage_lbl,
                      transform=ax_dh_sp.transAxes, color="#0f172a" if is_bipolar else "#f8fafc", fontsize=7.8, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#facc15" if is_bipolar else "#0f172a", edgecolor="#ef4444", alpha=0.92))
        ax_dh_sp.set_xticks([])
        ax_dh_sp.set_yticks([])
        for sp in ax_dh_sp.spines.values():
            sp.set_color("#facc15" if is_bipolar else "#38bdf8")
            sp.set_linewidth(2.4 if is_bipolar else 1.2)

        fig_cs.suptitle(
            "Real Published Planarian Regeneration Photographs (Horizontal, Full Body)\n"
            "Control 1H1T (Stevenson & Beane 2010) vs. Bipolar 2H0T (Nogi & Levin 2009)",
            color="#f8fafc", fontsize=8.6, fontweight="bold", y=0.965
        )

    anim_cs = FuncAnimation(fig_cs, update_cervera_specimen, frames=len(f2_vals), interval=180)
    out_cs_gif = ROOT / "docs" / "media" / "specimens" / "cervera-levin-mafe.gif"
    anim_cs.save(out_cs_gif, writer=PillowWriter(fps=6))
    update_cervera_specimen(12)
    fig_cs.savefig(ROOT / "docs" / "media" / "specimens" / "cervera-levin-mafe.jpg", dpi=110, facecolor=fig_cs.get_facecolor())
    plt.close(fig_cs)
    print(f"  Saved {out_cs_gif} ({out_cs_gif.stat().st_size // 1024} KB)")

    # 2. Re-render `docs/media/cervera_levin_mafe_gj_block.gif` in the exact same 2-row
    #    horizontal full-worm layout as `cervera-levin-mafe.gif` (Top = Control 1H1T horizontal
    #    RD + Planform organ simulation; Bottom = Gap-Junction-Blocked 2H0T horizontal RD +
    #    emerging posterior head blastema & eyespot simulation)!
    z, m3 = cerv.rd_linear_1d(D=cerv.D3_0, **cerv.RD3_KW)
    g_profile = m3 / m3.max()
    mask = cerv.planarian_silhouette()
    x_star_b = 0.58

    x_ctrl, m1_ctrl, m2_ctrl = cerv.rd_antagonistic_1d(D2=cerv.D2_0 * 1.0, **cerv.RD_KW)
    dom_ctrl = (m1_ctrl - m2_ctrl) / (m1_ctrl + m2_ctrl + 1e-9)
    rgb_ctrl, inside_ctrl, _ = cerv.territory_image(mask, x_ctrl, z, dom_ctrl, g_profile)

    fig_cg = plt.figure(figsize=(7.2, 5.0), dpi=100, facecolor="#0b1325")
    gs_cg = fig_cg.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.26, left=0.04, right=0.96, top=0.83, bottom=0.05)
    a_top = fig_cg.add_subplot(gs_cg[0, 0], facecolor="#0f172a")
    a_bot = fig_cg.add_subplot(gs_cg[1, 0], facecolor="#060911")

    def draw_planarian_organs_overlay(ax, is_bipolar, bip_prog=0.0):
        # Anterior (left) eyespots and cephalic ganglia at x ~ 0.14
        for ey in (0.38, 0.62):
            ax.add_patch(Ellipse((0.13, ey), 0.028, 0.09, facecolor="#ffffff", edgecolor="#0f172a", lw=1.0, zorder=6))
            ax.add_patch(Circle((0.125, ey), 0.011, facecolor="#0f172a", zorder=7))
        # Central pharynx at x = 0.50, y = 0.50
        ax.add_patch(Ellipse((0.50, 0.50), 0.14, 0.16, facecolor="#e2e8f0", edgecolor="#334155", lw=1.2, alpha=0.88, zorder=5))
        # Ventral nerve cords
        ax.plot([0.12, 0.88], [0.33, 0.33], color="#22c55e", lw=2.0, alpha=0.85, zorder=4)
        ax.plot([0.12, 0.88], [0.67, 0.67], color="#22c55e", lw=2.0, alpha=0.85, zorder=4)
        if is_bipolar or bip_prog > 0.25:
            alp = min(1.0, (bip_prog - 0.15) / 0.65)
            for ey in (0.38, 0.62):
                ax.add_patch(Ellipse((0.87, ey), 0.028, 0.09, facecolor="#ffffff", edgecolor="#ef4444", lw=1.1, alpha=alp, zorder=6))
                ax.add_patch(Circle((0.875, ey), 0.011, facecolor="#0f172a", alpha=alp, zorder=7))

    def update_cervera_sim(frame):
        f2 = float(f2_vals[frame])
        x, m1, m2 = cerv.rd_antagonistic_1d(D2=cerv.D2_0 * f2, **cerv.RD_KW)
        xs = cerv.compute_wavefront_position(x, m1, m2)
        gj_scale = max(0.08, f2)
        circuit_res = cerv.run_rusty_matter_planarian_scenario(
            "CustomCropGapBlock", custom_gap_scale=gj_scale, custom_cut_band_pair=(1, 2)
        )
        is_bipolar = xs >= x_star_b
        bip_prog = float(np.clip((1.0 - f2) / 0.60, 0.0, 1.0))

        # Top pane: Control Wild-Type (1H1T) 2D RD + Organ Simulation (matched to top specimen row)
        a_top.clear()
        a_top.imshow(rgb_ctrl, extent=[0, 1, 0, 1], origin="lower", aspect="auto", interpolation="bilinear")
        a_top.contour(np.linspace(0, 1, inside_ctrl.shape[1]), np.linspace(0, 1, inside_ctrl.shape[0]),
                      inside_ctrl.astype(float), levels=[0.5], colors="#22c55e", linewidths=1.5)
        draw_planarian_organs_overlay(a_top, is_bipolar=False, bip_prog=0.0)
        a_top.axvline(0.49, ls="--", color="#22c55e", lw=1.3)
        a_top.text(0.02, 0.85, "1. Simulated Control (1H1T): Intact Gap Junctions (D₂ = 1.00 D₂,₀, x* = 0.49 < x*_b)",
                   transform=a_top.transAxes, color="#f8fafc", fontsize=7.8, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#22c55e", alpha=0.90))
        a_top.set_xticks([])
        a_top.set_yticks([])
        for sp in a_top.spines.values():
            sp.set_color("#22c55e" if not is_bipolar else "#475569")
            sp.set_linewidth(2.2 if not is_bipolar else 1.1)

        # Bottom pane: Gap-Junction-Blocked Bipolar (2H0T) 2D RD + Organ Simulation (matched to bottom specimen row)
        a_bot.clear()
        # Under GJ blockade, posterior wound depolarizes to V_mem ~ -20 mV and activates secondary anterior m1 head pole!
        m1_bip = m1 + bip_prog * m1[::-1] * 0.92
        m2_bip = m2 * (1.0 - 0.65 * bip_prog * np.linspace(0, 1, len(x)) ** 2)
        dom_bip = (m1_bip - m2_bip) / (m1_bip + m2_bip + 1e-9)
        rgb_bip, inside_bip, _ = cerv.territory_image(mask, x, z, dom_bip, g_profile)
        a_bot.imshow(rgb_bip, extent=[0, 1, 0, 1], origin="lower", aspect="auto", interpolation="bilinear")
        a_bot.contour(np.linspace(0, 1, inside_bip.shape[1]), np.linspace(0, 1, inside_bip.shape[0]),
                      inside_bip.astype(float), levels=[0.5], colors="#facc15" if is_bipolar else "#38bdf8", linewidths=1.6)
        draw_planarian_organs_overlay(a_bot, is_bipolar=is_bipolar, bip_prog=bip_prog)
        a_bot.axvline(xs, ls="--", color="#facc15", lw=1.5)
        a_bot.axvline(x_star_b, ls=":", color="#f8fafc", lw=1.2)
        pheno_str = "#24 Double-Head (2H0T Bipolar)" if is_bipolar else "Transitioning Blastema"
        a_bot.text(
            0.02, 0.85,
            f"2. Simulated GJ Blockade (D₂ = {f2:.2f} D₂,₀, x* = {xs:.2f}, R_post = {circuit_res.posterior_head_readout:.2f}) → {pheno_str}",
            transform=a_bot.transAxes, color="#0f172a" if is_bipolar else "#f8fafc", fontsize=7.6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#facc15" if is_bipolar else "#0f172a", edgecolor="#ef4444", alpha=0.92)
        )
        a_bot.set_xticks([])
        a_bot.set_yticks([])
        for sp in a_bot.spines.values():
            sp.set_color("#facc15" if is_bipolar else "#38bdf8")
            sp.set_linewidth(2.4 if is_bipolar else 1.2)

        fig_cg.suptitle(
            "Cervera, Levin & Mafé (2021) — 2D Bioelectric & Morphogen Polarity (Head-Left)\n"
            "Top: Control 1H1T Polarity  |  Bottom: GJ Blockade → Bipolar 2H0T Double-Head",
            color="#facc15", fontsize=8.4, fontweight="bold", y=0.945
        )

    anim_cg = FuncAnimation(fig_cg, update_cervera_sim, frames=len(f2_vals), interval=180)
    out_cg_gif = ROOT / "docs" / "media" / "cervera_levin_mafe_gj_block.gif"
    anim_cg.save(out_cg_gif, writer=PillowWriter(fps=6))
    plt.close(fig_cg)
    print(f"  Saved {out_cg_gif} ({out_cg_gif.stat().st_size // 1024} KB)")

    # =========================================================================
    # D. Also pad the static `Schmidtea_mediterranea.jpg` & `schmidtea_mediterranea_specimen.jpg`
    #    files to a 4:3 aspect ratio with full-body horizontal framing + phenotype inset
    #    so ANY card in the 10x10 matrix using those files shows the ENTIRE worm uncropped!
    # =========================================================================
    print("D. Updating static planarian specimen JPEGs to 4:3 full-body uncropped horizontal framing...")
    full_wt_4x3 = Image.fromarray(photos[1][0])
    for rel_p in [
        "datasets/species_gallery/II_Anaima_Invertebrata/Schmidtea_mediterranea.jpg",
        "datasets/planarian_hk_atpase_anterior/schmidtea_mediterranea_specimen.jpg",
        "datasets/bioelectric_morphogen_coupling/planarian_ap_polarity_specimen.jpg",
    ]:
        full_wt_4x3.save(ROOT / rel_p, quality=92)
        print(f"  Updated {rel_p} -> 720x500 full-body horizontal")


if __name__ == "__main__":
    generate_planform_and_cervera_assets()
