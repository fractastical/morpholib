#!/usr/bin/env python3
"""
Generate the archival museum-plate / specimen-photo matching figure for
D'Arcy Thompson's 1917 arthropods
(`docs/thompson_1917_arthropod_museum_plates.png`).

This is the arthropod counterpart of
`thompson/generate_archival_field_plates_figure.py` (the 3x3 fish plates).

Layout
------
Row 1  1917 Fig. 369 silhouettes a–f (Scholtz, Knötel & Baum 2020, Fig. 1 row 3).
Row 2  Matching 3D laser-scanned carapaces (Scholtz et al. 2020, Fig. 2)
       with extracted silhouettes (cyan) and 8 homologous landmarks (gold).
Row 3  19th-century monograph lithographs of the copepods and amphipods
       (Giesbrecht 1892 Taf. 4; Sars 1895 Pls. 52, 69, 2). Thompson gives no
       source for Figs. 367–370, but his outlines match these habitus figures
       in pose, facing and detail. Fetched from Wikimedia Commons (public
       domain) and SHA-256 checked.
Row 4  Thompson's 1917 engravings (Figs. 367–368; Fig. 370) plus AntWeb
       pinned-head photographs (Pheidole soldier caste, Fig. 359 cordate
       geometry) and Antscan (2026) synchrotron micro-CT renders
       (Acanthomyrmex minor / major).
"""

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, label

REPO_ROOT = Path(__file__).resolve().parents[1]
CRAB_DIR = REPO_ROOT / ".thompson_cache" / "crabs"
ANT_DIR = REPO_ROOT / ".thompson_cache" / "antscan"
MONO_DIR = REPO_ROOT / ".thompson_cache" / "arthropod_plates"
OUT_FIG = REPO_ROOT / "docs" / "thompson_1917_arthropod_museum_plates.png"

COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"
MONOGRAPH_PLATES = {
    "giesbrecht1892_taf4_oithona_sapphirina.jpg": (
        COMMONS + "5/5e/Pelagischen_Copepoden_-_Fauna_und_flora_des_golfes_von_Neapel_%2819%29_-_"
        "Von_Wilhelm_Giesbrecht_%281892%29_-_Taf._4_-_BioDivLibrary_page_39312118.jpg",
        "b01aa055e6772c241023ecf590adfd420f286aaa3ff5e6147508ed4ca384bb5a",
    ),
    "sars1895_pl52_harpinia_plumosa.jpg": (
        COMMONS + "7/71/An_account_of_the_Crustacea_of_Norway_%28Pl._52%29_%286812223079%29.jpg",
        "4c70cef5ccff2f9c7015333b166ca0268b54c5445dd826f9f122db8eaa21a5ed",
    ),
    "sars1895_pl69_stegocephalus_inflatus.jpg": (
        COMMONS + "9/96/An_account_of_the_Crustacea_of_Norway_%28Pl._69%29_%286812229921%29.jpg",
        "945524a8840608494ccaef221f6b6b2dc7760ebfe7f6c4d9f2e8a67b924a6b45",
    ),
    "sars1895_pl2_hyperia_galba.jpg": (
        COMMONS + "6/6e/An_account_of_the_Crustacea_of_Norway_%28Pl._2%29_%286812191767%29.jpg",
        "1a8940277010a2788e0bf8785135d14b4413d69e605f685df3ba1e21eba52d46",
    ),
}

# (plate file, crop box in original pixels, panel title)
MONOGRAPH_CROPS = [
    ("giesbrecht1892_taf4_oithona_sapphirina.jpg", (1228, 193, 1522, 479),
     "Oithona nana (Fig. 367 source)\nGiesbrecht 1892, Taf. 4 fig. 8\nNaples monograph lithograph"),
    ("giesbrecht1892_taf4_oithona_sapphirina.jpg", (252, 100, 496, 505),
     "Sapphirina auronitens (Fig. 368)\nGiesbrecht 1892, Taf. 4 fig. 2\nNaples monograph lithograph"),
    ("sars1895_pl52_harpinia_plumosa.jpg", (199, 169, 1054, 596),
     "Harpinia plumosa (Fig. 370.1)\nSars 1895, Crustacea of Norway\nPl. 52 habitus"),
    ("sars1895_pl69_stegocephalus_inflatus.jpg", (348, 189, 1064, 765),
     "Stegocephalus inflatus (Fig. 370.2)\nSars 1895, Crustacea of Norway\nPl. 69 habitus"),
    ("sars1895_pl2_hyperia_galba.jpg", (388, 189, 934, 656),
     "Hyperia galba (Fig. 370.3)\nSars 1895, Crustacea of Norway\nPl. 2 habitus"),
    ("giesbrecht1892_taf4_oithona_sapphirina.jpg", None,
     "Giesbrecht 1892, Taf. 4 (full plate)\nFauna und Flora des Golfes\nvon Neapel 19"),
]

CRAB_SHORT = [
    ("a", "Geryon", "portunoid reference"),
    ("b", "Corystes", "helmet crab"),
    ("c", "Pisa / Scyramathia", r"majoid $f_1<0$"),
    ("d", "Lithodes maja", "anomuran king crab"),
    ("e", "Portunus / Lupa", "swimming crab"),
    ("f", "Leptomithrax", r"majoid $f_1<0$"),
]

# 8 Type-I/II landmarks in Scholtz Fig. 2 row-2 tile coordinates
# (rostrum, L/R orbit, L/R cardiac, L/mid/R posterior pleon).
REAL_LMS_PX = {
    0: [(53, 18), (40, 21), (67, 20), (48, 68), (61, 67), (40, 96), (54, 95), (67, 97)],
    1: [(59, 5), (50, 16), (68, 16), (52, 73), (66, 73), (52, 105), (59, 105), (66, 105)],
    2: [(59, 23), (44, 32), (73, 31), (52, 79), (65, 78), (51, 105), (58, 105), (65, 105)],
    3: [(57, 7), (53, 26), (61, 25), (50, 71), (64, 71), (32, 100), (58, 104), (83, 98)],
    4: [(56, 34), (46, 36), (65, 35), (51, 59), (61, 58), (41, 78), (56, 79), (70, 78)],
    5: [(54, 18), (42, 21), (66, 21), (51, 72), (59, 72), (37, 103), (54, 103), (71, 103)],
}

CYAN = "#06b6d4"
GOLD = "#f59e0b"
INK = "#0f172a"


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def fetch_monograph_plate(name: str) -> Path:
    url, sha = MONOGRAPH_PLATES[name]
    path = MONO_DIR / name
    if not path.exists():
        MONO_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "morpholib/1.0 (research)"})
        for attempt in range(6):
            try:
                path.write_bytes(urllib.request.urlopen(req).read())
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f"Wikimedia rate limit: {url}")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise ValueError(f"SHA-256 mismatch for {name}: {got}")
    return path


def crop_col(arr: np.ndarray, y0: int, y1: int, col: int, n: int = 6, pad: int = 2) -> np.ndarray:
    h, w = arr.shape[:2]
    x0 = int(col * w / n) + pad
    x1 = int((col + 1) * w / n) - pad
    return arr[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]


def largest_mask(flag: np.ndarray) -> np.ndarray:
    closed = binary_closing(flag, iterations=2)
    filled = binary_fill_holes(closed)
    lbl, n = label(filled)
    if n == 0:
        return filled
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == int(sizes.argmax())


def segment_ink_or_gray(rgb: np.ndarray, thr: float = 0.07) -> np.ndarray:
    """Largest non-white body (line drawing, gray 3D render, or brown photo)."""
    ink = 1.0 - rgb.astype(np.float32).mean(axis=2) / 255.0
    return largest_mask(ink > thr)


def segment_ant_head(rgb: np.ndarray) -> np.ndarray:
    """Pinned AntWeb head: dark cuticle on white, drop the scale bar strip."""
    h = rgb.shape[0]
    body = rgb[: int(h * 0.92)]
    ink = 1.0 - body.astype(np.float32).mean(axis=2) / 255.0
    mask = largest_mask(ink > 0.12)
    full = np.zeros(rgb.shape[:2], dtype=bool)
    full[: mask.shape[0]] = mask
    return full


def boundary_xy(mask: np.ndarray, n: int = 360) -> np.ndarray | None:
    if mask.sum() < 40:
        return None
    from scipy.ndimage import binary_erosion

    edge = mask & (~binary_erosion(mask))
    ys, xs = np.where(edge)
    if len(xs) < 12:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    ang = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(ang)
    pts = np.column_stack([xs[order], ys[order]]).astype(float)
    idx = np.linspace(0, len(pts) - 1, n).astype(int)
    return pts[idx]


def map_landmarks_to_photo(lms_px, photo_mask: np.ndarray, sil_shape: tuple[int, int]) -> np.ndarray:
    """Similarity-map Scholtz tile landmarks onto the 3D-scan tile via mask bbox."""
    ys, xs = np.where(photo_mask)
    if len(xs) < 20:
        scale_x = photo_mask.shape[1] / max(1, sil_shape[1])
        scale_y = photo_mask.shape[0] / max(1, sil_shape[0])
        return np.array([[x * scale_x, y * scale_y] for x, y in lms_px], dtype=float)
    px0, px1 = float(xs.min()), float(xs.max())
    py0, py1 = float(ys.min()), float(ys.max())
    sil_h, sil_w = float(sil_shape[0]), float(sil_shape[1])
    out = []
    for x, y in lms_px:
        xn = x / sil_w
        yn = y / sil_h
        out.append([px0 + xn * (px1 - px0), py0 + yn * (py1 - py0)])
    return np.asarray(out, dtype=float)


def show_plate(ax, rgb: np.ndarray, title: str, outline: np.ndarray | None = None,
               landmarks: np.ndarray | None = None, title_size: float = 8.2):
    ax.imshow(rgb)
    if outline is not None:
        ax.plot(outline[:, 0], outline[:, 1], color=CYAN, lw=1.6, solid_capstyle="round")
    if landmarks is not None:
        ax.scatter(
            landmarks[:, 0], landmarks[:, 1],
            s=18, c=GOLD, edgecolors=INK, linewidths=0.4, zorder=5,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=title_size, fontweight="bold", color=INK, pad=5)


def resize_max(rgb: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    scale = max_side / m
    im = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(im)


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, margin: float = 0.08) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return rgb
    h, w = rgb.shape[:2]
    pad_x = int(margin * (xs.max() - xs.min() + 1))
    pad_y = int(margin * (ys.max() - ys.min() + 1))
    x0, x1 = max(0, xs.min() - pad_x), min(w, xs.max() + pad_x + 1)
    y0, y1 = max(0, ys.min() - pad_y), min(h, ys.max() + pad_y + 1)
    return rgb[y0:y1, x0:x1]


def main() -> None:
    fig1 = load_rgb(CRAB_DIR / "scholtz_2020_fig1_thompson_drawings.png")
    fig2 = load_rgb(CRAB_DIR / "scholtz_2020_fig2_real_3d_crabs.png")
    cope = load_rgb(CRAB_DIR / "thompson_1917_fig367_368_copepods.png")
    amph = load_rgb(CRAB_DIR / "thompson_1917_fig370_amphipods.png")
    ph_min = load_rgb(ANT_DIR / "pheidole_aberrans_minor_head.jpg")
    ph_maj = load_rgb(ANT_DIR / "pheidole_aberrans_major_head.jpg")
    ac_min = resize_max(load_rgb(ANT_DIR / "Acanthomyrmex_minor_GAGA-04-06.png"), 1100)
    ac_maj = resize_max(load_rgb(ANT_DIR / "Acanthomyrmex_major_GAGA-04-07.png"), 1100)

    # Fig. 1 row 3 = filled 1917 silhouettes (row 1 line-drawings are too faint at plate scale).
    # Fig. 2 row 1 = 3D photographs; row 2 = landmark tiles used to map gold dots.
    draw_y0, draw_y1 = 268, 354
    photo_y0, photo_y1 = 8, 112
    sil_y0, sil_y1 = 114, 229

    fig = plt.figure(figsize=(20.4, 18.0), dpi=220, facecolor="white")
    gs = gridspec.GridSpec(
        4, 6, figure=fig,
        hspace=0.40, wspace=0.10,
        top=0.905, bottom=0.06, left=0.045, right=0.985,
        height_ratios=[1.0, 1.05, 1.15, 1.18],
    )

    fig.suptitle(
        "Archival Museum Plates & Specimen Photographs for Thompson's 1917 Arthropods\n"
        "Fig. 369 drawings paired with Scholtz et al. (2020) 3D laser scans  ·  "
        "Giesbrecht (1892) & Sars (1895) monograph lithographs  ·  "
        "AntWeb pinned heads & Antscan (2026) synchrotron micro-CT",
        fontsize=13.4, fontweight="bold", color=INK, y=0.978,
    )

    # --- Rows 1–2: crab drawing ↔ 3D scan photo-match ----------------------
    for col, (letter, name, note) in enumerate(CRAB_SHORT):
        drawing = crop_col(fig1, draw_y0, draw_y1, col)
        photo = crop_col(fig2, photo_y0, photo_y1, col)
        sil_tile = crop_col(fig2, sil_y0, sil_y1, col)

        mask = segment_ink_or_gray(photo, thr=0.055)
        outline = boundary_xy(mask)
        lms = map_landmarks_to_photo(REAL_LMS_PX[col], mask, sil_tile.shape[:2])

        ax_d = fig.add_subplot(gs[0, col])
        show_plate(ax_d, drawing, f"{letter}. {name}\n1917 silhouette · {note}", title_size=7.5)

        ax_p = fig.add_subplot(gs[1, col])
        show_plate(
            ax_p, photo,
            f"{letter}. {name}\n3D laser scan · {note}",
            outline=outline, landmarks=lms, title_size=7.5,
        )
        if col == 0:
            ax_p.plot([], [], color=CYAN, lw=1.8, label="Extracted silhouette")
            ax_p.scatter([], [], s=22, c=GOLD, edgecolors=INK, label="8 Scholtz landmarks")
            ax_p.legend(loc="lower left", fontsize=6.2, framealpha=0.92, facecolor="white")

    # --- Row 3: monograph lithographs Thompson redrew --------------------
    for col, (name, box, title) in enumerate(MONOGRAPH_CROPS):
        plate = Image.open(fetch_monograph_plate(name)).convert("RGB")
        if box is not None:
            plate = plate.crop(box)
        ax = fig.add_subplot(gs[2, col])
        show_plate(ax, np.asarray(plate), title, title_size=7.3)

    # --- Row 4: Thompson's engravings, ant caste photographs ---------------
    row4 = [
        (cope, "Figs. 367–368, p. 743\nOithona nana → Sapphirina\n1917 archival engraving",
         None, False),
        (amph, "Fig. 370, p. 746\nHarpinia → Stegocephalus → Hyperia\n1917 archival engraving",
         None, False),
        (ph_min, "Pheidole aberrans minor\nAntWeb pinned-head photograph\nFig. 359 ovate worker",
         segment_ant_head, False),
        (ph_maj, "Pheidole aberrans major / soldier\nAntWeb pinned-head photograph\nFig. 359 cordate lobes",
         segment_ant_head, False),
        (ac_min, "Acanthomyrmex minor\nAntscan 2026 micro-CT\nGAGA-04-06 synchrotron",
         None, True),
        (ac_maj, "Acanthomyrmex major / soldier\nAntscan 2026 micro-CT\nGAGA-04-07 hyperallometric head",
         None, True),
    ]
    for col, (rgb, title, seg_fn, crop_mesh) in enumerate(row4):
        outline = None
        if crop_mesh:
            rgb = crop_to_mask(rgb, segment_ink_or_gray(rgb, 0.045), margin=0.10)
        elif seg_fn is not None:
            outline = boundary_xy(seg_fn(rgb))
        ax = fig.add_subplot(gs[3, col])
        show_plate(ax, rgb, title, outline=outline, title_size=7.3)

    for y, label_txt in [
        (0.835, "1917 Fig. 369\nSilhouettes"),
        (0.625, "Scholtz 2020\n3D Laser Scans"),
        (0.395, "Giesbrecht 1892 · Sars 1895\nMonograph Lithographs"),
        (0.155, "Thompson 1917 Engravings\nAnt Castes"),
    ]:
        fig.text(0.018, y, label_txt, ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color=INK, rotation=90)

    fig.text(
        0.5, 0.012,
        "Provenance  ·  Crabs: Scholtz, Knötel & Baum 2020, Zoomorphology 139:293–308, Figs. 1–2   |   "
        "Copepods: Giesbrecht 1892, Fauna u. Flora d. Golfes v. Neapel 19, Taf. 4   |   "
        "Amphipods: G.O. Sars 1895, An Account of the Crustacea of Norway I, Pls. 2, 52, 69 (BHL / Wikimedia, public domain)   |   "
        "1917 engravings: Project Gutenberg i743 / i746\n"
        "Ant heads: AntWeb pinned specimens of Pheidole aberrans   |   "
        "Micro-CT: Antscan 2026 (Nature Methods) Acanthomyrmex GAGA-04-06/07   |   "
        "[Cyan] extracted silhouette   [Gold] 8 homologous Type-I/II landmarks",
        ha="center", va="bottom", fontsize=8.0, color=INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0),
    )

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)
    print(f"Saved {OUT_FIG} ({OUT_FIG.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
