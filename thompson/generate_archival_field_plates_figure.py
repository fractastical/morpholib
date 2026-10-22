#!/usr/bin/env python3
"""
Generate a comprehensive 3x3 Archival & Field Specimen Provenance Plate
('docs/thompson_1917_archival_field_plates.png') displaying the actual
historical museum lithographs (Monaco 1896, Monaco 1911, Goode & Bean 1896,
Bloch 1796, BMNH 1887) and NOAA/IFREMER field specimen photographs for all
9 of D'Arcy Thompson's 1917 taxa, overlaid with their extracted 600-semilandmark
outlines (emerald/cyan) and 17 homologous skeletal landmarks (9 craniofacial in
gold, 8 fin-base insertions in magenta).
"""

import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".thompson_cache"
EXACT_JSON = REPO_ROOT / "datasets" / "thompson" / "thompson_1917_exact_species_landmarks.json"
OUT_FIG = REPO_ROOT / "docs" / "thompson_1917_archival_field_plates.png"


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def segment_boundary_on_image(im_rgb: Image.Image, sp_mod, is_field_photo: bool = False, tag: str = ""):
    if is_field_photo:
        arr = np.asarray(im_rgb)
        if tag == "scarus":
            m = sp_mod.segment_scarus(arr)
        else:
            m = sp_mod.segment_pomacanthus(arr)
        B_px = sp_mod.boundary_points(m, 600)
        # B_px has y upward (H - row); convert back to image pixel coordinates (col, row):
        H = arr.shape[0]
        return np.column_stack([B_px[:, 0], H - B_px[:, 1]])
    else:
        W0, H0 = im_rgb.size
        scale = 450.0 / max(W0, H0)
        W_s, H_s = int(W0 * scale), int(H0 * scale)
        im_s = im_rgb.resize((W_s, H_s), Image.LANCZOS)
        a = np.asarray(im_s).astype(float) / 255.0
        border = np.zeros((H_s, W_s), bool)
        border[:6, :] = border[-6:, :] = border[:, :6] = border[:, -6:] = True
        lab = color.rgb2lab(a)
        bg_mu = np.median(lab[border], axis=0)
        d = np.sqrt(((lab - bg_mu) ** 2).sum(-1))
        thr = np.percentile(d[border], 92) + 6.0
        m = morphology.closing(filters.gaussian(d, 1.2) > thr, morphology.disk(3))
        m = ndi.binary_fill_holes(m)
        lab_m = measure.label(m)
        if lab_m.max() > 0:
            sizes = np.bincount(lab_m.ravel())
            sizes[0] = 0
            m = lab_m == sizes.argmax()
        B_px = sp_mod.boundary_points(m, 600)
        col_orig = B_px[:, 0] * (W0 / W_s)
        row_orig = (H_s - B_px[:, 1]) * (H0 / H_s)
        return np.column_stack([col_orig, row_orig])


def main():
    sil_mod = load_module("sil_mod", "thompson/1917-thompson-silhouette-area-paradox.py")
    at, sp_mod = sil_mod.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)
    meta = json.loads(EXACT_JSON.read_text(encoding="utf-8"))

    head_idx = [0, 1, 10, 11, 12, 13, 14, 15, 16]
    fin_idx = [2, 3, 4, 5, 6, 7, 8, 9]
    trunk_loop = [1, 2, 3, 4, 5, 6, 7, 0, 1]

    fig = plt.figure(figsize=(17.5, 12.2), dpi=220, facecolor="white")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.34, wspace=0.16, top=0.88, bottom=0.07, left=0.03, right=0.97)

    fig.suptitle(
        "Archival Museum Plates & Field Specimen Photographs for All Nine of D'Arcy Thompson's 1917 Taxa\n"
        "Showing Extracted 600-Semilandmark Whole-Body Outlines (Cyan) vs. 17 Homologous Skeletal Landmarks (9 Head in Gold, 8 Fin Bases in Pink)",
        fontsize=13.2, fontweight="bold", color="#0f172a", y=0.965,
    )

    spec_cards = [
        ("Argyropelecus_olfersii", "1. Argyropelecus olfersii (Fig. 373, p. 748)\nMonaco Deep-Sea Lithograph (Collett 1896)"),
        ("Sternoptyx_diaphana", "2. Sternoptyx diaphana (Fig. 374, p. 748)\nMonaco Deep-Sea Lithograph (Zugmayer 1911)"),
        ("Scarus_ghobban", "3. Scarus ghobban (Fig. 375, p. 749)\nIFREMER Collected Specimen Photograph (Huet)"),
        ("Pomacanthus_semicirculatus", "4. Pomacanthus semicirculatus (Fig. 376, p. 749)\nField Underwater Photograph (Levy, CC BY 2.0)"),
        ("Polyprion_americanus", "5. Polyprion americanus (Fig. 377, p. 750)\nUS National Museum Plate (Goode & Bean 1896)"),
        ("Pristigenys_alta", "6. Pristigenys alta [= Pseudopriacanthus] (Fig. 378)\nNOAA Fisheries SEFSC Specimen Photograph"),
        ("Scorpaena_scrofa", "7. Scorpaena scrofa (Fig. 379, p. 750)\nBloch (1796) Ichthyologie Plate 182 (BHL)"),
        ("Antigonia_capros", "8. Antigonia capros (Fig. 380, p. 750)\nNOAA Fisheries SEFSC Specimen Photograph"),
        ("Mola_mola", "9. Mola mola [= Orthagoriscus mola] (Fig. 382, p. 751)\nBritish Museum Natural History Engraving (1887)"),
    ]

    for idx_card, (sp_key, card_title) in enumerate(spec_cards):
        r_i, c_i = divmod(idx_card, 3)
        ax = fig.add_subplot(gs[r_i, c_i])

        if sp_key in ("Scarus_ghobban", "Pomacanthus_semicirculatus"):
            ph_data = O_sp["photos"][sp_key]
            img_arr = ph_data["img"]
            H_im = img_arr.shape[0]
            B_px_up = ph_data["boundary_px"]
            L_px_up = ph_data["landmarks_px"]
            B_px = np.column_stack([B_px_up[:, 0], H_im - B_px_up[:, 1]])
            pts_px = np.column_stack([L_px_up[:, 0], H_im - L_px_up[:, 1]])
            ax.imshow(img_arr)
            ax.plot(B_px[:, 0], B_px[:, 1], color="#00f5d4", lw=2.2, label="600-Semilandmark Whole-Body Outline")
        else:
            cfg = meta["specimens"][sp_key]
            img_path = CACHE_DIR / cfg["cache_file"]
            im = Image.open(img_path).convert("RGB")
            W0, H0 = im.size
            B_px = segment_boundary_on_image(im, sp_mod, is_field_photo=False)
            pts_px = np.array([[x * W0, y * H0] for x, y in cfg["landmarks_fractional_top_left"]], dtype=float)
            ax.imshow(im)
            ax.plot(B_px[:, 0], B_px[:, 1], color="#06b6d4", lw=2.2, label="600-Semilandmark Whole-Body Outline")

        ax.plot(pts_px[trunk_loop, 0], pts_px[trunk_loop, 1], color="#facc15", lw=1.5, ls="--", alpha=0.9, label="8-Vertex Landmark Trunk Wireframe")
        ax.scatter(pts_px[head_idx, 0], pts_px[head_idx, 1], color="#f59e0b", edgecolor="#0f172a", s=55, zorder=8,
                   label="9 Craniofacial Landmarks (LM 1–2, 11–17)")
        ax.scatter(pts_px[fin_idx, 0], pts_px[fin_idx, 1], color="#ec4899", edgecolor="#0f172a", s=50, marker="s", zorder=8,
                   label="8 Fin-Base Insertion Landmarks (LM 3–10)")

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(card_title, fontsize=9.8, fontweight="bold", color="#0f172a", pad=8)
        if idx_card == 0:
            ax.legend(loc="lower left", fontsize=7.0, framealpha=0.92, facecolor="white")

    fig.text(
        0.5, 0.015,
        "Provenance & Segmentation Key:   [Cyan Contour] 600-Semilandmark Whole-Body Outline (Captures Dorsal, Anal, Caudal Fins & Ventral Keel)   |   "
        "[Gold Circles] 9 Craniofacial Landmarks (53% Weight)   |   [Pink Squares] 8 Basal Fin-Insertion Landmarks (0% on Distal Fin Margins)",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.0),
    )

    fig.savefig(OUT_FIG, dpi=220)
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
