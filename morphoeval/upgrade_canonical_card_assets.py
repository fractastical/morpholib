#!/usr/bin/env python3
"""
Upgrade the Specimen & Simulation visual assets for the top Canonical Foundational Cards
in `docs/index.html` and `README.md`:

1. 1917 D'Arcy Thompson (`thompson-scarus-pomacanthus`):
   - `datasets/thompson/scarus_pomacanthus_600pt_specimen.jpg`: Horizontally Procrustes-aligned
     real photographs of *Scarus ghobban* and *Pomacanthus semicirculatus* with their
     600-point continuous boundary silhouettes overlaid at the exact same horizontal angle.
   - `docs/media/thompson_scarus_pomacanthus.gif`: Animated 600-point continuous fish silhouette
     + 8-parameter oblique coaxial-circle coordinate grid + live Area-of-Difference (|Ω_A Δ Ω_B|)
     morphing animation (IoU: 0.662 -> 0.749, |A Δ B|: 47.0% -> 28.5%).

2. 1952 Alan Turing / Kondo (`turing-gray-scott`):
   - `datasets/turing/pomacanthus_imperator_flank_stripes_specimen.jpg`: High-clarity close-up
     of the striped flank of *Pomacanthus imperator* showing the parallel and bifurcating
     yellow-and-blue reaction-diffusion stripes filling the frame.

3. 1966 David Raup (`raup-shell-coiling`):
   - `datasets/raup/raup_coiled_shells_specimen.jpg`: 2x3 whole-shell specimen plate of the 6
     real coiled shell genera (*Nautilus pompilius*, *Turritella communis*, *Dactylioceras*
     ammonite, *Conus textile*, *Cerithium vulgatum*, *Helix pomatia*) in the same whole-shell
     viewing angles as the 2x3 Raup simulation.
   - `docs/media/raup_shell_coiling.gif`: 2x3 3D Raup shell coiling animation rotating in
     continuous oblique/lateral 3D perspective so all 6 shell spires and whorls match the
     specimen viewing angles on every frame.
"""
from __future__ import annotations

import importlib.util
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_commons_thumb(file_title: str, width: int = 960) -> Image.Image:
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&titles="
        + urllib.parse.quote(file_title)
        + f"&prop=imageinfo&iiprop=url&iiurlwidth={width}&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "MorphoSpaceBot/1.0 (https://github.com/fractastical/morpholib)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        pages = json.loads(r.read().decode())["query"]["pages"]
        page = next(iter(pages.values()))
        ii = page["imageinfo"][0]
        img_url = ii.get("thumburl", ii["url"])
    req_img = urllib.request.Request(img_url, headers={"User-Agent": "MorphoSpaceBot/1.0"})
    with urllib.request.urlopen(req_img, timeout=20) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def build_thompson_600pt_assets():
    print("1. Building 1917 Thompson 600-point continuous silhouette specimen & animation...")
    paradox = load_module("paradox", "thompson/1917-thompson-silhouette-area-paradox.py")
    at, sp_mod = paradox.load_modules()
    names, families, X = at.load_landmarks()
    Y = at.gpa(X)
    O_sp = sp_mod.outline_analysis(Y, names, allow_download=False)

    s_ph = O_sp["photos"]["Scarus_ghobban"]
    t_ph = O_sp["photos"]["Pomacanthus_semicirculatus"]
    S, BS = s_ph["landmarks"], s_ph["boundary"]
    T, BT = t_ph["landmarks"], t_ph["boundary"]

    # Unwarped Procrustes similarity baseline & 17-landmark fit & Direct 600-pt coaxial fit
    BS_unw = paradox.unwarped_similarity_silhouette(S, BS, T, at, sp_mod)
    p_lm, _ = at.fit_nonlinear_single("coaxial", S, T)
    BS_lm = at.FAMILY_FUNCS["coaxial"](p_lm, BS)
    bd_lm = paradox.area_diff_breakdown(BS_lm, BT, grid=420)

    p_sil, BS_600, apply_p = paradox.fit_direct_to_silhouette("coaxial", S, BS, T, BT, at)
    bd_600 = paradox.area_diff_breakdown(BS_600, BT, grid=420)

    # --- 1A. Build `datasets/thompson/scarus_pomacanthus_600pt_specimen.jpg` ---
    # Rotate each specimen image by the exact angle of its similarity transform `ab[0]`
    # so both fish face left horizontally at the exact same angle as the 600-pt silhouettes!
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), dpi=160, facecolor="#0b1325")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.04, hspace=0.22)
    fig.suptitle(
        "Real Horizontal Specimens & 600-Point Continuous Silhouettes (Procrustes Frame)",
        color="#f8fafc", fontsize=11.5, fontweight="bold", y=0.965
    )

    spec_configs = [
        ("Scarus_ghobban", s_ph, BS_unw, "#38bdf8", "Source Specimen: Scarus ghobban (Blue-barred Parrotfish, N = 600 pts)"),
        ("Pomacanthus_semicirculatus", t_ph, BT, "#f43f5e", "Target Specimen: Pomacanthus semicirculatus (Semicircle Angelfish, N = 600 pts)"),
    ]
    for ax, (key, ph, B_al, col, title) in zip(axes, spec_configs):
        img = ph["img"]
        B_px = ph["boundary_px"]  # (x, y_bottom_up)
        ab = ph["ab"]
        # Rotate image & boundary_px by angle(ab[0]) so snout->peduncle is strictly horizontal
        ang_deg = float(np.degrees(np.angle(ab[0])))
        H, W = img.shape[:2]
        cx = W / 2.0
        pil_im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        bg_rgb = tuple(int(v * 255) for v in np.median(img[:8, :8], axis=(0, 1)))
        pil_rot = pil_im.rotate(-ang_deg, resample=Image.BICUBIC, expand=False, fillcolor=bg_rgb)

        # Rotate B_px around (cx, H / 2) by +ang_deg in bottom-up coordinates
        rad = np.radians(ang_deg)
        c_r, s_r = np.cos(rad), np.sin(rad)
        dx = B_px[:, 0] - cx
        dy = B_px[:, 1] - (H / 2.0)
        bx_rot = cx + c_r * dx - s_r * dy
        by_rot = (H / 2.0) + s_r * dx + c_r * dy

        # Crop tightly around the rotated fish contour with clean padding
        xmin, xmax = max(0, int(bx_rot.min() - 22)), min(W, int(bx_rot.max() + 22))
        ymin_bu, ymax_bu = max(0, int(by_rot.min() - 18)), min(H, int(by_rot.max() + 18))
        row_top = H - ymax_bu
        row_bot = H - ymin_bu
        crop_arr = np.asarray(pil_rot)[row_top:row_bot, xmin:xmax]

        ax.imshow(crop_arr, extent=(xmin, xmax, ymin_bu, ymax_bu), aspect="equal")
        ax.plot(bx_rot, by_rot, color="#ffffff", lw=2.6, alpha=0.92)
        ax.plot(bx_rot, by_rot, color=col, lw=1.6)
        ax.scatter(bx_rot[::10], by_rot[::10], color=col, edgecolor="#0b1325", s=14, linewidths=0.5, zorder=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin_bu, ymax_bu)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.5)
        ax.set_title(title, color=col, fontsize=9.8, fontweight="bold", pad=5)

    out_spec = ROOT / "datasets" / "thompson" / "scarus_pomacanthus_600pt_specimen.jpg"
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_spec, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_spec} ({out_spec.stat().st_size // 1024} KB)")

    # --- 1B. Build `docs/media/thompson_scarus_pomacanthus.gif` ---
    base_grid = sp_mod.grid_lines(S, n=9, m=90, pad=0.14)
    grid_unw = [paradox.unwarped_similarity_silhouette(S, ln, T, at, sp_mod) for ln in base_grid]
    grid_600 = [apply_p(p_sil, ln) for ln in base_grid]

    t_vals = np.concatenate([
        np.linspace(0.0, 0.0, 3),
        np.linspace(0.0, 1.0, 15),
        np.linspace(1.0, 1.0, 7),
        np.linspace(1.0, 0.0, 7),
    ])

    fig_anim, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=105, facecolor="#0b1325")
    fig_anim.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.08, wspace=0.10)

    def render_frame(idx):
        t = float(t_vals[idx])
        ax_l.clear()
        ax_r.clear()
        ax_l.set_facecolor("#0f172a")
        ax_r.set_facecolor("#0f172a")

        cur_B = (1.0 - t) * BS_unw + t * BS_600
        bd_t = paradox.area_diff_breakdown(cur_B, BT, grid=280)

        # Left Panel: Deforming Coaxial-Circle Grid + 600-Point Continuous Silhouettes
        for g0, g1 in zip(grid_unw, grid_600):
            gt = (1.0 - t) * g0 + t * g1
            ax_l.plot(gt[:, 0], gt[:, 1], color="#38bdf8", lw=0.75, alpha=0.45)

        ax_l.fill(BT[:, 0], BT[:, 1], color="#f43f5e", alpha=0.22)
        ax_l.plot(BT[:, 0], BT[:, 1], color="#fb7185", lw=1.5, ls="--", label="Target: Pomacanthus (600 pts)")
        ax_l.fill(cur_B[:, 0], cur_B[:, 1], color="#22c55e", alpha=0.28)
        ax_l.plot(cur_B[:, 0], cur_B[:, 1], color="#4ade80", lw=2.0, label="Warped Scarus (600-pt Coaxial Fit)")
        ax_l.scatter(cur_B[::12, 0], cur_B[::12, 1], color="#facc15", s=8, zorder=5)
        ax_l.set_xlim(-0.48, 0.48)
        ax_l.set_ylim(-0.42, 0.42)
        ax_l.set_aspect("equal")
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        for sp in ax_l.spines.values():
            sp.set_color("#334155")
        ax_l.set_title(
            f"600-Pt Silhouette & Coaxial Grid (t = {t:.2f})\nScarus ghobban → Pomacanthus semicirculatus",
            color="#f8fafc", fontsize=9.5, fontweight="bold"
        )
        ax_l.legend(loc="lower left", fontsize=7.2, facecolor="#0b1325", edgecolor="#334155", labelcolor="#f8fafc")

        # Right Panel: Live Area-of-Difference |Ω_A(t) Δ Ω_B|
        ra, rb = bd_t["ra"], bd_t["rb"]
        xmin, xmax, ymin, ymax = bd_t["extent"]
        H, W = ra.shape
        img_rgba = np.zeros((H, W, 4), dtype=float)
        img_rgba[:, :] = [0.06, 0.09, 0.16, 1.0]
        img_rgba[ra & rb] = [0.13, 0.77, 0.37, 0.88]        # Green: shared overlap
        img_rgba[rb & (~ra)] = [0.94, 0.24, 0.36, 0.88]     # Crimson: missing target
        img_rgba[ra & (~rb)] = [0.22, 0.58, 0.98, 0.86]     # Royal blue: source overhang
        ax_r.imshow(img_rgba, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal")
        ax_r.plot(BT[:, 0], BT[:, 1], color="#fecdd3", lw=1.2, ls="--")
        ax_r.plot(cur_B[:, 0], cur_B[:, 1], color="#ffffff", lw=1.4)
        if t > 0.65:
            ax_r.plot(BS_lm[:, 0], BS_lm[:, 1], color="#f59e0b", lw=1.2, ls=":", label="Old 17-LM Fit (IoU=0.618)")
            ax_r.legend(loc="lower right", fontsize=7.0, facecolor="#0b1325", edgecolor="#f59e0b", labelcolor="#facc15")
        ax_r.set_xlim(-0.48, 0.48)
        ax_r.set_ylim(-0.42, 0.42)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        for sp in ax_r.spines.values():
            sp.set_color("#334155")
        ax_r.set_title(
            f"Area of Difference |Ω_A Δ Ω_B| = {bd_t['sym_diff_pct_of_target']:.1f}%  (IoU = {bd_t['iou']:.3f})\n"
            f"[Green] Overlap  [Red] Miss {bd_t['missing_pct']:.1f}%  [Blue] Over {bd_t['overhang_pct']:.1f}%",
            color="#4ade80" if t > 0.8 else "#38bdf8", fontsize=9.2, fontweight="bold"
        )
        fig_anim.suptitle(
            "D'Arcy Thompson (1917) — Direct 600-Point Continuous Silhouette Coaxial Transformation\n"
            f"600-Pt Fit: IoU = {bd_600['iou']:.3f} (|Ω_A Δ Ω_B| = {bd_600['sym_diff_pct_of_target']:.1f}%)   vs.   17-Landmark Stick Fit: IoU = {bd_lm['iou']:.3f} (|Ω_A Δ Ω_B| = {bd_lm['sym_diff_pct_of_target']:.1f}%)",
            color="#facc15", fontsize=10.2, fontweight="bold", y=0.96
        )

    anim = FuncAnimation(fig_anim, render_frame, frames=len(t_vals), interval=120)
    out_gif = ROOT / "docs" / "media" / "thompson_scarus_pomacanthus.gif"
    anim.save(out_gif, writer=PillowWriter(fps=9))
    plt.close(fig_anim)
    print(f"  Saved {out_gif} ({out_gif.stat().st_size // 1024} KB)")


def build_turing_flank_closeup():
    print("2. Building 1952 Turing / Kondo Pomacanthus imperator flank stripes close-up specimen...")
    try:
        im_hi = fetch_commons_thumb(
            "File:Pez ángel emperador (Pomacanthus imperator), mar Rojo, Egipto, 2023-04-17, DD 103.jpg",
            width=1280,
        )
    except Exception:
        im_hi = Image.open(ROOT / "datasets" / "turing" / "pomacanthus_imperator_stripes_specimen.jpg").convert("RGB")

    W, H = im_hi.size
    box_flank = (int(W * 0.30), int(H * 0.27), int(W * 0.74), int(H * 0.67))
    flank_crop = im_hi.crop(box_flank).resize((800, 600), Image.LANCZOS)

    box_fish = (int(W * 0.08), int(H * 0.12), int(W * 0.92), int(H * 0.84))
    fish_inset = im_hi.crop(box_fish).resize((230, 155), Image.LANCZOS)
    fish_inset = ImageOps.expand(fish_inset, border=3, fill=(250, 204, 21))
    flank_crop.paste(fish_inset, (800 - fish_inset.width - 14, 600 - fish_inset.height - 14))

    out_turing = ROOT / "datasets" / "turing" / "pomacanthus_imperator_flank_stripes_specimen.jpg"
    flank_crop.save(out_turing, quality=92)
    flank_crop.save(ROOT / "datasets" / "turing" / "pomacanthus_imperator_stripes_specimen.jpg", quality=92)
    print(f"  Saved {out_turing} ({out_turing.stat().st_size // 1024} KB)")


def build_raup_six_shells_and_gif():
    print("3. Building 1966 Raup 2x3 whole-shell specimen plate & matching-angle 3D animation...")
    shell_sources = [
        ("Planispiral — Nautilus pompilius\n(W=3.0, D=0.0, T=0.0)", "File:Concha de Nautilus pompilius achada em 1758.jpg", "#2E86AB"),
        ("Turreted — Turritella communis\n(W=1.3, D=0.05, T=3.5)", "File:Gemeine Turmschnecke Turritella communis.jpg", "#A23B72"),
        ("Evolute disc — Dactylioceras ammonite\n(W=2.0, D=0.45, T=0.0)", "File:Dactylioceras commune 45.jpg", "#F18F01"),
        ("Conical — Conus textile\n(W=3.5, D=0.0, T=1.5)", "LOCAL:datasets/conus/conus_textile_shell.jpg", "#C73E1D"),
        ("High-spired — Cerithium vulgatum\n(W=1.6, D=0.1, T=2.5)", "File:Cerithium vulgatum 01.JPG", "#6A994E"),
        ("Globose — Helix pomatia\n(W=3.0, D=0.1, T=1.0)", "File:Helix pomatia 004.JPG", "#BC4749"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), dpi=150, facecolor="#0b1325")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.08, hspace=0.22)
    fig.suptitle(
        "Empirical Coiled Shells Across Raup's (W, D, T) Morphospace (Whole-Shell Views)",
        color="#f8fafc", fontsize=12.0, fontweight="bold", y=0.96
    )

    for ax, (title, src, col) in zip(axes.flat, shell_sources):
        if src.startswith("LOCAL:"):
            im = Image.open(ROOT / src.split("LOCAL:", 1)[1]).convert("RGB")
            # Preserve the full vertical Conus shell from apex to base by padding the sides with its edge color
            w, h = im.size
            target_w = int(h * 4.0 / 3.0)
            edge_rgb = tuple(int(v) for v in np.median(np.asarray(im)[:, :10], axis=(0, 1)))
            padded = Image.new("RGB", (target_w, h), edge_rgb)
            padded.paste(im, ((target_w - w) // 2, 0))
            im_c = padded.resize((440, 330), Image.LANCZOS)
        else:
            im = fetch_commons_thumb(src, width=640)
            w, h = im.size
            target_ratio = 4.0 / 3.0
            cur_ratio = w / h
            if cur_ratio > target_ratio:
                nw = int(h * target_ratio)
                off = (w - nw) // 2
                im_c = im.crop((off, 0, off + nw, h))
            else:
                nh = int(w / target_ratio)
                off = (h - nh) // 2
                im_c = im.crop((0, off, w, off + nh))
            im_c = im_c.resize((440, 330), Image.LANCZOS)
        ax.imshow(np.asarray(im_c))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(2.0)
        ax.set_title(title, color="#f8fafc", fontsize=8.6, fontweight="bold", pad=4)

    out_raup_spec = ROOT / "datasets" / "raup" / "raup_coiled_shells_specimen.jpg"
    out_raup_spec.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_raup_spec, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_raup_spec} ({out_raup_spec.stat().st_size // 1024} KB)")

    raup_mod = load_module("raup_mod", "models/create_raup_animations.py")
    shells = raup_mod.panels(raup_mod.load_raup_data())
    fig_r = plt.figure(figsize=(12.0, 8.0), dpi=80, facecolor="#0b1325")
    fig_r.subplots_adjust(0.01, 0.04, 0.99, 0.88, wspace=0.02, hspace=0.14)
    axes_r = [fig_r.add_subplot(2, 3, i + 1, projection="3d", facecolor="#0f172a") for i in range(6)]
    meshes = [raup_mod.raup_shell(W=s["W"], D=s["D"], T=s["T"], S=s["S"], turns=s["turns"], res=52) for s in shells]
    n_frames = 32

    def animate_raup(frame):
        azim = (frame / n_frames) * 360.0
        elev = 24.0 + 14.0 * np.sin(2.0 * np.pi * frame / n_frames)
        for ax, s, (X, Y, Z) in zip(axes_r, shells, meshes):
            ax.clear()
            ax.set_facecolor("#0f172a")
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, color=s["color"], alpha=0.55, edgecolor=s["color"], linewidth=0.35)
            raup_mod.fit_axes(ax, X, Y, Z, zoom=1.32)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_title(
                f"{s['name']} — cf. {s['cf']}\nW={s['W']:.2f}  D={s['D']:.2f}  T={s['T']:.2f}",
                color="#f8fafc", fontsize=9.2, fontweight="bold", pad=2
            )
        fig_r.suptitle(
            f"Raup's Coiling Model (1966): Six Canonical Forms in 3D Perspective (elev={elev:.0f}°, azim={azim:.0f}°)",
            color="#facc15", fontsize=11.8, fontweight="bold", y=0.96
        )

    anim_r = FuncAnimation(fig_r, animate_raup, frames=n_frames, interval=110)
    out_raup_gif = ROOT / "docs" / "media" / "raup_shell_coiling.gif"
    anim_r.save(out_raup_gif, writer=PillowWriter(fps=10))
    plt.close(fig_r)
    print(f"  Saved {out_raup_gif} ({out_raup_gif.stat().st_size // 1024} KB)")


def build_conus_rule30_gif_and_sync_specimens():
    print("4. Upgrading 1984 Wolfram/Meinhardt Conus textile 3D pigmented shell GIF & syncing docs/media/specimens/...")
    import shutil
    # Generate Rule 30 + activator-substrate space-time field and map its mahogany/cream
    # tent pattern directly onto a 3D Conus textile Raup shell at the exact vertical specimen angle!
    raup_mod = load_module("raup_mod", "models/create_raup_animations.py")
    W_grid, T_steps = 160, 160
    ca = np.zeros((T_steps, W_grid), dtype=int)
    rng = np.random.default_rng(30)
    ca[0] = rng.choice([0, 1], size=W_grid, p=[0.78, 0.22])
    for t in range(1, T_steps):
        left = np.roll(ca[t - 1], 1)
        center = ca[t - 1]
        right = np.roll(ca[t - 1], -1)
        ca[t] = left ^ (center | right)  # Exact Wolfram Elementary Cellular Automaton Rule 30

    # Smooth slightly with anisotropic triangular kernel to match biological pigment diffusion
    from scipy.ndimage import gaussian_filter
    pigment = gaussian_filter(ca.astype(float), sigma=(0.8, 0.6))
    pigment = (pigment - pigment.min()) / (pigment.max() - pigment.min() + 1e-9)

    # Build RGB texture in Conus textile palette: cream white (#fff8eb) tents + mahogany brown (#6b240c) lines
    cream = np.array([1.0, 0.97, 0.91])
    amber = np.array([0.82, 0.48, 0.16])
    mahogany = np.array([0.36, 0.11, 0.04])
    tex_rgb = np.zeros((T_steps, W_grid, 3), dtype=float)
    for c in range(3):
        tex_rgb[:, :, c] = np.where(
            pigment < 0.48,
            cream[c] * (1.0 - pigment / 0.48) + amber[c] * (pigment / 0.48),
            amber[c] * (1.0 - (pigment - 0.48) / 0.52) + mahogany[c] * ((pigment - 0.48) / 0.52),
        )

    # 3D Conus textile mesh (orient vertically with apex at top, matching conus_textile_shell.jpg!)
    X, Y, Z = raup_mod.raup_shell(W=3.2, D=0.02, T=1.65, S=0.55, turns=4, res=72, apex_up=True)
    # Map (phi, theta) surface to the Rule 30 pigment texture
    n_phi, n_theta = X.shape
    facecolors = np.zeros((n_phi, n_theta, 4), dtype=float)
    for i in range(n_phi):
        for j in range(n_theta):
            ti = int((j / n_theta) * (T_steps - 1))
            xi = int((i / n_phi) * (W_grid - 1))
            facecolors[i, j, :3] = tex_rgb[ti, xi]
            facecolors[i, j, 3] = 1.0

    fig_c = plt.figure(figsize=(10.0, 4.8), dpi=95, facecolor="#0b1325")
    ax_3d = fig_c.add_subplot(1, 2, 1, projection="3d", facecolor="#0f172a")
    ax_kym = fig_c.add_subplot(1, 2, 2, facecolor="#0f172a")
    fig_c.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.09, wspace=0.12)

    n_frames = 28

    def animate_conus(frame):
        azim = (frame / n_frames) * 360.0
        t_cur = int(25 + (T_steps - 26) * (frame / (n_frames - 1)))

        # Left Panel: 3D Pigmented Conus textile Shell (same vertical apex-up angle as specimen!)
        ax_3d.clear()
        ax_3d.set_facecolor("#0f172a")
        fc_now = facecolors.copy()
        j_cut = int((t_cur / T_steps) * n_theta)
        if j_cut < n_theta:
            fc_now[:, j_cut:, :3] = 0.25 + 0.75 * fc_now[:, j_cut:, :3]
        ax_3d.plot_surface(
            X, Y, Z, rstride=1, cstride=2, facecolors=fc_now,
            antialiased=True, shade=True, linewidth=0.0
        )
        raup_mod.fit_axes(ax_3d, X, Y, Z, zoom=1.35)
        ax_3d.view_init(elev=14.0, azim=azim)
        ax_3d.set_axis_off()
        ax_3d.set_title(
            f"3D Conus textile Shell Surface\nRule 30 Mantle Accretion (azim={azim:.0f}°)",
            color="#f8fafc", fontsize=9.8, fontweight="bold"
        )

        # Right Panel: Live 1D Mantle-Edge Rule 30 Space-Time Kymograph
        ax_kym.clear()
        ax_kym.set_facecolor("#0f172a")
        shown = np.ones_like(tex_rgb) * 0.12
        shown[:t_cur, :, :] = tex_rgb[:t_cur, :, :]
        ax_kym.imshow(shown, aspect="auto", extent=[0, W_grid, T_steps, 0])
        ax_kym.axhline(t_cur, color="#38bdf8", lw=2.0, label=f"Active Mantle Edge (t = {t_cur})")
        ax_kym.set_xlabel("Mantle Edge Position x (Cells)", color="#cbd5e1", fontsize=8.5)
        ax_kym.set_ylabel("Accretion Time t ↓", color="#cbd5e1", fontsize=8.5)
        ax_kym.tick_params(colors="#94a3b8", labelsize=7.5)
        for sp in ax_kym.spines.values():
            sp.set_color("#38bdf8")
        ax_kym.set_title(
            "Wolfram Rule 30 Space-Time Kymograph\ns_i(t+1) = s_{i-1}(t) XOR (s_i(t) OR s_{i+1}(t))",
            color="#f8fafc", fontsize=9.8, fontweight="bold"
        )
        ax_kym.legend(loc="lower right", fontsize=7.5, facecolor="#0b1325", edgecolor="#38bdf8", labelcolor="#f8fafc")
        fig_c.suptitle(
            "Wolfram (1984) & Meinhardt (1987) — Conus textile Shell Pigmentation (3D Shell & Rule 30 Kymograph)",
            color="#facc15", fontsize=10.8, fontweight="bold", y=0.96
        )

    anim_c = FuncAnimation(fig_c, animate_conus, frames=n_frames, interval=110)
    out_conus_gif = ROOT / "docs" / "media" / "conus_textile_rule30.gif"
    anim_c.save(out_conus_gif, writer=PillowWriter(fps=10))
    plt.close(fig_c)
    print(f"  Saved {out_conus_gif} ({out_conus_gif.stat().st_size // 1024} KB)")

    # Copy all 8 canonical pillar specimens into `docs/media/specimens/` so they ALWAYS load locally
    # inside `docs/` without needing `../` traversal or remote fallback!
    spec_dir = ROOT / "docs" / "media" / "specimens"
    spec_dir.mkdir(parents=True, exist_ok=True)
    canonical_map = {
        "thompson-scarus-pomacanthus": "datasets/thompson/scarus_pomacanthus_600pt_specimen.jpg",
        "turing-gray-scott": "datasets/turing/pomacanthus_imperator_flank_stripes_specimen.jpg",
        "raup-shell-coiling": "datasets/raup/raup_coiled_shells_specimen.jpg",
        "vogel-phyllotaxis": "datasets/phyllotaxis/helianthus_annuus_capitulum_specimen.jpg",
        "wolfram-conus-textile-rule30": "datasets/conus/conus_textile_shell.jpg",
        "runions-leaf-venation": "datasets/venation/reticulate_vein_skeleton_specimen.jpg",
        "planform-morphology-animation": "datasets/species_gallery/II_Anaima_Invertebrata/Schmidtea_mediterranea.jpg",
        "cervera-levin-mafe": "datasets/species_gallery/II_Anaima_Invertebrata/Schmidtea_mediterranea.jpg",
    }
    for slug, rel_src in canonical_map.items():
        dst = spec_dir / f"{slug}.jpg"
        shutil.copy2(ROOT / rel_src, dst)
        print(f"  Mirrored {rel_src} -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    build_thompson_600pt_assets()
    build_turing_flank_closeup()
    build_raup_six_shells_and_gif()
    build_conus_rule30_gif_and_sync_specimens()
