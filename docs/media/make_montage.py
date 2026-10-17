#!/usr/bin/env python3
"""Tile the README animations into one GIF: every form in the repository, side by side, with the
MorphoEval scores in the last tile.

    python3 docs/media/make_montage.py                # -> docs/media/all_forms_montage.gif
    python3 docs/media/make_montage.py --frames 36 --tile-width 216 --out /tmp/m.gif

Each clip is resampled to the same number of frames across its own loop (so a 5-second Turing run and
a 4-minute Limbform slideshow both play once per montage loop), fitted whole into its tile (no
cropping; the panels inside the source GIFs stay intact), and labelled year + name.  One shared
128-colour palette, Floyd-Steinberg dither.  Scores come from the audit YAMLs via
morphoeval/report/build_report.py, not from this file.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# (gif, year, label, script) in year order; the label is what fits under a ~200-px tile
CLIPS = [
    ("thompson_transformations.gif",       "1917", "Thompson, transformations (real fish)",  "thompson/1917-thompson.py"),
    ("thompson_scarus_pomacanthus.gif",    "1917", "Thompson, Scarus → Pomacanthus", "thompson/1917-thompson-scarus-pomacanthus.py"),
    ("turing_gray_scott.gif",              "1952", "Turing / Gray–Scott patterns",            "1952-turing-morpho.py"),
    ("raup_shell_coiling.gif",             "1966", "Raup, shell coiling",                     "create_raup_animations.py"),
    ("petroff_stromatolite_dome_to_cone.gif", "2010", "Petroff, stromatolites",               "2010-petroff-stromatolite.py"),
    ("chindapol_coral_flow_growth.gif",    "2013", "Chindapol, coral in flow",                "2013-chindapol-coral-flow.py"),
    ("planform_morphospace_timeline.gif",  "2013", "Planform, planarian morphologies",        "2013-lobo-planform-morphology.py"),
    ("limbform_morphospace_timeline.gif",  "2014", "Limbform, limb morphologies",             "2014-lobo-limbform-morphology.py"),
    ("celegans_embryo_timeline.gif",       "2015", "WormGUIDES, C. elegans embryo",           "2015-santella-wormguides-celegans.py"),
    ("hoffmann_wing_lloyd.gif",            "2018", "Hoffmann, wing venation",                 "2018-hoffmann-wing-venation.py"),
    ("cervera_levin_mafe_gj_block.gif",    "2021", "Cervera–Levin–Mafe, bioelectric",         "2021-Cervera–Levin–Mafe.py"),
    ("kahng_coral_geometry_growth.gif",    "2024", "Kahng, coral geometry",                   "2024-kahng-coral-geometry.py"),
    ("odonata_wing_cells_deep_time.gif",   "2026", "Eshghi, Odonata wing cells",              "2026-eshghi-odonata-wing-cells.py"),
]
BG, FG, DIM, ACCENT = (250, 250, 248), (30, 41, 59), (100, 116, 139), (23, 190, 207)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return ImageFont.truetype(str(d / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def scores() -> dict[str, float]:
    spec = importlib.util.spec_from_file_location("br", ROOT / "morphoeval" / "report" / "build_report.py")
    br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)
    import yaml
    rubric = yaml.safe_load(open(br.RUBRIC))
    latest, _ = br.collect(br.load_audits(), rubric)
    return {r["script"]: r["overall"] for r in latest}


def resample(path: Path, n: int) -> list[Image.Image]:
    im = Image.open(path)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in idx]


def fit(frame: Image.Image, w: int, h: int) -> Image.Image:
    f = frame.copy(); f.thumbnail((w, h), Image.LANCZOS)
    out = Image.new("RGB", (w, h), BG)
    out.paste(f, ((w - f.width) // 2, (h - f.height) // 2))
    return out


def title_tile(w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), FG); d = ImageDraw.Draw(im)
    d.text((10, 10), "Infinite", font=font(22, True), fill=(255, 255, 255))
    d.text((10, 36), "Morphospace", font=font(22, True), fill=(255, 255, 255))
    d.multiline_text((10, 70), f"{len(CLIPS)} reproductions of published\nmodels of biological form,\n1917–2026, each scored for\nfaithfulness by MorphoEval",
                     font=font(10), fill=(203, 213, 225), spacing=3)
    return im


def score_tile(w: int, h: int, sc: dict[str, float]) -> Image.Image:
    im = Image.new("RGB", (w, h), BG); d = ImageDraw.Draw(im)
    d.text((8, 6), "MorphoEval overall (0–100)", font=font(10, True), fill=FG)
    short = {"1952-turing-morpho.py": "Turing", "create_raup_animations.py": "Raup", "2015-santella-wormguides-celegans.py": "C. elegans",
             "2026-eshghi-odonata-wing-cells.py": "Odonata", "thompson/1917-thompson-scarus-pomacanthus.py": "Thompson fish"}
    rows = [(c[1], short.get(c[3], c[2].split(",")[0]), sc[c[3]]) for c in CLIPS if c[3] in sc]
    rows.sort(key=lambda r: -r[2])
    y0, bar_h = 24, max(8, (h - 30) // max(1, len(rows)))
    fnt = font(8)
    for i, (year, name, v) in enumerate(rows):
        y = y0 + i * bar_h
        d.text((8, y), name, font=fnt, fill=FG)
        x0 = 72; x1 = x0 + int((w - x0 - 30) * v / 100)
        d.rectangle([x0, y + 1, x1, y + bar_h - 3], fill=ACCENT if v >= 65 else (DIM if v >= 50 else (220, 38, 38)))
        d.text((x1 + 4, y), f"{v:.0f}", font=fnt, fill=FG)
    return im


def label(tile: Image.Image, year: str, name: str, score: float | None) -> Image.Image:
    d = ImageDraw.Draw(tile); w, h = tile.size
    d.rectangle([0, h - 16, w, h], fill=(255, 255, 255))
    limit = w - 8
    if score is not None:
        s = f"{score:.0f}"; tw = d.textlength(s, font=font(9, True))
        d.text((w - tw - 4, h - 14), s, font=font(9, True), fill=ACCENT if score >= 65 else DIM)
        limit = w - tw - 12
    text = f"{year}  {name}"
    while d.textlength(text, font=font(9)) > limit and len(text) > 8:
        text = text[:-2].rstrip() + "…"
    d.text((4, h - 14), text, font=font(9), fill=FG)
    return tile


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "all_forms_montage.gif"))
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--tile-width", type=int, default=216)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--duration", type=int, default=130, help="ms per frame")
    args = ap.parse_args(argv)

    tw = args.tile_width; th = int(tw * 0.68); gap = 4
    sc = scores()
    clips = [resample(HERE / c[0], args.frames) for c in CLIPS]
    tiles_static = {0: title_tile(tw, th), len(CLIPS) + 1: score_tile(tw, th, sc)}
    n_tiles = len(CLIPS) + 2
    rows = -(-n_tiles // args.cols)
    W, H = args.cols * tw + (args.cols + 1) * gap, rows * th + (rows + 1) * gap

    frames = []
    for k in range(args.frames):
        canvas = Image.new("RGB", (W, H), (226, 232, 240))
        for t in range(n_tiles):
            r, c = divmod(t, args.cols)
            x, y = gap + c * (tw + gap), gap + r * (th + gap)
            if t in tiles_static:
                tile = tiles_static[t]
            else:
                gif, year, name, script = CLIPS[t - 1]
                tile = label(fit(clips[t - 1][k], tw, th), year, name, sc.get(script))
            canvas.paste(tile, (x, y))
        frames.append(canvas)

    # one shared palette from a strip of sampled frames
    strip = Image.new("RGB", (W, H * 4))
    for i, f in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
        strip.paste(f, (0, i * H))
    pal = strip.quantize(colors=128, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]
    out = Path(args.out)
    q[0].save(out, save_all=True, append_images=q[1:], duration=args.duration, loop=0, optimize=True)
    print(f"saved {out}  {W}x{H}, {len(q)} frames, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
