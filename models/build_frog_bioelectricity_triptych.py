#!/usr/bin/env python3
"""
Three frogs, one bioelectric thread: an animated triptych for the Substack opener.

Layout: Galvani and the face time-lapse share the top row; the Ca2+ wave embryos, which are
about 4.7:1, run full width underneath.

Panels (each cropped from an existing animation and re-timed onto one shared loop):
    1. Galvani (1791): the bimetallic arc closes and the frog leg twitches.
       Source: docs/media/galvani_1791_neuromuscular_excitation.gif (models/1791-galvani-bioelectricity.py)
    2. Xenopus laevis embryos, GCaMP Ca2+ imaging (WGD84): a wave in one embryo is followed
       by a wave in its uncontacted neighbour; tracked wavefront vectors in magenta.
       Source: wave-vector-analysis/pretty_waves/no_contact/WGD84_C-oHTHT-pT-Zoom5.60_neighbor_clean.gif
    3. "Face of a frog": time-lapse of the bioelectric pre-pattern of the Xenopus face
       (Vandenberg, Morrie & Adams 2011, Dev. Dyn. 240:1889), as hosted on bioelectricity.net.
       Source: docs/media/face_of_a_frog_timelapse.gif (frames 0-7 are an interview and are skipped)

Usage:
    python3 models/build_frog_bioelectricity_triptych.py
    python3 models/build_frog_bioelectricity_triptych.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
OUT_GIF = MEDIA / "frog_bioelectricity_triptych.gif"

CANVAS_W = 1000
GUTTER = 14
MARGIN = 18
LABEL_H = 54
FOOTER_H = 30
BG = (11, 16, 28)
FG = (232, 236, 244)
SUB = (150, 164, 188)

LOOP_MS = 8400
FRAME_MS = 120

GALVANI = {
    "path": MEDIA / "galvani_1791_neuromuscular_excitation.gif",
    "box": (664, 104, 1300, 820),
    "first": 0,
    "loops": 3,
    "title": "1791 \u00b7 Galvani's frog",
    "sub": "metal arc closes, the leg twitches",
    "accent": (56, 189, 248),
}
WAVES = {
    "path": ROOT / "wave-vector-analysis" / "pretty_waves" / "no_contact" / "WGD84_C-oHTHT-pT-Zoom5.60_neighbor_clean.gif",
    "box": (20, 112, 970, 315),
    "first": 0,
    "loops": 1,
    "title": "Xenopus embryos \u00b7 Ca\u00b2\u207a waves",
    "sub": "a wave jumps to the untouched neighbour; tracked vectors in magenta",
    "accent": (217, 70, 239),
}
FACE = {
    "path": MEDIA / "face_of_a_frog_timelapse.gif",
    "box": None,
    "first": 8,
    "loops": 1,
    "title": "Face of a frog",
    "sub": "the bioelectric pre-pattern of the face appears",
    "accent": (244, 63, 94),
}
PANELS = [GALVANI, WAVES, FACE]

FOOTER = "Bioelectricity from Galvani's frog leg to the face of a frog \u00b7 github.com/fractastical/morpholib"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def _source_size(panel: dict) -> tuple[int, int]:
    if panel["box"] is not None:
        x0, y0, x1, y1 = panel["box"]
        return x1 - x0, y1 - y0
    return Image.open(panel["path"]).size


def _load_frames(panel: dict, size: tuple[int, int]) -> tuple[list[Image.Image], list[int]]:
    im = Image.open(panel["path"])
    frames, durations = [], []
    for i in range(panel["first"], im.n_frames):
        im.seek(i)
        fr = im.convert("RGB")
        if panel["box"] is not None:
            fr = fr.crop(panel["box"])
        frames.append(fr.resize(size, Image.LANCZOS))
        durations.append(im.info.get("duration", 100) or 100)
    return frames, durations


def _frame_at(t_ms: float, durations: list[int], loops: int) -> int:
    period = LOOP_MS / loops
    t = (t_ms % period) / period * sum(durations)
    acc = 0
    for i, d in enumerate(durations):
        acc += d
        if t < acc:
            return i
    return len(durations) - 1


def layout() -> tuple[int, int, dict[int, tuple[int, int, int, int]]]:
    """Return canvas (W, H) and each panel's image rect (x, y, w, h); a label sits above each rect."""
    inner_w = CANVAS_W - 2 * MARGIN
    gw, gh = _source_size(GALVANI)
    fw, fh = _source_size(FACE)
    ww, wh = _source_size(WAVES)

    top_h = round((inner_w - GUTTER) / (gw / gh + fw / fh))
    galvani_w = round(top_h * gw / gh)
    face_w = inner_w - GUTTER - galvani_w
    waves_h = round(inner_w * wh / ww)

    top_y = MARGIN + LABEL_H
    waves_y = top_y + top_h + GUTTER + LABEL_H
    rects = {
        id(GALVANI): (MARGIN, top_y, galvani_w, top_h),
        id(FACE): (MARGIN + galvani_w + GUTTER, top_y, face_w, top_h),
        id(WAVES): (MARGIN, waves_y, inner_w, waves_h),
    }
    return CANVAS_W, waves_y + waves_h + FOOTER_H, rects


def build() -> Path:
    W, H, rects = layout()
    title_font, sub_font, foot_font = _font(20, bold=True), _font(14), _font(14)

    base = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(base)
    sources = []
    for panel in PANELS:
        x, y, w, h = rects[id(panel)]
        draw.text((x + 2, y - LABEL_H + 2), panel["title"], font=title_font, fill=FG)
        draw.text((x + 2, y - LABEL_H + 28), panel["sub"], font=sub_font, fill=SUB)
        draw.rectangle([x, y - 6, x + w - 1, y - 3], fill=panel["accent"])
        sources.append((panel, (x, y), _load_frames(panel, (w, h))))
    draw.text((W / 2, H - FOOTER_H / 2 - 2), FOOTER, font=foot_font, fill=SUB, anchor="mm")

    n_out = LOOP_MS // FRAME_MS
    rgb = []
    for k in range(n_out):
        canvas = base.copy()
        for panel, origin, (frames, durations) in sources:
            canvas.paste(frames[_frame_at(k * FRAME_MS, durations, panel["loops"])], origin)
        rgb.append(canvas)

    # One shared palette keeps static pixels at identical indices, so the GIF optimizer can skip them.
    samples = rgb[:: max(1, n_out // 8)]
    strip = Image.new("RGB", (W, H * len(samples)))
    for i, fr in enumerate(samples):
        strip.paste(fr, (0, i * H))
    palette = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    out = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in rgb]

    out[0].save(OUT_GIF, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1)
    return OUT_GIF


def self_test() -> int:
    checks = []
    for panel in PANELS:
        name = panel["path"].name
        im = Image.open(panel["path"])
        checks.append((f"{name} exists and animates", im.n_frames > panel["first"] + 1))
        if panel["box"] is not None:
            box = panel["box"]
            checks.append((f"{name} crop inside frame", box[2] <= im.width and box[3] <= im.height))
    W, H, rects = layout()
    gx, gy, gw, gh = rects[id(GALVANI)]
    fx, fy, fw, fh = rects[id(FACE)]
    wx, wy, ww, wh = rects[id(WAVES)]
    checks.append(("top-row panels share a height", gy == fy and gh == fh))
    checks.append(("face panel keeps its aspect within 2%", abs(fw / fh - _source_size(FACE)[0] / _source_size(FACE)[1]) < 0.02 * fw / fh))
    checks.append(("canvas fits all panels", fx + fw + MARGIN == W and wx + ww + MARGIN == W and wy + wh + FOOTER_H == H))
    durations = [100] * 18
    checks.append(("three loops hit last frame three times", [_frame_at(t, durations, 3) for t in (2799, 5599, 8399)] == [17] * 3))
    checks.append(("loop starts at frame 0", _frame_at(0, durations, 1) == 0))
    ok = True
    for label, passed in checks:
        print(("PASS " if passed else "FAIL ") + label)
        ok &= passed
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Check sources, layout, and frame mapping, then exit")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    path = build()
    im = Image.open(path)
    print(f"wrote {path.relative_to(ROOT)}  {im.size[0]}x{im.size[1]}  {im.n_frames} frames  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
