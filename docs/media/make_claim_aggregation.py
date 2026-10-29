#!/usr/bin/env python3
"""Tile one animation (or still) per Ingressing Minds claim into a 4×2 loop.

    venv/bin/python docs/media/make_claim_aggregation.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

HERE = Path(__file__).resolve().parent

# (file, claim, short name, score)
TILES = [
    ("zhang_sorting_morphogenesis.gif", "1", "Frozen cells", "66.7%"),
    ("zhang_13yo_fig2_two_rulers_illusion.png", "2", "Two rulers", "13.5%"),
    ("zhang_13yo_fig3_sports_cars_vs_pedestrians.png", "3", "Clustering", "86.5%"),
    ("thompson_transformations.gif", "4", "Thompson", "96.1%"),
    ("cervera_levin_mafe_gj_block.gif", "5", "Planarian", "95.0%"),
    ("xenopus_craniofacial_remap_growth.gif", "6", "Tadpole", "95.0%"),
    ("xenobot_evolutionary_biobots_growth.gif", "7", "Xenobot", "90.1%"),
]
MEAN = "77.6%"
BG, FG, DIM, ACCENT = (250, 250, 248), (15, 23, 42), (100, 116, 139), (2, 132, 199)
HEADER = (11, 19, 37)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib

    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(d / name), size)


def frames_of(path: Path, n: int) -> list[Image.Image]:
    im = Image.open(path)
    raw = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    if not raw:
        raw = [im.convert("RGB")]
    if len(raw) == 1:
        return raw * n
    idx = [round(i * (len(raw) - 1) / (n - 1)) for i in range(n)]
    return [raw[i] for i in idx]


def fit(frame: Image.Image, w: int, h: int) -> Image.Image:
    f = frame.copy()
    f.thumbnail((w, h), Image.LANCZOS)
    out = Image.new("RGB", (w, h), BG)
    out.paste(f, ((w - f.width) // 2, (h - f.height) // 2))
    return out


def label(tile: Image.Image, claim: str, name: str, score: str) -> Image.Image:
    d = ImageDraw.Draw(tile)
    w, h = tile.size
    d.rectangle([0, 0, w, 22], fill=HEADER)
    d.text((6, 4), f"{claim}  {name}", font=font(11, True), fill=(248, 250, 252))
    tw = d.textlength(score, font=font(11, True))
    d.text((w - tw - 6, 4), score, font=font(11, True), fill=(125, 211, 252))
    return tile


def mean_tile(w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), HEADER)
    d = ImageDraw.Draw(im)
    d.text((12, 18), "Ingressing Minds", font=font(13, True), fill=(148, 163, 184))
    d.text((12, 42), "seven claims,", font=font(16, True), fill=(248, 250, 252))
    d.text((12, 64), "each counted once", font=font(16, True), fill=(248, 250, 252))
    d.text((12, h - 48), MEAN, font=font(28, True), fill=(125, 211, 252))
    d.text((12, h - 18), "unweighted mean", font=font(11), fill=(148, 163, 184))
    return im


def main() -> None:
    n_frames, tw, th, gap, header_h = 36, 280, 168, 6, 30
    cols, rows = 4, 2
    clips = [frames_of(HERE / name, n_frames) for name, *_ in TILES]
    mean = mean_tile(tw, th)
    W = cols * tw + (cols + 1) * gap
    H = header_h + rows * th + (rows + 1) * gap
    frames = []
    for k in range(n_frames):
        canvas = Image.new("RGB", (W, H), (226, 232, 240))
        dh = ImageDraw.Draw(canvas)
        dh.rectangle([0, 0, W, header_h], fill=HEADER)
        dh.text(
            (10, 7),
            "Seven claims in Ingressing Minds  ·  click a heading in the article for the walkthrough",
            font=font(12, True),
            fill=(248, 250, 252),
        )
        for t in range(8):
            r, c = divmod(t, cols)
            x = gap + c * (tw + gap)
            y = header_h + gap + r * (th + gap)
            if t == 7:
                tile = mean
            else:
                _fn, claim, name, score = TILES[t]
                tile = label(fit(clips[t][k], tw, th), claim, name, score)
            canvas.paste(tile, (x, y))
        frames.append(canvas)

    strip = Image.new("RGB", (W, H * 4))
    for i, f in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
        strip.paste(f, (0, i * H))
    pal = strip.quantize(colors=128, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]
    out = HERE / "substack_seven_claims_aggregation.gif"
    q[0].save(out, save_all=True, append_images=q[1:], duration=140, loop=0, optimize=True)
    print(f"saved {out}  {W}x{H}, {len(q)} frames, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
