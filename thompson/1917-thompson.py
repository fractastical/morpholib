"""D'Arcy Thompson's theory of transformations (On Growth and Form, 1917, ch. XVII).

Thompson drew one species on a Cartesian grid and showed that a related species
is obtained by deforming the *grid* — a shear, a fan of radii, a logarithmic
stretch — rather than by changing the parts individually.  This small script has
two modes:

    python3 thompson/1917-thompson.py                 # the original two-panel toy (opens a window)
    python3 thompson/1917-thompson.py --gif out.gif   # animated grid deformations in Thompson's style

The animation carries a real fish - the silhouette of a Scarus ghobban photograph
(Wikimedia Commons, segmented by thompson/1917-thompson-scarus-pomacanthus.py; the
file is fetched and its SHA-256 checked there) on a square grid - through four of
Thompson's transformation families, each blended from the
identity (t = 0) to its full deformation (t = 1): the oblique shear of his
Argyropelecus -> Sternoptyx figure (Figs. 373-374, "axes inclined at an angle of
70 degrees"), the coaxial circles of Scarus -> Pomacanthus (Figs. 375-376), a fan
of coordinates (Polyprion -> Pseudopriacanthus, Figs. 377-378) and a radial/polar
expansion (Diodon -> Orthagoriscus, the sunfish, Figs. 381-382).  In the coaxial
panel the silhouette of a real Pomacanthus semicirculatus is drawn dashed as the
target.  The deformation strengths are Thompson's illustrative ones, not fitted to
the specimens; the fitted maps, scored against landmarks and outlines, are in
thompson/1917-thompson-scarus-pomacanthus.py.  If the photographs cannot be
fetched the script falls back to a stylised fish and says so in the title.

The two-panel toy applies the Scarus -> Pomacanthus coaxial-circle map to the
same silhouette.  (Until 2026-10-16 it applied the same polynomial to both coordinates,
p = q, which collapses every shape onto the line y = x; that was a bug, not a
transformation.)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Real silhouettes, borrowed from the sibling script (photographs pinned there)
# --------------------------------------------------------------------------
def _sibling():
    p = Path(__file__).with_name("1917-thompson-scarus-pomacanthus.py")
    spec = importlib.util.spec_from_file_location("thompson_scarus_pomacanthus", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def real_silhouette(key="Scarus_ghobban", allow_download=True, n=600):
    """Boundary of the segmented photograph, anterior to the RIGHT (as the grids below are oriented), centred,
    scaled to a body length of 4 (x in [-2, 2]).  None if the photograph is not available."""
    try:
        mod = _sibling()
        path = mod.fetch_photo(key, allow_download)
    except Exception as e:                                   # no network, or scikit-image missing
        print(f"  real silhouette unavailable ({e}); using the stylised fish", file=sys.stderr)
        return None
    if path is None:
        return None
    ph = mod.PHOTOS[key]; img = mod.load_photo(path)
    mask = mod.segment_white(img) if ph["kind"] == "white" else mod.segment_field(img, ph["box"], ph.get("clip_below"))
    if ph["faces"] == "left":
        mask = mask[:, ::-1]
    B = mod.boundary_points(mask, n)
    B = B - (B.max(0) + B.min(0)) / 2
    return B * (4.0 / (B[:, 0].max() - B[:, 0].min()))


def register(B, A):
    """Similarity (scale + translation, no rotation) matching B's bounding-box
    width and centre to A's. (Not a snout/tail-tip landmark registration.)"""
    sa, sb = A[:, 0].max() - A[:, 0].min(), B[:, 0].max() - B[:, 0].min()
    return (B - (B.max(0) + B.min(0)) / 2) * (sa / sb) + (A.max(0) + A.min(0)) / 2


def transform_points(x, y, c=0.18):
    """Thompson's Scarus -> Pomacanthus grid: the Cartesian lines become two pencils of coaxial
    circles, which is what the complex map z -> z / (1 - c z) does to them (a Moebius
    transformation: lines go to circles through the image of infinity, 1/c on the axis)."""
    z = x + 1j * y
    w = z / (1 - c * z)
    return w.real, w.imag


def demo():
    S = real_silhouette()
    if S is None:                                            # fall back: an ellipse standing in for Scarus
        theta = np.linspace(0, 2*np.pi, 100); S = np.column_stack([2 * np.cos(theta), np.sin(theta)])
        what = "an ellipse standing in for Scarus"
    else:
        what = "Scarus ghobban, silhouette of a photograph"
    original_x, original_y = S[:, 0], S[:, 1]

    # Apply transformation
    transformed_x, transformed_y = transform_points(original_x, original_y)

    # Plot original and transformed
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(original_x, original_y, 'b-')
    axs[0].set_title(f'Original shape ({what})')
    axs[0].axis('equal')
    axs[1].plot(transformed_x, transformed_y, 'r-')
    axs[1].set_title('Under the coaxial-circle map z -> z/(1 - cz), c = 0.18 (Pomacanthus-wise)')
    axs[1].axis('equal')
    plt.show()


# --------------------------------------------------------------------------
# Animated grid deformations
# --------------------------------------------------------------------------
def fish_outline(n=400):
    """A stylised fish: body ellipse with a forked tail and a dorsal fin, in a
    [-2, 2] x [-1, 1] box (x = anterior at the right, as Thompson drew them)."""
    t = np.linspace(0, 2 * np.pi, n)
    bx, by = 1.6 * np.cos(t), 0.8 * np.sin(t) * (1 - 0.15 * np.cos(t))
    body = np.column_stack([bx, by])
    tail = np.array([[-1.5, 0.15], [-2.1, 0.7], [-1.9, 0.0], [-2.1, -0.7], [-1.5, -0.15]])
    fin = np.array([[0.4, 0.75], [0.1, 1.05], [-0.5, 1.0], [-0.7, 0.75]])
    return body, tail, fin


def shear(x, y, t):            # Argyropelecus -> Sternoptyx: oblique coordinates (Thompson: axes at 70 deg)
    return x + np.tan(np.radians(20)) * t * y * 2.2, y


def coaxial(x, y, t):          # Scarus -> Pomacanthus: Cartesian lines -> coaxial circles (Moebius map)
    z = x + 1j * y
    w = z / (1 - 0.18 * t * z)
    return w.real, w.imag


def fan(x, y, t):              # Polyprion -> Pseudopriacanthus: ordinates fan out posteriorly
    return x * (1 + 0.35 * t * y), y * (1 + 0.5 * t)


def radial(x, y, t):           # Diodon -> Orthagoriscus: polar, radii expanding with y
    r = np.hypot(x, y); th = np.arctan2(y, x)
    r2 = r * (1 + 0.6 * t * (1 - np.cos(th)) ** 1.0)
    return r2 * np.cos(th), r2 * np.sin(th) * (1 + 0.4 * t)


FAMILIES = [("oblique shear  (Argyropelecus → Sternoptyx, Figs. 373–374)", shear),
            ("coaxial circles  (Scarus → Pomacanthus, Figs. 375–376)", coaxial),
            ("fan of coordinates  (Polyprion → Pseudopriacanthus, Figs. 377–378)", fan),
            ("radial expansion  (Diodon → Orthagoriscus, Figs. 381–382)", radial)]


def draw_frame(ax, f, t, label, parts, target=None):
    g = np.linspace(-2.5, 2.5, 11); h = np.linspace(-1.5, 1.5, 7)
    s = np.linspace(-2.5, 2.5, 60); u = np.linspace(-1.5, 1.5, 40)
    for gx in g:
        X, Y = f(np.full_like(u, gx), u, t); ax.plot(X, Y, color="#999", lw=0.6)
    for hy in h:
        X, Y = f(s, np.full_like(s, hy), t); ax.plot(X, Y, color="#999", lw=0.6)
    if target is not None and t > 0:                         # the real second species, as the target of this family
        ax.plot(target[:, 0], target[:, 1], color="#b91c1c", lw=1.1, ls="--", alpha=min(1.0, 1.4 * t))
    for part in parts:
        X, Y = f(part[:, 0], part[:, 1], t)
        ax.fill(X, Y, color="#1f4e79", alpha=0.12, lw=0) if len(parts) == 1 else None
        ax.plot(X, Y, color="#1f4e79", lw=2)
    ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-2.6, 2.6); ax.axis("off")
    ax.set_title(f"{label}\nt = {t:.2f}", fontsize=10)


def make_gif(output, n_steps=20, fps=10):
    from PIL import Image
    S = real_silhouette()
    if S is None:
        parts, target = list(fish_outline()), None
        head = "D'Arcy Thompson 1917: a related form is the same drawing on a deformed grid  (stylised fish: photographs unavailable)"
    else:
        parts = [S]
        P = real_silhouette("Pomacanthus_semicirculatus")
        # the target sits where the fully deformed Scarus lands (same body length and centre), so the eye can
        # compare the two shapes rather than their placement
        target = register(P, np.column_stack(coaxial(S[:, 0], S[:, 1], 1.0))) if P is not None else None
        head = ("D'Arcy Thompson 1917: a related form is the same drawing on a deformed grid\n"
                "solid: a real Scarus ghobban (silhouette of a photograph) · dashed (coaxial panel only): "
                "a real Pomacanthus semicirculatus")
    frames = []
    for label, f in FAMILIES:
        ts = np.concatenate([np.linspace(0, 1, n_steps), np.full(fps, 1.0), np.linspace(1, 0, n_steps)[1:]])
        for t in ts:
            fig, ax = plt.subplots(figsize=(7, 4.6))
            draw_frame(ax, f, t, label, parts, target if f is coaxial else None)
            fig.suptitle(head, fontsize=8.6)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
            plt.close(fig)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Saved animation to {output}  ({len(frames)} frames)")


def self_test() -> int:
    """Deterministic offline checks (Pass N+1: this script previously had no
    --self-test, so the flag silently launched the network demo). Every
    deformation family must be the identity at t=0, the Moebius map at c=0
    must be the identity, and register() must match bounding boxes."""
    rng = np.random.default_rng(1917)
    x, y = rng.normal(size=200), rng.normal(size=200)
    ok = True
    for label, f in FAMILIES:
        X, Y = f(x.copy(), y.copy(), 0.0)
        good = bool(np.allclose(X, x) and np.allclose(Y, y))
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {label.split('(')[0].strip()}: identity at t=0")
    X, Y = transform_points(x, y, c=0.0)
    good = bool(np.allclose(X, x) and np.allclose(Y, y))
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] coaxial Moebius map: identity at c=0")
    A = rng.normal(size=(50, 2)) * [3.0, 1.0] + [1.0, 2.0]
    B = rng.normal(size=(50, 2)) * [1.0, 2.0] - [2.0, 0.0]
    R = register(B, A)
    good = bool(np.isclose(R[:, 0].max() - R[:, 0].min(), A[:, 0].max() - A[:, 0].min())
                and np.allclose((R.max(0) + R.min(0)) / 2, (A.max(0) + A.min(0)) / 2))
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] register(): bounding-box width/centre matched")
    print(f"[{'PASS' if ok else 'FAIL'}] 1917-thompson self-test")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        raise SystemExit(self_test())
    if len(sys.argv) >= 3 and sys.argv[1] == "--gif":
        make_gif(sys.argv[2])
    else:
        demo()
