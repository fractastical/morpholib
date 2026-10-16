"""D'Arcy Thompson's theory of transformations (On Growth and Form, 1917, ch. XVII).

Thompson drew one species on a Cartesian grid and showed that a related species
is obtained by deforming the *grid* — a shear, a fan of radii, a logarithmic
stretch — rather than by changing the parts individually.  This small script has
two modes:

    python3 thompson/1917-thompson.py                 # the original two-panel toy (opens a window)
    python3 thompson/1917-thompson.py --gif out.gif   # animated grid deformations in Thompson's style

The animation is schematic: a stylised fish outline on a square grid is carried
through three of Thompson's transformation families, each blended from the
identity (t = 0) to its full deformation (t = 1): an oblique shear (his
Scarus -> Pomacanthus figure), a fan of coordinates (Polyprion -> Pseudopriacanthus),
and a radial/polar expansion (Diodon -> Orthagoriscus, the sunfish).  It is a
demonstration of the method, not a fit to measured specimens.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt


def transform_points(x, y):
    # Example nonlinear transformation inspired by Thompson: quadratic warp
    # p(x,y) = x^2 + xy + y^2 + x + y; q(x,y) similar (adjust as needed)
    p = x**2 + x*y + y**2 + x + y
    q = x**2 + x*y + y**2 + x + y
    return p, q


def demo():
    # Generate a simple 'fish' shape as points (ellipse for body)
    theta = np.linspace(0, 2*np.pi, 100)
    original_x = 2 * np.cos(theta)  # Elongated ellipse
    original_y = np.sin(theta)

    # Apply transformation
    transformed_x, transformed_y = transform_points(original_x, original_y)

    # Plot original and transformed
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(original_x, original_y, 'b-')
    axs[0].set_title('Original Shape (e.g., Scarus)')
    axs[0].axis('equal')
    axs[1].plot(transformed_x, transformed_y, 'r-')
    axs[1].set_title('Transformed Shape (e.g., Pomacanthus)')
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


def shear(x, y, t):            # Scarus -> Pomacanthus: oblique coordinates
    return x + 0.8 * t * y, y


def fan(x, y, t):              # Polyprion -> Pseudopriacanthus: ordinates fan out posteriorly
    return x * (1 + 0.35 * t * y), y * (1 + 0.5 * t)


def radial(x, y, t):           # Diodon -> Orthagoriscus: polar, radii expanding with y
    r = np.hypot(x, y); th = np.arctan2(y, x)
    r2 = r * (1 + 0.6 * t * (1 - np.cos(th)) ** 1.0)
    return r2 * np.cos(th), r2 * np.sin(th) * (1 + 0.4 * t)


FAMILIES = [("oblique shear  (Scarus → Pomacanthus)", shear),
            ("fan of coordinates  (Polyprion → Pseudopriacanthus)", fan),
            ("radial expansion  (Diodon → Orthagoriscus)", radial)]


def draw_frame(ax, f, t, label):
    g = np.linspace(-2.5, 2.5, 11); h = np.linspace(-1.5, 1.5, 7)
    s = np.linspace(-2.5, 2.5, 60); u = np.linspace(-1.5, 1.5, 40)
    for gx in g:
        X, Y = f(np.full_like(u, gx), u, t); ax.plot(X, Y, color="#999", lw=0.6)
    for hy in h:
        X, Y = f(s, np.full_like(s, hy), t); ax.plot(X, Y, color="#999", lw=0.6)
    body, tail, fin = fish_outline()
    for part, col in ((body, "#1f4e79"), (tail, "#1f4e79"), (fin, "#1f4e79")):
        X, Y = f(part[:, 0], part[:, 1], t); ax.plot(X, Y, color=col, lw=2)
    ax.set_aspect("equal"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-2.6, 2.6); ax.axis("off")
    ax.set_title(f"{label}\nt = {t:.2f}", fontsize=10)


def make_gif(output, n_steps=20, fps=10):
    from PIL import Image
    frames = []
    for label, f in FAMILIES:
        ts = np.concatenate([np.linspace(0, 1, n_steps), np.full(fps, 1.0), np.linspace(1, 0, n_steps)[1:]])
        for t in ts:
            fig, ax = plt.subplots(figsize=(7, 4.6))
            draw_frame(ax, f, t, label)
            fig.suptitle("D'Arcy Thompson 1917: a related form is the same drawing on a deformed grid", fontsize=10.5)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
            plt.close(fig)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Saved animation to {output}  ({len(frames)} frames)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--gif":
        make_gif(sys.argv[2])
    else:
        demo()
