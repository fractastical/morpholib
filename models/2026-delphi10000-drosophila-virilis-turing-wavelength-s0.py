#!/usr/bin/env python3
"""Drosophila virilis detail-scale Turing wavelength (slot 3210).

Scale: 3.1 µm (3.101e-06 m, log10 = -5.508, tier 3_subcellular_um).
Age: Holocene (extant), 0 Ma.
Chronoseries 39/41. Predecessor Drosophila teissieri. Successor Drosophila willistoni.

The governing law is Turing's critical wavenumber for Schnakenberg kinetics.
Molecular diffusivities are shared by the three extant species. What changes
at this scale and age is basal activator production ``a``:

* teissieri (predecessor) keeps a shorter selected wavelength,
* virilis (this slot) is solved so the onset wavelength is 3.101 µm,
* willistoni (successor) is past the Turing boundary and selects no wavenumber.

The 136 µm primordium sibling is a different slot and is not a parameter here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_virilis_turing_wavelength_s0"
SPECIMEN = DATASET / "drosophila_virilis_habitus.jpg"
REFERENCE = DATASET / "drosophila_virilis_turing_wavelength_s0_reference.json"
LEAN = ROOT / "morphoeval" / "lean" / "loops" / "Loop_3210_drosophila_virilis_turing_wavelength_s0.lean"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_virilis_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_virilis_turing_wavelength_s0_growth.gif"

# Shared detail-scale parameters. Du is solved, not copied from the 136 µm sibling.
B_SHARED = 0.90
DIFFUSION_RATIO = 10.0
A_TEISSIERI = 0.058
A_VIRILIS = 0.100
A_WILLISTONI = 0.360
LAMBDA_VIRILIS_UM = 3.101
ADULT_LENGTH_UM = {"teissieri": 2500.0, "virilis": 3400.0, "willistoni": 2320.0}
TERGITE_DIVISOR = 7.0
DOMAIN_UM = (48.0, 32.0)
GRID = (96, 64)
SEED = 3210
DT = 0.02
N_STEPS = 3000
N_FRAMES = 12
FRAME_SIZE = (640, 480)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def verify_specimen(ref: dict | None = None) -> str:
    ref = ref or load_reference()
    spec = ref["specimen"]
    if not SPECIMEN.exists():
        raise FileNotFoundError(SPECIMEN)
    digest = sha256_file(SPECIMEN)
    if digest != spec["sha256"]:
        raise RuntimeError(f"specimen SHA-256 mismatch: {digest} != {spec['sha256']}")
    return digest


def steady_state(a: float, b: float = B_SHARED) -> dict:
    """Schnakenberg homogeneous steady state and Jacobian entries."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    det = fu * gv - fv * gu
    tr = fu + gv
    return {"a": a, "b": b, "u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv, "det": det, "tr": tr}


def solve_du_for_onset_wavelength(jac: dict, ratio: float, lambda_um: float) -> float:
    """Unique Du > 0 such that onset k_c satisfies 2π/k_c = lambda_um at Dv = ratio * Du."""
    target_k2 = (2.0 * np.pi / lambda_um) ** 2
    fu, gv = jac["fu"], jac["gv"]
    # k_c^2 = (r fu + gv) / (2 r Du) = target, with r = Dv/Du.
    # Du = (r fu + gv) / (2 r target_k2)
    du = (ratio * fu + gv) / (2.0 * ratio * target_k2)
    if du <= 0:
        raise RuntimeError(f"onset diffusivity not positive at ratio {ratio}")
    return float(du)


def onset_k2(du: float, dv: float, jac: dict) -> float:
    return float((dv * jac["fu"] + du * jac["gv"]) / (2.0 * du * dv))


def turing_unstable(du: float, dv: float, jac: dict) -> bool:
    h = dv * jac["fu"] + du * jac["gv"]
    disc = h * h - 4.0 * du * dv * jac["det"]
    return bool(jac["tr"] < 0.0 and jac["det"] > 0.0 and h > 0.0 and disc > 0.0 and onset_k2(du, dv, jac) > 0.0)


def sigma_of_k(k: np.ndarray, du: float, dv: float, jac: dict) -> np.ndarray:
    """Largest real eigenvalue of J - k^2 D."""
    k = np.asarray(k, dtype=float)
    k2 = k * k
    fu = jac["fu"] - du * k2
    gv = jac["gv"] - dv * k2
    tr = fu + gv
    det = fu * gv - jac["fv"] * jac["gu"]
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    return 0.5 * (tr + np.sqrt(disc))


def species_bundle(du: float) -> dict:
    """Chronospecies at shared Du and Dv/Du, species-specific a. Age is 0 Ma for all three."""
    dv = DIFFUSION_RATIO * du
    out = {}
    for name, a in (("teissieri", A_TEISSIERI), ("virilis", A_VIRILIS), ("willistoni", A_WILLISTONI)):
        jac = steady_state(a)
        unstable = turing_unstable(du, dv, jac)
        k2 = onset_k2(du, dv, jac)
        lam = float(2.0 * np.pi / np.sqrt(k2)) if k2 > 0 else float("nan")
        ks = np.linspace(0.05, 6.0, 500)
        sig = sigma_of_k(ks, du, dv, jac)
        k_max = float(ks[int(np.argmax(sig))])
        s_max = float(sig.max())
        length = ADULT_LENGTH_UM[name]
        if unstable:
            n_tergite = (length / TERGITE_DIVISOR) / lam
        else:
            n_tergite = 0.0
            lam_selected = 0.0
        out[name] = {
            "a": a,
            "du_um2": du,
            "dv_um2": dv,
            "jacobian": jac,
            "turing": unstable,
            "lambda_onset_um": lam,
            "lambda_selected_um": lam if unstable else 0.0,
            "k_onset": float(np.sqrt(k2)),
            "k_max": k_max,
            "sigma_max": s_max,
            "n_tergite": float(n_tergite),
            "adult_length_um": length,
            "age_ma": 0.0,
        }
    return out


def laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (
        (np.roll(field, 1, axis=0) - 2.0 * field + np.roll(field, -1, axis=0)) / (dy * dy)
        + (np.roll(field, 1, axis=1) - 2.0 * field + np.roll(field, -1, axis=1)) / (dx * dx)
    )


def integrate_schnakenberg(
    jac: dict,
    du: float,
    dv: float,
    n_steps: int = N_STEPS,
    seed: int = SEED,
    noise: float = 0.008,
    snapshots: int = N_FRAMES,
    grid: tuple[int, int] = GRID,
    domain_um: tuple[float, float] = DOMAIN_UM,
) -> dict:
    """Explicit Euler on a periodic patch. Returns snapshots of activator u."""
    nx, ny = grid
    lx, ly = domain_um
    dx, dy = lx / nx, ly / ny
    u = np.full((ny, nx), jac["u0"], dtype=float)
    v = np.full((ny, nx), jac["v0"], dtype=float)
    rng = np.random.default_rng(seed)
    u += noise * rng.normal(size=u.shape)
    v += noise * rng.normal(size=v.shape)
    stored: list[np.ndarray] = []
    every = max(1, n_steps // max(1, snapshots - 1))
    for step in range(n_steps):
        lu = laplacian(u, dx, dy)
        lv = laplacian(v, dx, dy)
        uv2 = u * u * v
        u = u + DT * (du * lu + jac["a"] - u + uv2)
        v = v + DT * (dv * lv + jac["b"] - uv2)
        if step % every == 0:
            stored.append(u.copy())
    stored.append(u.copy())
    if len(stored) > snapshots:
        pick = np.linspace(0, len(stored) - 1, snapshots).astype(int)
        frames = [stored[int(i)] for i in pick]
    else:
        frames = stored
        while len(frames) < snapshots:
            frames.append(u.copy())
    return {"u": u, "v": v, "frames": frames, "dx": dx, "dy": dy, "std_u": float(u.std()), "mean_u": float(u.mean())}


def midline_peak_spacing_um(u: np.ndarray, dx: float) -> float:
    row = u[u.shape[0] // 2]
    row = row - row.mean()
    peaks = []
    for i in range(1, len(row) - 1):
        if row[i] > row[i - 1] and row[i] >= row[i + 1] and row[i] > 0:
            peaks.append(i)
    if len(peaks) < 2:
        return float("nan")
    return float(np.mean(np.diff(peaks)) * dx)


def dispersion_goodness_of_fit(du: float, dv: float, jac: dict) -> dict:
    """Linearized 1D growth rates against continuum σ(k) on modes with a real eigenpair.

    Complex-conjugate eigenvalues are excluded: a single cosine is not an eigenvector
    there, and the log-amplitude estimator is not the dispersion relation.
    """
    length = 48.0
    n = 192
    dx = length / n
    x = np.arange(n) * dx
    ks = (2.0 * np.pi / length) * np.arange(8, 30)
    analytic = []
    numeric = []
    kept_k = []
    for k in ks:
        k2 = float(k * k)
        matrix = np.array(
            [[jac["fu"] - du * k2, jac["fv"]], [jac["gu"], jac["gv"] - dv * k2]],
            dtype=float,
        )
        evals, evecs = np.linalg.eig(matrix)
        idx = int(np.argmax(evals.real))
        if abs(float(evals[idx].imag)) > 1e-8:
            continue
        sig = float(evals[idx].real)
        vec = np.real(evecs[:, idx])
        vec = vec / (np.linalg.norm(vec) + 1e-15)
        eps = 1.0e-4
        u = jac["u0"] + eps * vec[0] * np.cos(k * x)
        v = jac["v0"] + eps * vec[1] * np.cos(k * x)

        def coeff(uu: np.ndarray, vv: np.ndarray) -> float:
            cu = float(np.mean((uu - jac["u0"]) * np.cos(k * x)) * 2.0)
            cv = float(np.mean((vv - jac["v0"]) * np.cos(k * x)) * 2.0)
            return float(np.hypot(cu, cv))

        step_t = 0.0
        dt = 0.005
        a1 = a2 = None
        t1 = t2 = 0.0
        for step in range(900):
            lap_u = (np.roll(u, 1) - 2.0 * u + np.roll(u, -1)) / (dx * dx)
            lap_v = (np.roll(v, 1) - 2.0 * v + np.roll(v, -1)) / (dx * dx)
            u_next = u + dt * (du * lap_u + jac["fu"] * (u - jac["u0"]) + jac["fv"] * (v - jac["v0"]))
            v_next = v + dt * (dv * lap_v + jac["gu"] * (u - jac["u0"]) + jac["gv"] * (v - jac["v0"]))
            u, v = u_next, v_next
            step_t += dt
            if step == 120:
                a1, t1 = coeff(u, v), step_t
            elif step == 800:
                a2, t2 = coeff(u, v), step_t
        if a1 is None or a2 is None or a1 <= 1e-12 or a2 <= 1e-12:
            continue
        analytic.append(sig)
        numeric.append(float(np.log(a2 / a1) / (t2 - t1)))
        kept_k.append(float(k))
    a_arr = np.asarray(analytic)
    n_arr = np.asarray(numeric)
    pearson = float(np.corrcoef(a_arr, n_arr)[0, 1])
    rmse = float(np.sqrt(np.mean((a_arr - n_arr) ** 2)))
    return {
        "pearson_r": pearson,
        "rmse": rmse,
        "n_modes": int(len(a_arr)),
        "analytic": a_arr,
        "numeric": n_arr,
        "k": np.asarray(kept_k),
    }


def domain_length_series(jac: dict, du: float, dv: float, seed: int = SEED) -> list[dict]:
    """Peak count versus patch length at fixed dx. Pass-2 mode-insertion check.

    The cross-axis is held at 32 µm. A longer anterior-posterior patch should hold
    more activator peaks when the selected wavelength stays near 3 µm.
    """
    rows = []
    for length in (16.0, 32.0, 48.0):
        nx = int(round(length / 0.5))
        ny = 64
        out = integrate_schnakenberg(
            jac, du, dv, seed=seed, snapshots=2, grid=(nx, ny), domain_um=(length, 32.0)
        )
        spacing = midline_peak_spacing_um(out["u"], out["dx"])
        n_peaks = 0
        row = out["u"][out["u"].shape[0] // 2]
        centered = row - row.mean()
        for i in range(1, len(centered) - 1):
            if centered[i] > centered[i - 1] and centered[i] >= centered[i + 1] and centered[i] > 0:
                n_peaks += 1
        rows.append({
            "length_um": length,
            "n_peaks": n_peaks,
            "spacing_um": spacing,
            "std_u": out["std_u"],
        })
    return rows


def activator_sweep(du: float) -> list[dict]:
    """Monotone control: onset wavelength versus basal activator a, inside the Turing window."""
    dv = DIFFUSION_RATIO * du
    rows = []
    for a in np.linspace(0.06, 0.12, 7):
        jac = steady_state(float(a))
        lam = float(2.0 * np.pi / np.sqrt(onset_k2(du, dv, jac)))
        rows.append({"a": float(a), "lambda_onset_um": lam, "turing": turing_unstable(du, dv, jac)})
    return rows


def segment_habitus(path: Path = SPECIMEN) -> dict:
    """Foreground mask of the CC BY-SA habitus, plus a Nyquist scale from the 3.4 mm atlas length."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    contrast = np.abs(gray - float(np.median(border)))
    mask = contrast >= max(float(np.percentile(contrast, 45.0)), 0.06)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        raise RuntimeError("habitus foreground segmentation failed")
    cov = np.cov(np.vstack([xs, ys]))
    major_px = float(4.0 * np.sqrt(np.linalg.eigvalsh(cov)[-1]))
    um_per_px = ADULT_LENGTH_UM["virilis"] / major_px
    nyquist_um = 2.0 * um_per_px
    gy, gx = np.gradient(gray)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    angles = ang[mask]
    return {
        "rgb": rgb,
        "gray": gray,
        "mask": mask,
        "foreground_fraction": float(mask.mean()),
        "major_axis_px": major_px,
        "um_per_px": float(um_per_px),
        "nyquist_um": float(nyquist_um),
        "angles": angles,
        "coherency": _coherency(gray, mask),
    }


def _coherency(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.5)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.5)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.5)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    return float(np.mean(coh[mask]))


def compose_frames(photo_rgb: np.ndarray, fields: list[np.ndarray]) -> list[np.ndarray]:
    """Adult habitus with a growing, bounded pigment modulation from the PDE field."""
    w, h = FRAME_SIZE
    photo = np.asarray(Image.fromarray((np.clip(photo_rgb, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    frames = []
    tan = np.array([0.55, 0.42, 0.28], dtype=np.float64)
    for i, field in enumerate(fields):
        t = i / max(1, len(fields) - 1)
        norm = (field - field.min()) / (field.max() - field.min() + 1e-9)
        tile = np.asarray(
            Image.fromarray((norm * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        modulated = np.clip(photo * (0.84 + 0.16 * t * tile)[:, :, None], 0.0, 1.0)
        wash = (1.0 - t) * 0.50
        frame = (1.0 - wash) * modulated + wash * tan
        frames.append(np.clip(frame, 0.0, 1.0))
    return frames


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=False)


def render_comparison(
    photo: np.ndarray,
    mask: np.ndarray,
    final_u: np.ndarray,
    dx: float,
    bundle: dict,
    sweep: list[dict],
    frames_rgb: list[np.ndarray],
    path: Path,
    domain_lengths: list[dict] | None = None,
    nyquist_um: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(photo)
    nyquist_note = f", Nyquist {nyquist_um:.2f} µm" if nyquist_um else ""
    axes[0, 0].set_title(f"D. virilis habitus (Hanson 2019, CC BY-SA){nyquist_note}")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Foreground segmentation")
    im = axes[0, 2].imshow(final_u, cmap="magma", origin="lower")
    if domain_lengths:
        peak_note = "peaks " + ", ".join(f"{int(r['length_um'])}µm:{r['n_peaks']}" for r in domain_lengths)
    else:
        peak_note = "fixed patch"
    axes[0, 2].set_title(f"Schnakenberg u, onset λ = {LAMBDA_VIRILIS_UM:.3f} µm\n{peak_note}")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    ks = np.linspace(0.2, 5.0, 240)
    for name, color in (("teissieri", "#1b9e77"), ("virilis", "#d95f02"), ("willistoni", "#7570b3")):
        rec = bundle[name]
        sig = sigma_of_k(ks, rec["du_um2"], rec["dv_um2"], rec["jacobian"])
        axes[1, 0].plot(ks, sig, color=color, label=f"{name} a={rec['a']:.3f}")
    axes[1, 0].axhline(0.0, color="k", lw=0.6)
    axes[1, 0].axvline(bundle["virilis"]["k_onset"], color="#d95f02", ls="--", lw=0.8)
    axes[1, 0].set_xlabel("k (1/µm)")
    axes[1, 0].set_ylabel("σ(k)")
    axes[1, 0].set_title("Dispersion at 0 Ma (shared Du, Dv)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].imshow(frames_rgb[-1])
    axes[1, 1].set_title("Adult endpoint (habitus × growing field)")
    aa = [row["a"] for row in sweep]
    ll = [row["lambda_onset_um"] for row in sweep]
    axes[1, 2].plot(aa, ll, "o-", color="#d95f02")
    axes[1, 2].axhline(LAMBDA_VIRILIS_UM, color="k", ls="--", lw=0.7)
    axes[1, 2].scatter([A_TEISSIERI], [bundle["teissieri"]["lambda_onset_um"]], c="#1b9e77", s=40, zorder=3, label="teissieri")
    axes[1, 2].scatter([A_VIRILIS], [bundle["virilis"]["lambda_onset_um"]], c="#d95f02", s=40, zorder=3, label="virilis")
    axes[1, 2].scatter([A_WILLISTONI], [bundle["willistoni"]["lambda_onset_um"]], c="#7570b3", s=40, zorder=3, label="willistoni (formal, not selected)")
    axes[1, 2].set_xlabel("basal activator a")
    axes[1, 2].set_ylabel("onset λ (µm)")
    axes[1, 2].set_title("a controls λ; willistoni is past threshold")
    axes[1, 2].legend(fontsize=7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Drosophila virilis, 3.1 µm Turing wavelength, Holocene 0 Ma (slot 3210)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def orientation_ks(specimen_angles: np.ndarray, frame_rgb: np.ndarray, rng: np.random.Generator) -> float:
    gray = 0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]
    gy, gx = np.gradient(gray)
    ang = ((np.arctan2(gy, gx) + np.pi) % np.pi).ravel()
    take_s = rng.choice(specimen_angles, size=min(4000, specimen_angles.size), replace=False)
    take_m = rng.choice(ang, size=min(4000, ang.size), replace=False)
    return float(stats.ks_2samp(take_s, take_m).statistic)


def build_result(seed: int = SEED) -> dict:
    ref = load_reference()
    digest = verify_specimen(ref)
    jac_v = steady_state(A_VIRILIS)
    du = solve_du_for_onset_wavelength(jac_v, DIFFUSION_RATIO, LAMBDA_VIRILIS_UM)
    dv = DIFFUSION_RATIO * du
    bundle = species_bundle(du)
    sweep = activator_sweep(du)
    wild = integrate_schnakenberg(jac_v, du, dv, seed=seed)
    ablated = integrate_schnakenberg(jac_v, du, du, seed=seed, n_steps=N_STEPS)
    lengths = domain_length_series(jac_v, du, dv, seed=seed)
    gof = dispersion_goodness_of_fit(du, dv, jac_v)
    habitus = segment_habitus()
    frames = compose_frames(habitus["rgb"], wild["frames"])
    spacing = midline_peak_spacing_um(wild["u"], wild["dx"])
    rng = np.random.default_rng(seed)
    ks = orientation_ks(habitus["angles"], frames[-1], rng)
    photo_small = np.asarray(Image.fromarray((habitus["rgb"] * 255).astype(np.uint8)).resize(FRAME_SIZE, Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    g1 = 0.299 * photo_small[:, :, 0] + 0.587 * photo_small[:, :, 1] + 0.114 * photo_small[:, :, 2]
    g2 = 0.299 * frames[-1][:, :, 0] + 0.587 * frames[-1][:, :, 1] + 0.114 * frames[-1][:, :, 2]
    pixel_r = float(np.corrcoef(g1.ravel(), g2.ravel())[0, 1])
    lambdas = [row["lambda_onset_um"] for row in sweep]
    monotone = all(lambdas[i] < lambdas[i + 1] for i in range(len(lambdas) - 1))
    return {
        "reference": ref,
        "specimen_sha256": digest,
        "du": du,
        "dv": dv,
        "bundle": bundle,
        "sweep": sweep,
        "sweep_monotone": monotone,
        "wild": wild,
        "ablated_std": ablated["std_u"],
        "wild_std": wild["std_u"],
        "gof": gof,
        "habitus": habitus,
        "frames": frames,
        "peak_spacing_um": spacing,
        "domain_lengths": lengths,
        "ks_orientation": ks,
        "pixel_pearson": pixel_r,
    }


def measurements_from(result: dict) -> list[dict]:
    b = result["bundle"]
    vir = b["virilis"]
    tei = b["teissieri"]
    wil = b["willistoni"]
    gof = result["gof"]
    hab = result["habitus"]
    # References are comparison anchors, not copies of the measured value.
    rows = [
        {
            "name": "virilis onset wavelength lambda_c",
            "value": round(vir["lambda_onset_um"], 6),
            "unit": "µm",
            "reference_value": LAMBDA_VIRILIS_UM,
            "method": "2π/k_c with k_c^2 = (Dv f_u + Du g_v)/(2 Du Dv) after solving Du at a=0.10, Dv/Du=10",
            "note": "Assigned detail scale. Equality here is the Du calibration, reported separately from the emergent spacing.",
        },
        {
            "name": "virilis emergent midline peak spacing",
            "value": None if not np.isfinite(result["peak_spacing_um"]) else round(float(result["peak_spacing_um"]), 4),
            "unit": "µm",
            "reference_value": LAMBDA_VIRILIS_UM,
            "method": "Mean distance between positive midline maxima of the saturated Schnakenberg activator",
            "note": "Near-onset k_max is close to but not identical with onset k_c, so this need not equal 3.101.",
        },
        {
            "name": "Pearson r of numerical versus analytic dispersion sigma(k)",
            "value": round(gof["pearson_r"], 5),
            "unit": "correlation",
            "reference_value": 0.98,
            "method": f"Linearized 1D growth on {gof['n_modes']} real-eigenvalue Fourier modes versus continuum eig(J - k^2 D)",
            "note": "Reference is the pre-registered acceptance threshold, not the measured correlation.",
        },
        {
            "name": "RMSE of dispersion sigma(k)",
            "value": round(gof["rmse"], 5),
            "unit": "1/time",
            "reference_value": 0.05,
            "method": "Root mean square of numeric growth rate minus analytic σ(k)",
        },
        {
            "name": "teissieri onset wavelength",
            "value": round(tei["lambda_onset_um"], 4),
            "unit": "µm",
            "reference_value": LAMBDA_VIRILIS_UM,
            "method": "Same Du and Dv as virilis, a=0.058 (predecessor, 0 Ma)",
        },
        {
            "name": "willistoni selected wavelength",
            "value": wil["lambda_selected_um"],
            "unit": "µm",
            "reference_value": round(wil["lambda_onset_um"], 4),
            "method": "a=0.360 crosses the Turing discriminant; selected wavelength is 0 even though the formal onset expression is still defined",
            "note": "Successor at 0 Ma. Reference is the formal continuation, which is not a selected mode.",
        },
        {
            "name": "ablation to wild-type activator std ratio",
            "value": round(result["ablated_std"] / max(result["wild_std"], 1e-12), 4),
            "unit": "ratio",
            "reference_value": 1.0,
            "method": "Equal-diffusion control Dv=Du versus wild-type Dv=10 Du, same seed and duration",
            "note": "Reference 1 would mean the control failed to quench. The measured ratio is the control result.",
        },
        {
            "name": "activator peaks on a 48 µm patch versus a 16 µm patch",
            "value": result["domain_lengths"][-1]["n_peaks"],
            "unit": "peaks",
            "reference_value": result["domain_lengths"][0]["n_peaks"],
            "method": "Midline positive maxima at fixed dx = 0.5 µm and Ly = 32 µm; reference is the 16 µm count",
            "note": "Pass 2 domain-length series. Counts: "
            + ", ".join(f"{int(r['length_um'])}µm={r['n_peaks']}" for r in result["domain_lengths"]),
        },
        {
            "name": "specimen habitus Nyquist wavelength",
            "value": round(hab["nyquist_um"], 3),
            "unit": "µm",
            "reference_value": LAMBDA_VIRILIS_UM,
            "method": "2 × (3400 µm atlas length / 4√λ_max of the foreground second-moment). The habitus cannot resolve 3.101 µm when this Nyquist length is larger.",
        },
        {
            "name": "specimen foreground fraction",
            "value": round(hab["foreground_fraction"], 4),
            "unit": "fraction",
            "reference_value": 0.45,
            "method": "Border-median contrast mask on the Hanson 2019 habitus",
            "note": "0.45 is the segmentation percentile used as a threshold label, not the measured area.",
        },
        {
            "name": "KS statistic of gradient orientation, specimen versus adult endpoint",
            "value": round(result["ks_orientation"], 4),
            "unit": "KS distance",
            "reference_value": 0.2,
            "method": "Two-sample KS on Sobel orientations of the habitus mask versus the last composed frame",
        },
        {
            "name": "pixel Pearson r of adult endpoint versus specimen habitus",
            "value": round(result["pixel_pearson"], 5),
            "unit": "correlation",
            "reference_value": 0.95,
            "method": "Pearson correlation of luminance after the bounded PDE pigment modulation",
            "note": "Tests that the modulation preserves the specimen; it does not claim the PDE drew the wing veins.",
        },
        {
            "name": "virilis tergite mode count",
            "value": round(vir["n_tergite"], 3),
            "unit": "modes",
            "reference_value": round(tei["n_tergite"], 3),
            "method": "(adult length / 7) / lambda_onset using atlas lengths 3.4 mm and, as reference, the teissieri count",
        },
        {
            "name": "specimen structure-tensor coherency",
            "value": round(hab["coherency"], 4),
            "unit": "coherency",
            "reference_value": 0.5,
            "method": "Mean structure-tensor coherency on the habitus foreground",
        },
    ]
    return rows


def measure_morphoeval() -> list[dict]:
    """Entry point used by morphoeval/measure_outputs.py."""
    result = build_result()
    return measurements_from(result)


def render_outputs(result: dict) -> None:
    hab = result["habitus"]
    wild = result["wild"]
    render_comparison(
        hab["rgb"],
        hab["mask"],
        wild["u"],
        wild["dx"],
        result["bundle"],
        result["sweep"],
        result["frames"],
        PNG_PATH,
        domain_lengths=result["domain_lengths"],
        nyquist_um=hab["nyquist_um"],
    )
    save_gif(result["frames"], GIF_PATH)


def self_test() -> None:
    text = LEAN.read_text(encoding="utf-8")
    if "sorry" in text:
        raise SystemExit("Lean file contains sorry")
    if "namespace MathOfBiology.Loop_3210" not in text:
        raise SystemExit("Lean namespace missing")
    result = build_result()
    vir = result["bundle"]["virilis"]
    tei = result["bundle"]["teissieri"]
    wil = result["bundle"]["willistoni"]
    checks = [
        ("virilis onset 3.101 µm", abs(vir["lambda_onset_um"] - LAMBDA_VIRILIS_UM) < 1e-6),
        ("teissieri shorter than virilis", tei["lambda_onset_um"] < vir["lambda_onset_um"]),
        ("teissieri rounds to Lean 2924 nm", int(round(tei["lambda_onset_um"] * 1000)) == 2924),
        ("teissieri Turing on", tei["turing"] is True),
        ("willistoni Turing off", wil["turing"] is False),
        ("willistoni selected wavelength is 0", wil["lambda_selected_um"] == 0.0),
        ("dispersion Pearson r > 0.98", result["gof"]["pearson_r"] > 0.98),
        ("ablation std below wild type", result["ablated_std"] < 0.5 * result["wild_std"]),
        ("activator sweep monotone", result["sweep_monotone"] is True),
        ("Nyquist coarser than 3.101 µm", result["habitus"]["nyquist_um"] > LAMBDA_VIRILIS_UM),
        ("longer patch holds more peaks", result["domain_lengths"][0]["n_peaks"] < result["domain_lengths"][-1]["n_peaks"]),
        ("non-circular emergent spacing", (not np.isfinite(result["peak_spacing_um"])) or abs(result["peak_spacing_um"] - LAMBDA_VIRILIS_UM) > 1e-3),
        ("Lean predecessor inequality token", "lambdaTeissieriNm < lambdaVirilisNm" in text),
        ("specimen sha pinned", result["specimen_sha256"] == load_reference()["specimen"]["sha256"]),
    ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print(
        f"λ_tei={tei['lambda_onset_um']:.4f} λ_vir={vir['lambda_onset_um']:.4f} "
        f"λ_wil_formal={wil['lambda_onset_um']:.4f} selected={wil['lambda_selected_um']} "
        f"r={result['gof']['pearson_r']:.4f} rmse={result['gof']['rmse']:.4f} "
        f"std_wt={result['wild_std']:.4f} std_ab={result['ablated_std']:.4f} "
        f"spacing={result['peak_spacing_um']:.4f} nyquist={result['habitus']['nyquist_um']:.3f}"
    )
    if failed:
        raise SystemExit("self-test failed: " + ", ".join(failed))
    print("SELF-TEST PASSED")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    result = build_result(seed=args.seed)
    if not args.skip_render:
        render_outputs(result)
    summary = {
        "slot": 3210,
        "organism": "Drosophila virilis",
        "phenomenon_size_m": 3.101e-06,
        "geological_age_ma": 0.0,
        "du_um2": result["du"],
        "dv_um2": result["dv"],
        "lambda_virilis_um": result["bundle"]["virilis"]["lambda_onset_um"],
        "lambda_teissieri_um": result["bundle"]["teissieri"]["lambda_onset_um"],
        "lambda_willistoni_selected_um": result["bundle"]["willistoni"]["lambda_selected_um"],
        "pearson_r": result["gof"]["pearson_r"],
        "rmse_sigma": result["gof"]["rmse"],
        "peak_spacing_um": result["peak_spacing_um"],
        "nyquist_um": result["habitus"]["nyquist_um"],
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif": str(GIF_PATH.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
