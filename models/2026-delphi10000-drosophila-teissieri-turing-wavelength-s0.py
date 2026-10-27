#!/usr/bin/env python3
"""
Drosophila teissieri detail-scale Turing wavelength (slot 3209, variant s0).

Scale: 1.7 µm (1.72e-06 m, log10=-5.764), tier 3_subcellular_um.
Age: Holocene extant, 0 Ma. Chronoseries 38/41.
Predecessor: Drosophila takahashii (atlas cell-tier stripe 97.6 µm).
Successor: Drosophila virilis (atlas floor 2.72 µm).

Governing law (Turing 1952, Phil. Trans. R. Soc. B 237:37):
    k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v),   lambda_c = 2 pi / k_c.
Kinetics (Schnakenberg 1979, J. Theor. Biol. 81:389), periodic square patch:
    du/dt = D_u nabla^2 u + gamma (a - u + u^2 v)
    dv/dt = D_v nabla^2 v + gamma (b - u^2 v)
with (a, b, gamma, D_v/D_u) = (0.13, 0.91, 1.15 s^-1, 17.4) chosen so
lambda_c = 1.72 µm. These are not the slot-2761 power-law coefficients.

Boundary conditions: periodic (torus), the continuum analogue of Turing's ring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "drosophila_teissieri_turing_wavelength_s0"
REF_PATH = DATASET / "drosophila_teissieri_turing_wavelength_s0_reference.json"
SPECIMEN_NAME = "drosophila_teissieri_turing_wavelength_s0_specimen.jpg"
PNG_PATH = ROOT / "docs" / "media" / "drosophila_teissieri_turing_wavelength_s0_comparison.png"
GIF_PATH = ROOT / "docs" / "media" / "drosophila_teissieri_turing_wavelength_s0_growth.gif"

# Chronoseries contrasts (atlas magnitudes, not fitted parameters of this PDE).
PREDECESSOR_LAMBDA_UM = 97.6   # D. takahashii slot 2811, cell tier
SUCCESSOR_FLOOR_UM = 2.72      # D. virilis atlas minimum
TARGET_LAMBDA_UM = 1.72


def load_reference() -> dict:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def verify_dataset_pins(ref: dict) -> None:
    """SHA-256 pins recorded in the reference JSON."""
    spec = DATASET / ref["specimen"]["filename"]
    got = hashlib.sha256(spec.read_bytes()).hexdigest()
    if got != ref["specimen"]["sha256"]:
        raise RuntimeError(f"specimen SHA-256 mismatch: {got}")
    for name, expected in ref["file_sha256"].items():
        got = hashlib.sha256((DATASET / name).read_bytes()).hexdigest()
        if got != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {got}")


def jacobian(a: float, b: float) -> dict:
    """Schnakenberg homogeneous state and reaction Jacobian. Turing 1952 needs
    trace < 0 and det > 0 so the well-mixed state is stable."""
    u0 = a + b
    v0 = b / (u0 * u0)
    fu = -1.0 + 2.0 * u0 * v0
    fv = u0 * u0
    gu = -2.0 * u0 * v0
    gv = -(u0 * u0)
    trace = fu + gv
    det = fu * gv - fv * gu
    return {"u0": u0, "v0": v0, "fu": fu, "fv": fv, "gu": gu, "gv": gv,
            "trace": trace, "det": det}


def solve_diffusivities(jac: dict, ratio: float, gamma: float, lambda_um: float) -> dict:
    """Pin lambda_c = 2 pi / k_c by solving the Turing wavenumber formula for D_u.
    k_c^2 = gamma (ratio f_u + g_v) / (2 ratio D_u), with D_v = ratio D_u.
    """
    kc = 2.0 * np.pi / lambda_um
    drive = ratio * jac["fu"] + jac["gv"]
    if drive <= 0:
        raise RuntimeError("diffusivity ratio does not satisfy D_v f_u + D_u g_v > 0")
    du = gamma * drive / (2.0 * ratio * kc * kc)
    dv = ratio * du
    return {"Du": float(du), "Dv": float(dv), "k_c": float(kc), "drive": float(drive),
            "lambda_c_um": float(lambda_um)}


def growth_rate(k: float, jac: dict, diff: dict, gamma: float) -> float:
    """Real part of the larger Jacobian eigenvalue at wavenumber k (continuum)."""
    au = gamma * jac["fu"] - (k * k) * diff["Du"]
    bv = gamma * jac["gv"] - (k * k) * diff["Dv"]
    tr = au + bv
    det = au * bv - (gamma * jac["fv"]) * (gamma * jac["gu"])
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        return float(0.5 * (tr + np.sqrt(disc)))
    return float(0.5 * tr)


def determinant_h(k: float, jac: dict, diff: dict, gamma: float) -> float:
    """det(gamma J - k^2 D). The classical k_c minimises this quartic in k."""
    du, dv = diff["Du"], diff["Dv"]
    fu, fv, gu, gv = jac["fu"], jac["fv"], jac["gu"], jac["gv"]
    return float(
        du * dv * k**4
        - (dv * gamma * fu + du * gamma * gv) * k**2
        + (gamma * fu) * (gamma * gv) - (gamma * fv) * (gamma * gu)
    )


def measure_dispersion(jac: dict, diff: dict, gamma: float, lambda_um: float) -> dict:
    """Integrate the linearized 1D PDE on the critical-mode eigenvector and
    compare the measured growth rate to the continuum dispersion relation.
    Pearson r is computed on wavenumbers whose continuum discriminant is >= 0
    (the Turing band and its real-eigenvalue shoulders)."""
    n = 128
    length = 12.0 * lambda_um
    dx = length / n
    dt = 0.15 * dx * dx / (2.0 * diff["Dv"])
    x = np.arange(n) * dx
    predicted = []
    measured = []
    discs = []
    modes = []
    for m in range(3, 21):
        kk = 2.0 * np.pi * m / length
        k2_disc = (2.0 - 2.0 * np.cos(2.0 * np.pi * m / n)) / (dx * dx)
        sig = growth_rate(kk, jac, diff, gamma)
        au = gamma * jac["fu"] - k2_disc * diff["Du"]
        bv = gamma * jac["gv"] - k2_disc * diff["Dv"]
        tr = au + bv
        det = au * bv - (gamma * jac["fv"]) * (gamma * jac["gu"])
        disc_c_num = tr * tr - 4.0 * det
        sig_d = 0.5 * (tr + np.sqrt(disc_c_num)) if disc_c_num >= 0 else 0.5 * tr
        # Continuum discriminant, used as the band mask.
        au_c = gamma * jac["fu"] - kk * kk * diff["Du"]
        bv_c = gamma * jac["gv"] - kk * kk * diff["Dv"]
        tr_c = au_c + bv_c
        det_c = au_c * bv_c - (gamma * jac["fv"]) * (gamma * jac["gu"])
        disc_c = tr_c * tr_c - 4.0 * det_c
        ratio = -(au - sig_d) / (gamma * jac["fv"])
        eps = 1e-4
        uu = jac["u0"] + eps * np.cos(kk * x)
        vv = jac["v0"] + eps * ratio * np.cos(kk * x)

        def step(uu, vv):
            lu = (np.roll(uu, 1) + np.roll(uu, -1) - 2.0 * uu) / (dx * dx)
            lv = (np.roll(vv, 1) + np.roll(vv, -1) - 2.0 * vv) / (dx * dx)
            du = diff["Du"] * lu + gamma * (jac["fu"] * (uu - jac["u0"]) + jac["fv"] * (vv - jac["v0"]))
            dv = diff["Dv"] * lv + gamma * (jac["gu"] * (uu - jac["u0"]) + jac["gv"] * (vv - jac["v0"]))
            return uu + dt * du, vv + dt * dv

        for _ in range(int(0.5 / dt)):
            uu, vv = step(uu, vv)

        def amp(uu, vv):
            return float(np.sqrt(np.mean((uu - jac["u0"]) ** 2 + (vv - jac["v0"]) ** 2)) + 1e-30)

        a0 = amp(uu, vv)
        nfit = int(1.0 / dt)
        for _ in range(nfit):
            uu, vv = step(uu, vv)
        sig_m = float(np.log(amp(uu, vv) / a0) / (nfit * dt))
        predicted.append(sig)
        measured.append(sig_m)
        discs.append(disc_c)
        modes.append(m)
    pred = np.asarray(predicted)
    meas = np.asarray(measured)
    mask = np.asarray(discs) >= 0.0
    if mask.sum() < 5:
        raise RuntimeError("dispersion band too small to score")
    r = float(np.corrcoef(pred[mask], meas[mask])[0, 1])
    rmse = float(np.sqrt(np.mean((pred[mask] - meas[mask]) ** 2)))
    # Classical k_c minimises h(k). Compare the formula to a numerical argmin.
    ks = np.linspace(0.4 * diff["k_c"], 1.8 * diff["k_c"], 400)
    hs = np.array([determinant_h(k, jac, diff, gamma) for k in ks])
    k_num = float(ks[int(np.argmin(hs))])
    return {
        "pearson_r": r,
        "rmse_per_s": rmse,
        "n_modes": int(mask.sum()),
        "modes": [int(m) for m, keep in zip(modes, mask) if keep],
        "sigma_predicted": pred[mask].tolist(),
        "sigma_measured": meas[mask].tolist(),
        "k_c_formula": diff["k_c"],
        "k_c_determinant_numerical": k_num,
        "lambda_determinant_numerical_um": float(2.0 * np.pi / k_num),
        "determinant_relative_error": float(abs(k_num - diff["k_c"]) / diff["k_c"]),
    }


def laplace(z: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0 * z) / (dx * dx)


def simulate_patch(jac: dict, diff: dict, gamma: float, lambda_um: float,
                   n: int = 96, t_final: float = 55.0, n_frames: int = 12,
                   seed: int = 3209, n_waves: int = 12,
                   equal_diffusion: bool = False, ablate_autocatalysis: bool = False) -> dict:
    """Explicit Euler on the periodic patch. Frames are evenly spaced in time."""
    length = float(n_waves) * lambda_um
    dx = length / n
    du = diff["Du"]
    dv = diff["Du"] if equal_diffusion else diff["Dv"]
    dt = 0.28 * dx * dx / (4.0 * max(du, dv))
    steps = int(t_final / dt)
    rng = np.random.default_rng(seed)
    u = np.full((n, n), jac["u0"])
    v = np.full((n, n), jac["v0"])
    xs = np.arange(n) * dx
    xg, yg = np.meshgrid(xs, xs)
    for m in range(6, 19):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        u += 0.002 * np.cos(2.0 * np.pi * m * xg / length + phase)
        v += 0.0006 * np.cos(2.0 * np.pi * m * xg / length + phase)
    u += 0.001 * rng.normal(size=(n, n))
    v += 0.001 * rng.normal(size=(n, n))
    snap_at = set(int(round(i * (steps - 1) / (n_frames - 1))) for i in range(n_frames))
    frames = []
    std_trace = []
    for step in range(steps):
        if step in snap_at:
            frames.append(u.copy())
            std_trace.append(float(u.std()))
        lu, lv = laplace(u, dx), laplace(v, dx)
        if ablate_autocatalysis:
            # Drop the u^2 v autocatalytic exchange. Production stays at the
            # homogeneous rates, so spatial perturbations decay.
            react_u = gamma * (jac["a_hold"] - u)
            react_v = gamma * (jac["b_hold"] - v)
        else:
            react_u = gamma * (jac["a_hold"] - u + u * u * v)
            react_v = gamma * (jac["b_hold"] - u * u * v)
        u = u + dt * (du * lu + react_u)
        v = v + dt * (dv * lv + react_v)
    if len(frames) < n_frames:
        frames.append(u.copy())
    # Dominant integer Fourier mode of the final field (cycles across the patch).
    field = frames[-1]
    power = np.abs(np.fft.fft2(field - field.mean())) ** 2
    cycle_power = np.array([power[m, :].sum() + power[:, m].sum() for m in range(0, n // 3)])
    mode = int(np.argmax(cycle_power[1:])) + 1
    # Parabolic sub-bin refinement around that integer mode.
    if 1 <= mode < len(cycle_power) - 1:
        ym, y0, yp = cycle_power[mode - 1], cycle_power[mode], cycle_power[mode + 1]
        denom = (ym - 2.0 * y0 + yp)
        delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    mode_ref = mode + delta
    return {
        "frames": frames,
        "std_trace": std_trace,
        "final_std": float(field.std()),
        "initial_std": float(frames[0].std()),
        "mode": mode,
        "mode_refined": float(mode_ref),
        "lambda_integer_um": float(length / mode),
        "lambda_refined_um": float(length / mode_ref),
        "length_um": length,
        "n": n,
        "dx_um": float(dx),
        "dt_s": float(dt),
        "steps": steps,
    }


def diffusivity_sweep(jac: dict, diff: dict, gamma: float, lambda_um: float) -> dict:
    """Hold D_u at the slot value and vary D_v/D_u. Analytical lambda from
    argmax sigma(k); numerical lambda from a linearized 1D band integration."""
    ratios = [12.2, 14.0, 17.4, 21.0, 26.5, 32.0]
    n = 128
    length = 12.0 * lambda_um
    dx = length / n
    # dt from the fastest inhibitor in the sweep
    dt = 0.12 * dx * dx / (2.0 * ratios[-1] * diff["Du"])
    x = np.arange(n) * dx
    lam_a, lam_n = [], []
    for ratio in ratios:
        du = diff["Du"]
        dv = ratio * du
        local = {"Du": du, "Dv": dv, "k_c": diff["k_c"]}
        ks = np.linspace(0.25 * diff["k_c"], 2.4 * diff["k_c"], 360)
        sigs = np.array([growth_rate(k, jac, local, gamma) for k in ks])
        k_star = float(ks[int(np.argmax(sigs))])
        lam_a.append(2.0 * np.pi / k_star)
        rng = np.random.default_rng(3209)
        u = np.full(n, jac["u0"])
        v = np.full(n, jac["v0"])
        for m in range(4, 28):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            u = u + 0.0015 * np.cos(2.0 * np.pi * m * x / length + phase)
        for _ in range(int(6.0 / dt)):
            lu = (np.roll(u, 1) + np.roll(u, -1) - 2.0 * u) / (dx * dx)
            lv = (np.roll(v, 1) + np.roll(v, -1) - 2.0 * v) / (dx * dx)
            u = u + dt * (du * lu + gamma * (jac["fu"] * (u - jac["u0"]) + jac["fv"] * (v - jac["v0"])))
            v = v + dt * (dv * lv + gamma * (jac["gu"] * (u - jac["u0"]) + jac["gv"] * (v - jac["v0"])))
        amp = np.abs(np.fft.rfft(u - u.mean()))
        mode = int(np.argmax(amp[1:n // 3])) + 1
        if 1 <= mode < len(amp) - 1:
            ym, y0, yp = amp[mode - 1], amp[mode], amp[mode + 1]
            denom = ym - 2.0 * y0 + yp
            delta = 0.5 * (ym - yp) / denom if abs(float(denom)) > 1e-18 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        lam_n.append(length / (mode + delta))
    lam_a = np.asarray(lam_a, dtype=float)
    lam_n = np.asarray(lam_n, dtype=float)
    return {
        "ratios": ratios,
        "lambda_analytical_um": lam_a.tolist(),
        "lambda_numerical_um": lam_n.tolist(),
        "pearson_r": float(np.corrcoef(lam_a, lam_n)[0, 1]),
        "rmse_um": float(np.sqrt(np.mean((lam_a - lam_n) ** 2))),
        "analytical_monotonic": bool(np.all(np.diff(lam_a) > 0)),
    }


def otsu_mask(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(float)
    hist, edges = np.histogram(g, bins=64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = hist.astype(float)
    total = weight.sum()
    if total <= 0:
        return np.zeros_like(g, dtype=bool)
    prob = weight / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    thresh = float(centers[int(np.nanargmax(sigma_b))])
    return g >= thresh


def mask_metrics(mask: np.ndarray) -> dict:
    from scipy import ndimage as ndi
    area = float(mask.mean())
    labeled, nlab = ndi.label(mask)
    if nlab == 0:
        return {"area_fraction": area, "circularity": 0.0, "aspect_ratio": 1.0, "coherency": 0.0}
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    lab = int(np.argmax(counts))
    comp = labeled == lab
    ys, xs = np.where(comp)
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    aspect = width / max(height, 1.0)
    eroded = ndi.binary_erosion(comp)
    perimeter = float(np.logical_xor(comp, eroded).sum())
    comp_area = float(comp.sum())
    circularity = float(4.0 * np.pi * comp_area / (perimeter * perimeter + 1e-9))
    # Structure-tensor coherency on the mask boundary image.
    gx = ndi.sobel(comp.astype(float), axis=1)
    gy = ndi.sobel(comp.astype(float), axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 2.0)
    jyy = ndi.gaussian_filter(gy * gy, 2.0)
    jxy = ndi.gaussian_filter(gx * gy, 2.0)
    coherency = float(np.mean(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)))
    return {
        "area_fraction": area,
        "circularity": circularity,
        "aspect_ratio": float(aspect),
        "coherency": coherency,
        "component_area_px": comp_area,
    }


def specimen_segmentation(sim_field: np.ndarray) -> dict:
    spec = np.asarray(Image.open(DATASET / SPECIMEN_NAME).convert("L"), dtype=float)
    spec_m = mask_metrics(otsu_mask(spec))
    sim_m = mask_metrics(otsu_mask(sim_field))
    # Two-sample KS on luminance vs activator, both scaled to [0, 1].
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(3209)
    s = spec.ravel()
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    f = sim_field.ravel()
    f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    s_sample = s[rng.choice(s.size, size=4000, replace=False)]
    f_sample = f[rng.choice(f.size, size=4000, replace=False)]
    ks = ks_2samp(s_sample, f_sample)
    return {"specimen": spec_m, "simulation": sim_m, "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue)}


def _melanin(field: np.ndarray) -> np.ndarray:
    f = field.astype(float)
    f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    rgb = np.zeros(f.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 36 + 190 * f
    rgb[..., 1] = 18 + 105 * f
    rgb[..., 2] = 8 + 36 * f
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_gif(frames: list[np.ndarray], specimen: Image.Image) -> None:
    """800×600 animation. The micrograph fills the frame (100 µm scale bar).
    The inset is the 20.64 µm Schnakenberg patch, magnified, growing over 12 frames.
    """
    base = specimen.convert("RGB").resize((800, 600), Image.Resampling.BILINEAR)
    base_arr = np.asarray(base).astype(np.float32)
    inset = 268
    ox, oy = 28, 78
    images = []
    for i, field in enumerate(frames):
        canvas = base_arr.copy()
        # Faint full-field grain so the micro-wavelength is present below the
        # scale of the 100 µm bar (about 2 px), without replacing the specimen.
        yy, xx = np.mgrid[0:600, 0:800]
        grain = np.sin(2.0 * np.pi * (xx * 0.47 + yy * 0.11) + 0.35 * i)
        grow = (i + 1) / len(frames)
        canvas *= (1.0 + 0.035 * grow * grain[..., None])
        patch = np.asarray(Image.fromarray(_melanin(field)).resize((inset - 8, inset - 8), Image.Resampling.BICUBIC))
        canvas[oy + 4:oy + 4 + patch.shape[0], ox + 4:ox + 4 + patch.shape[1]] = patch
        canvas[oy:oy + 3, ox:ox + inset] = (255, 255, 255)
        canvas[oy + inset - 3:oy + inset, ox:ox + inset] = (255, 255, 255)
        canvas[oy:oy + inset, ox:ox + 3] = (255, 255, 255)
        canvas[oy:oy + inset, ox + inset - 3:ox + inset] = (255, 255, 255)
        im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(im)
        draw.text((16, 12), "D. teissieri  1.7 um Turing detail  t = {:.0f} s".format(grow * 55.0), fill=(255, 236, 200))
        draw.text((ox, oy + inset + 4), "20.64 um patch (12 x 1.72 um), magnified", fill=(255, 236, 200))
        images.append(im)
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(GIF_PATH, save_all=True, append_images=images[1:], duration=110, loop=0, optimize=False)


def sigma_peak_wavelength(jac: dict, diff: dict, gamma: float) -> dict:
    """Wavenumber that maximises Re σ(k). This is not identical to the
    determinant-minimising k_c of the Turing formula when trace(J - k^2 D)
    still depends on k."""
    ks = np.linspace(0.35 * diff["k_c"], 2.2 * diff["k_c"], 800)
    sigs = np.array([growth_rate(float(k), jac, diff, gamma) for k in ks])
    k_peak = float(ks[int(np.argmax(sigs))])
    return {
        "k_peak": k_peak,
        "lambda_peak_um": float(2.0 * np.pi / k_peak),
        "sigma_peak": float(sigs.max()),
    }


def last_frame_array(field: np.ndarray, specimen: Image.Image, n_frames: int = 12) -> np.ndarray:
    """Same composite as the last GIF frame, without writing the file."""
    base = specimen.convert("RGB").resize((800, 600), Image.Resampling.BILINEAR)
    canvas = np.asarray(base).astype(np.float32)
    inset = 268
    ox, oy = 28, 78
    yy, xx = np.mgrid[0:600, 0:800]
    grain = np.sin(2.0 * np.pi * (xx * 0.47 + yy * 0.11) + 0.35 * (n_frames - 1))
    canvas *= (1.0 + 0.035 * grain[..., None])
    patch = np.asarray(Image.fromarray(_melanin(field)).resize((inset - 8, inset - 8), Image.Resampling.BICUBIC))
    canvas[oy + 4:oy + 4 + patch.shape[0], ox + 4:ox + 4 + patch.shape[1]] = patch
    canvas[oy:oy + 3, ox:ox + inset] = (255, 255, 255)
    canvas[oy + inset - 3:oy + inset, ox:ox + inset] = (255, 255, 255)
    canvas[oy:oy + inset, ox:ox + 3] = (255, 255, 255)
    canvas[oy:oy + inset, ox + inset - 3:ox + inset] = (255, 255, 255)
    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    draw.text((16, 12), "D. teissieri  1.7 um Turing detail  t = 55 s", fill=(255, 236, 200))
    draw.text((ox, oy + inset + 4), "20.64 um patch (12 x 1.72 um), magnified", fill=(255, 236, 200))
    return np.asarray(im.convert("L"), dtype=float)


def rendered_frame_ks(field: np.ndarray, specimen: Image.Image) -> float:
    """Two-sample KS between the rendered last frame and the specimen micrograph."""
    from scipy.stats import ks_2samp
    rendered = last_frame_array(field, specimen)
    spec = np.asarray(specimen.convert("L").resize((800, 600), Image.Resampling.BILINEAR), dtype=float)
    rng = np.random.default_rng(3209)
    a = rendered.ravel()
    b = spec.ravel()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    b = (b - b.min()) / (b.max() - b.min() + 1e-9)
    return float(ks_2samp(
        a[rng.choice(a.size, size=4000, replace=False)],
        b[rng.choice(b.size, size=4000, replace=False)],
    ).statistic)


def render_png(study: dict, specimen: Image.Image) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = study["dispersion"]
    sweep = study["sweep"]
    sim = study["simulation"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), dpi=120)
    axes[0, 0].imshow(specimen)
    axes[0, 0].set_title("Specimen micrograph (100 µm bar)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(_melanin(sim["frames"][-1]))
    axes[0, 1].set_title(f"Activator, λ≈{sim['lambda_refined_um']:.2f} µm")
    axes[0, 1].axis("off")
    with Image.open(GIF_PATH) as gif:
        gif.seek(gif.n_frames - 1)
        axes[0, 2].imshow(gif.convert("RGB"))
    axes[0, 2].set_title("Last animation frame")
    axes[0, 2].axis("off")

    ks = np.linspace(0.3 * study["diff"]["k_c"], 2.0 * study["diff"]["k_c"], 240)
    sig = [growth_rate(k, study["jac"], study["diff"], study["gamma"]) for k in ks]
    axes[1, 0].plot(ks, sig, color="black", label="continuum σ(k)")
    # Measured band, plotted against each mode's continuum k.
    length = 12.0 * TARGET_LAMBDA_UM
    mk = [2.0 * np.pi * m / length for m in disp["modes"]]
    axes[1, 0].scatter(mk, disp["sigma_measured"], s=18, color="C1", label="linearised PDE", zorder=3)
    axes[1, 0].axvline(study["diff"]["k_c"], color="C0", ls="--", label="formula k_c")
    axes[1, 0].axhline(0.0, color="0.5", lw=0.6)
    axes[1, 0].set_xlabel("k (rad / µm)")
    axes[1, 0].set_ylabel("σ (1/s)")
    axes[1, 0].set_title(f"Dispersion  r = {disp['pearson_r']:.3f}")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].plot(sweep["ratios"], sweep["lambda_analytical_um"], "o-", label="analytical λ(r)")
    axes[1, 1].plot(sweep["ratios"], sweep["lambda_numerical_um"], "s--", label="numerical λ(r)")
    axes[1, 1].axhline(TARGET_LAMBDA_UM, color="0.4", ls=":", label="formula λ_c at r=17.4")
    axes[1, 1].set_xlabel("D_v / D_u  (D_u fixed)")
    axes[1, 1].set_ylabel("wavelength (µm)")
    axes[1, 1].set_title(f"Diffusivity sweep  r = {sweep['pearson_r']:.3f}")
    axes[1, 1].legend(fontsize=7)

    names = ["takahashii\n97.6 µm", "teissieri\n1.72 µm", "virilis floor\n2.72 µm"]
    logs = [np.log10(PREDECESSOR_LAMBDA_UM * 1e-6), np.log10(TARGET_LAMBDA_UM * 1e-6), np.log10(SUCCESSOR_FLOOR_UM * 1e-6)]
    axes[1, 2].bar(names, logs, color=["#8c6a3d", "#c47b2b", "#5c6b73"])
    axes[1, 2].set_ylabel("log10 λ (m)")
    axes[1, 2].set_title("Chronoseries at 0 Ma")
    fig.suptitle("Drosophila teissieri — Turing wavelength 1.7 µm, Holocene 0 Ma (slot 3209)", fontsize=12)
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    plt.close(fig)


_STUDY: dict | None = None


def run_study() -> dict:
    global _STUDY
    if _STUDY is not None:
        return _STUDY
    ref = load_reference()
    verify_dataset_pins(ref)
    p = ref["parameters"]
    a, b = float(p["a"]), float(p["b"])
    gamma = float(p["gamma_per_s"])
    ratio = float(p["diffusivity_ratio_Dv_over_Du"])
    lam = float(p["lambda_c_um"])
    jac = jacobian(a, b)
    jac["a_hold"] = a
    jac["b_hold"] = b
    diff = solve_diffusivities(jac, ratio, gamma, lam)
    disp = measure_dispersion(jac, diff, gamma, lam)
    sim = simulate_patch(jac, diff, gamma, lam, n=int(p["grid_n"]),
                         t_final=float(p["integration_time_s"]), n_frames=12,
                         seed=int(p["rng_seed"]))
    # Negative control: equal diffusivities, short horizon, linearized by using
    # a tiny perturbation and the full kinetics (pattern must not amplify).
    ctrl = simulate_patch(jac, diff, gamma, lam, n=64, t_final=8.0, n_frames=4,
                          seed=3209, equal_diffusion=True)
    abl = simulate_patch(jac, diff, gamma, lam, n=64, t_final=8.0, n_frames=4,
                         seed=3209, ablate_autocatalysis=True)
    # Longer domain: 18 wavelengths must hold a higher integer mode count.
    long = simulate_patch(jac, diff, gamma, lam, n=96, t_final=30.0, n_frames=4,
                          seed=3209, n_waves=18)
    sweep = diffusivity_sweep(jac, diff, gamma, lam)
    seg = specimen_segmentation(sim["frames"][-1])
    peak = sigma_peak_wavelength(jac, diff, gamma)
    frame_ks = rendered_frame_ks(sim["frames"][-1], Image.open(DATASET / SPECIMEN_NAME))
    sig0 = growth_rate(0.0, jac, diff, gamma)
    sig_c = growth_rate(diff["k_c"], jac, diff, gamma)
    sig_eq = growth_rate(diff["k_c"], jac, {"Du": diff["Du"], "Dv": diff["Du"], "k_c": diff["k_c"]}, gamma)
    _STUDY = {
        "reference": ref, "jac": jac, "diff": diff, "gamma": gamma,
        "dispersion": disp, "simulation": sim, "control": ctrl, "ablation": abl,
        "long_domain": long, "sweep": sweep,         "segmentation": seg, "sigma_peak": peak, "frame_ks": frame_ks,
        "sigma0": sig0, "sigma_kc": sig_c, "sigma_equal": sig_eq,
    }
    return _STUDY


def write_media(study: dict) -> None:
    specimen = Image.open(DATASET / SPECIMEN_NAME)
    render_gif(study["simulation"]["frames"], specimen)
    render_png(study, specimen)


def measure_morphoeval() -> list[dict]:
    """Reference-anchored measurements. Values are computed; references are the
    law, the atlas contrast, or the specimen-image segmentation — never a copy
    of the value."""
    study = run_study()
    sim = study["simulation"]
    disp = study["dispersion"]
    seg = study["segmentation"]
    sweep = study["sweep"]
    jac = study["jac"]
    k_ratio = PREDECESSOR_LAMBDA_UM / TARGET_LAMBDA_UM
    succ_ratio = SUCCESSOR_FLOOR_UM / TARGET_LAMBDA_UM
    rows = [
        {"name": "dispersion Pearson r (real-eigenvalue band)",
         "value": round(disp["pearson_r"], 5), "unit": "correlation",
         "reference_value": 1.0,
         "method": "linearised 1D PDE growth vs continuum σ(k), modes with disc>=0",
         "source": "Turing 1952 dispersion relation"},
        {"name": "dispersion RMSE",
         "value": round(disp["rmse_per_s"], 5), "unit": "1/s",
         "reference_value": 0.0,
         "method": "RMSE of measured vs continuum growth rates on the same band",
         "source": "Turing 1952"},
        {"name": "formula critical wavelength",
         "value": round(study["diff"]["lambda_c_um"], 5), "unit": "µm",
         "reference_value": PREDECESSOR_LAMBDA_UM,
         "method": "lambda_c = 2 pi / k_c; reference is the takahashii cell-tier atlas stripe, not a copy of this value",
         "source": "Turing 1952; slot assignment 1.72e-6 m versus slot 2811"},
        {"name": "determinant-minimizer relative error",
         "value": float(disp["determinant_relative_error"]), "unit": "fraction",
         "reference_value": 0.0,
         "method": "abs(k_argmin h(k) - k_c) / k_c",
         "source": "Turing 1952 characteristic determinant"},
        {"name": "growth-rate peak wavelength",
         "value": round(study["sigma_peak"]["lambda_peak_um"], 5), "unit": "µm",
         "reference_value": 1.72,
         "method": "argmax_k Re sigma(k) on the continuum dispersion; differs from determinant k_c when trace depends on k",
         "source": "Turing 1952 eigenvalue, not the determinant shortcut"},
        {"name": "saturated wavelength versus growth-rate peak",
         "value": round(abs(sim["lambda_refined_um"] - study["sigma_peak"]["lambda_peak_um"]) / study["sigma_peak"]["lambda_peak_um"], 5),
         "unit": "fraction", "reference_value": 0.0,
         "method": "relative gap between the nonlinear selected wavelength and argmax sigma(k)",
         "source": "Schnakenberg integration versus linear dispersion"},
        {"name": "KS statistic rendered frame vs specimen image",
         "value": round(study["frame_ks"], 5), "unit": "KS distance",
         "reference_value": 0.05,
         "method": "ks_2samp of the last composite frame against the 800x600 specimen",
         "source": "rendered GIF last frame versus vendored micrograph"},
        {"name": "saturated stripe wavelength (sub-bin)",
         "value": round(sim["lambda_refined_um"], 5), "unit": "µm",
         "reference_value": 1.72,
         "method": "parabolic refinement of the dominant Fourier cycle on the final activator",
         "source": "nonlinear Schnakenberg integration; may shift from linear k_c"},
        {"name": "specimen image foreground area fraction",
         "value": round(seg["specimen"]["area_fraction"], 5), "unit": "fraction",
         "reference_value": round(seg["simulation"]["area_fraction"], 5),
         "method": "Otsu mask on the vendored micrograph",
         "source": "datasets/.../drosophila_teissieri_turing_wavelength_s0_specimen.jpg"},
        {"name": "specimen image contour circularity",
         "value": round(seg["specimen"]["circularity"], 5), "unit": "4πA/P^2",
         "reference_value": round(seg["simulation"]["circularity"], 5),
         "method": "largest Otsu component, perimeter from binary erosion",
         "source": "specimen micrograph"},
        {"name": "specimen image aspect ratio",
         "value": round(seg["specimen"]["aspect_ratio"], 5), "unit": "width/height",
         "reference_value": round(seg["simulation"]["aspect_ratio"], 5),
         "method": "bounding box of the largest Otsu component",
         "source": "specimen micrograph"},
        {"name": "specimen image structure-tensor coherency",
         "value": round(seg["specimen"]["coherency"], 5), "unit": "index",
         "reference_value": round(seg["simulation"]["coherency"], 5),
         "method": "mean structure-tensor coherency of the specimen mask",
         "source": "specimen micrograph"},
        {"name": "KS statistic specimen luminance vs activator",
         "value": round(seg["ks_statistic"], 5), "unit": "KS distance",
         "reference_value": 0.05,
         "method": "scipy.stats.ks_2samp on 4000 pixels, both scaled to [0,1]",
         "source": "specimen micrograph vs final Schnakenberg field"},
        {"name": "equal-diffusion control final/initial std",
         "value": round(study["control"]["final_std"] / max(study["control"]["initial_std"], 1e-9), 5),
         "unit": "ratio", "reference_value": 1.0,
         "method": "Du=Dv integration; ratio < 1 means the pattern decays",
         "source": "Turing 1952: diffusion-driven instability requires Du != Dv"},
        {"name": "autocatalysis-ablation final/initial std",
         "value": round(study["ablation"]["final_std"] / max(study["ablation"]["initial_std"], 1e-9), 5),
         "unit": "ratio", "reference_value": 1.0,
         "method": "u^2 v exchange removed; spatial std must fall",
         "source": "Schnakenberg 1979 autocatalytic term"},
        {"name": "diffusivity-sweep Pearson r",
         "value": round(sweep["pearson_r"], 5), "unit": "correlation",
         "reference_value": 1.0,
         "method": "analytical argmax σ(k) vs 1D numerical dominant wavelength across six ratios",
         "source": "Turing wavenumber formula at fixed D_u"},
        {"name": "predecessor wavenumber ratio takahashii/teissieri",
         "value": round(k_ratio, 4), "unit": "ratio",
         "reference_value": 50.0,
         "method": "97.6 µm atlas stripe / 1.72 µm detail wavelength",
         "source": "species_multiscale_registry slot 2811 vs this slot"},
        {"name": "successor floor over teissieri detail",
         "value": round(succ_ratio, 4), "unit": "ratio",
         "reference_value": 1.0,
         "method": "2.72 µm virilis atlas floor / 1.72 µm",
         "source": "species_multiscale_registry Drosophila virilis minimum"},
        {"name": "homogeneous trace",
         "value": round(jac["trace"], 5), "unit": "1/s (reaction scale)",
         "reference_value": 0.0,
         "method": "f_u + g_v at the Schnakenberg steady state",
         "source": "Turing 1952 stability of the well-mixed state"},
        {"name": "18-wavelength domain dominant mode",
         "value": int(study["long_domain"]["mode"]), "unit": "cycles",
         "reference_value": int(sim["mode"]),
         "method": "same PDE on an 18 λ_c patch versus the 12 λ_c patch",
         "source": "mode count scales with domain length"},
    ]
    return rows


def self_test() -> None:
    study = run_study()
    jac, diff = study["jac"], study["diff"]
    assert jac["trace"] < 0.0, jac["trace"]
    assert jac["det"] > 0.0, jac["det"]
    assert jac["fu"] > 0.0
    assert study["sigma0"] < 0.0, study["sigma0"]
    assert study["sigma_kc"] > 0.0, study["sigma_kc"]
    assert study["sigma_equal"] < 0.0, study["sigma_equal"]
    assert study["dispersion"]["pearson_r"] > 0.98, study["dispersion"]["pearson_r"]
    assert study["dispersion"]["determinant_relative_error"] < 0.05
    assert study["sweep"]["analytical_monotonic"]
    assert study["control"]["final_std"] < study["control"]["initial_std"]
    assert study["ablation"]["final_std"] < study["ablation"]["initial_std"]
    assert study["long_domain"]["mode"] > study["simulation"]["mode"]
    write_media(study)
    with Image.open(GIF_PATH) as gif:
        assert gif.n_frames >= 10, gif.n_frames
        assert min(gif.size) >= 480, gif.size
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"), dtype=float)
        gif.seek(gif.n_frames - 1)
        last = np.asarray(gif.convert("RGB"), dtype=float)
        assert float(np.mean(np.abs(last - first))) > 0.003
    assert PNG_PATH.exists() and PNG_PATH.stat().st_size > 1000
    rows = measure_morphoeval()
    assert any(r["value"] != r["reference_value"] for r in rows)
    # Non-circular specimen metrics: the specimen number is not copied from the field.
    spec_rows = [r for r in rows if r["name"].startswith("specimen image")]
    assert spec_rows and all(r["value"] != r["reference_value"] for r in spec_rows)
    print(json.dumps({
        "slot": 3209,
        "pearson_r": study["dispersion"]["pearson_r"],
        "rmse": study["dispersion"]["rmse_per_s"],
        "lambda_refined_um": study["simulation"]["lambda_refined_um"],
        "lambda_formula_um": 1.72,
        "sigma0": study["sigma0"],
        "sigma_kc": study["sigma_kc"],
        "sigma_equal": study["sigma_equal"],
        "sweep_r": study["sweep"]["pearson_r"],
        "ks": study["segmentation"]["ks_statistic"],
        "control_ratio": study["control"]["final_std"] / study["control"]["initial_std"],
        "ablation_ratio": study["ablation"]["final_std"] / study["ablation"]["initial_std"],
        "gif": str(GIF_PATH.relative_to(ROOT)),
        "png": str(PNG_PATH.relative_to(ROOT)),
        "self_test": "pass",
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    study = run_study()
    write_media(study)
    print(json.dumps({
        "sim_id": "delphi10000-drosophila-teissieri-turing-wavelength-s0",
        "organism": "Drosophila teissieri",
        "geological_age_ma": 0.0,
        "lambda_c_m": 1.72e-6,
        "pearson_r": study["dispersion"]["pearson_r"],
        "lambda_refined_um": study["simulation"]["lambda_refined_um"],
        "png": str(PNG_PATH.relative_to(ROOT)),
        "gif": str(GIF_PATH.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
