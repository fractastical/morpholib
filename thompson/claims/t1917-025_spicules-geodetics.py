#!/usr/bin/env python3
"""
Thompson (1917) T1917-025 — radiolarian geodesic / minimal spherical frameworks.

Claim (Ch. IX-X, pp. 411-492, Figs. 221-235, 236-259):
    Radiolarian and similar siliceous skeletons approximate geodesic/minimal frameworks on a
    sphere (latticed shells as systems of shortest lines and near-equal partitions).

Pass 1 baseline when PASS2=False; Pass 2 adds full freq sweep, ablations, specimen CV/KS stats.

    python3 thompson/claims/t1917-025_spicules-geodetics.py --self-test
    python3 thompson/claims/t1917-025_spicules-geodetics.py [--pass2] [--no-gif]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "thompson_1917" / "t1917-025"
REF = DATA / "t1917_025_spicules_geodetics_reference.json"
RESULT = DATA / "result.json"
OUT_PNG = ROOT / "docs" / "media" / "t1917_025_spicules_geodetics_comparison.png"
OUT_GIF = ROOT / "docs" / "media" / "t1917_025_spicules_geodetics_growth.gif"

CHALLENGER = DATA / "radiolaria_challenger_plate111.jpg"
HAECKEL = DATA / "haeckel_radiolaria_plate_1888.jpg"
# Crop one central radiolarian shell from Challenger plate 111 (pixel box in full scan).
CHALLENGER_CROP = (420, 180, 920, 680)

PASS2 = False
RNG = np.random.default_rng(1917025)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_datasets() -> dict[str, str]:
    ref = json.loads(REF.read_text())
    out = {}
    for name, meta in ref["vendored_files"].items():
        p = DATA / name
        d = sha256_file(p)
        if d != meta["sha256"]:
            raise SystemExit(f"SHA-256 mismatch {name}: {d} != {meta['sha256']}")
        out[name] = d
    return out


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def icosahedron_vertices() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=float,
    )
    return _normalize_rows(verts)


def icosahedron_faces() -> list[tuple[int, int, int]]:
    return [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]


def fibonacci_sphere(n: int) -> np.ndarray:
    """Nearly uniform points on S² (golden-angle spiral)."""
    i = np.arange(n, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return _normalize_rows(np.stack([x, y, z], axis=1))


def knn_spherical_edges(verts: np.ndarray, k: int = 6) -> list[tuple[int, int]]:
    from scipy.spatial import cKDTree

    tree = cKDTree(verts)
    edges: set[tuple[int, int]] = set()
    for i in range(len(verts)):
        _, idx = tree.query(verts[i], k=min(k + 1, len(verts)))
        for j in idx[1:]:
            edges.add(tuple(sorted((int(i), int(j)))))
    return list(edges)


def geodesic_mesh(freq: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Map refinement `freq` to vertex count; freq=0 is the icosahedron."""
    if freq <= 0:
        v = icosahedron_vertices()
        edges = set()
        for a, b, c in icosahedron_faces():
            for u, w in ((a, b), (b, c), (c, a)):
                edges.add(tuple(sorted((u, w))))
        return v, list(edges)
    n = int(12 + 20 * freq * freq)
    v = fibonacci_sphere(n)
    return v, knn_spherical_edges(v, k=6)


def geodesic_edge_lengths(verts: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    lens = []
    for i, j in edges:
        dot = float(np.clip(np.dot(verts[i], verts[j]), -1.0, 1.0))
        lens.append(np.arccos(dot))
    return np.array(lens, dtype=float)


def edge_length_cv(lengths: np.ndarray) -> float:
    if len(lengths) < 2:
        return float("nan")
    return float(np.std(lengths) / (np.mean(lengths) + 1e-12))


def geodesic_reference(freq: int) -> dict[str, Any]:
    verts, edges = geodesic_mesh(freq)
    lens = geodesic_edge_lengths(verts, edges)
    return {
        "freq": freq,
        "n_vertices": len(verts),
        "n_edges": len(edges),
        "edge_lengths": lens,
        "edge_cv": edge_length_cv(lens),
        "mean_valence": float(2 * len(edges) / len(verts)),
    }


def stereographic_project(verts: np.ndarray) -> np.ndarray:
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    denom = 1.0 - z
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return np.stack([x / denom, y / denom], axis=1)


def random_chord_network(n: int, rng: np.random.Generator) -> np.ndarray:
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    lens = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    return lens


def euclidean_grid_edges(grid: int) -> np.ndarray:
    """Square grid on a disc — non-geodesic control."""
    xs = np.linspace(-1, 1, grid)
    ys = np.linspace(-1, 1, grid)
    pts = [(x, y) for x in xs for y in ys if x * x + y * y <= 1.0]
    lens = []
    for i, (x, y) in enumerate(pts):
        if i + 1 < len(pts):
            dx = pts[i + 1][0] - x
            dy = pts[i + 1][1] - y
            if dx * dx + dy * dy < 0.25:
                lens.append(np.hypot(dx, dy))
    return np.array(lens, dtype=float)


def load_challenger_crop() -> np.ndarray:
    with Image.open(CHALLENGER) as im:
        gray = np.asarray(im.convert("L").crop(CHALLENGER_CROP), dtype=float) / 255.0
    return gray


def skeleton_segments(gray: np.ndarray) -> np.ndarray:
    """Extract approximate strut lengths from dark lattice lines."""
    inv = 1.0 - gray
    thresh = np.percentile(inv, 58)
    mask = inv >= thresh
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=3)
    dist = ndi.distance_transform_edt(mask)
    ridge = (dist == ndi.maximum_filter(dist, size=3)) & mask
    skel = ridge
    ys, xs = np.where(skel)
    if len(xs) < 20:
        return np.array([], dtype=float)
    pts = np.stack([xs, ys], axis=1).astype(float)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    lens: list[float] = []
    for i, p in enumerate(pts):
        d, j = tree.query(p, k=5)
        for k in range(1, 5):
            if 4.0 < d[k] < 45.0:
                lens.append(float(d[k]))
    return np.array(lens, dtype=float)


def compare_specimen_to_model(spec_lens: np.ndarray, model_lens: np.ndarray) -> dict[str, float]:
    if len(spec_lens) < 10 or len(model_lens) < 10:
        return {"pearson_r": float("nan"), "rmse": float("nan"), "ks_p": float("nan"), "spec_cv": float("nan")}
    # scale-normalise
    s = spec_lens / (np.mean(spec_lens) + 1e-9)
    m = model_lens / (np.mean(model_lens) + 1e-9)
    n = min(len(s), len(m), 200)
    lo = float(min(s.min(), m.min()))
    hi = float(max(s.max(), m.max()))
    bins = np.linspace(lo, hi, 16)
    hs, _ = np.histogram(s, bins=bins, density=True)
    hm, _ = np.histogram(m, bins=bins, density=True)
    if np.std(hs) < 1e-12 or np.std(hm) < 1e-12:
        r = float("nan")
    else:
        r = float(stats.pearsonr(hs, hm)[0])
    rmse = float(np.sqrt(np.mean((hs - hm) ** 2)))
    ks = stats.kstest(s, m)
    return {
        "pearson_r": r,
        "rmse": rmse,
        "ks_p": float(ks.pvalue),
        "spec_cv": edge_length_cv(spec_lens),
        "model_cv": edge_length_cv(model_lens),
    }


def parameter_sweep(freqs: list[int]) -> list[dict[str, Any]]:
    return [geodesic_reference(f) for f in freqs]


def measure_morphoeval(pass2: bool = False) -> dict[str, Any]:
    freqs = [0, 1, 2, 3] if pass2 else [1]
    refs = parameter_sweep(freqs)
    gray = load_challenger_crop()
    spec = skeleton_segments(gray)
    spec_cv = edge_length_cv(spec) if len(spec) > 5 else float("nan")

    def _score(r: dict[str, Any]) -> float:
        if np.isfinite(spec_cv):
            return abs(r["edge_cv"] - spec_cv)
        return r["edge_cv"]

    best = min(refs, key=_score)
    fit = compare_specimen_to_model(spec, best["edge_lengths"])
    controls = {}
    if pass2:
        controls["random_chords_cv"] = edge_length_cv(random_chord_network(24, RNG))
        controls["euclidean_grid_cv"] = edge_length_cv(euclidean_grid_edges(12))
    return {
        "geodesic_freq_sweep": [{"freq": r["freq"], "edge_cv": r["edge_cv"], "n_edges": r["n_edges"]} for r in refs],
        "best_freq": best["freq"],
        "best_edge_cv": best["edge_cv"],
        "specimen_edge_cv": fit.get("spec_cv"),
        "pearson_r": fit.get("pearson_r"),
        "rmse": fit.get("rmse"),
        "ks_p": fit.get("ks_p"),
        "controls": controls,
        "pass2": pass2,
    }


def render_comparison(metrics: dict[str, Any], pass2: bool) -> None:
    MEDIA = OUT_PNG.parent
    MEDIA.mkdir(parents=True, exist_ok=True)
    gray = load_challenger_crop()
    freq = metrics["best_freq"]
    ref = geodesic_reference(freq)
    verts, _ = geodesic_mesh(freq)
    proj = stereographic_project(verts)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Challenger Plate 111 crop (specimen lattice)")
    axes[0, 0].axis("off")

    axes[0, 1].scatter(proj[:, 0], proj[:, 1], s=8, c="C0")
    axes[0, 1].set_title(f"Geodesic reference freq={freq} (stereographic)")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].axis("off")

    axes[1, 0].hist(ref["edge_lengths"], bins=24, alpha=0.7, label="model", density=True)
    spec = skeleton_segments(gray)
    if len(spec):
        axes[1, 0].hist(spec / np.mean(spec), bins=24, alpha=0.5, label="specimen (norm)", density=True)
    axes[1, 0].set_title("Edge length distributions")
    axes[1, 0].legend()

    sweep = metrics["geodesic_freq_sweep"]
    axes[1, 1].plot([s["freq"] for s in sweep], [s["edge_cv"] for s in sweep], "o-", label="geodesic CV")
    if pass2 and metrics.get("controls"):
        axes[1, 1].axhline(metrics["controls"].get("random_chords_cv", np.nan), ls="--", label="random chords")
        axes[1, 1].axhline(metrics["controls"].get("euclidean_grid_cv", np.nan), ls=":", label="Euclidean grid")
    axes[1, 1].set_xlabel("subdivision freq")
    axes[1, 1].set_ylabel("edge length CV")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(f"Pass {'2' if pass2 else '1'} — r={metrics.get('pearson_r', float('nan')):.3f}")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def render_gif(freq: int = 2) -> None:
    try:
        import imageio.v3 as iio
    except ImportError:
        return
    frames = []
    for f in range(1, max(2, freq) + 1):
        ref = geodesic_reference(f)
        verts, edges = geodesic_mesh(f)
        proj = stereographic_project(verts)
        fig, ax = plt.subplots(figsize=(4, 4))
        for i, j in edges[:: max(1, len(edges) // 80)]:
            ax.plot(proj[[i, j], 0], proj[[i, j], 1], "k-", lw=0.6)
        ax.scatter(proj[:, 0], proj[:, 1], s=6, c="C3")
        ax.set_title(f"Geodesic refinement f={f}, CV={ref['edge_cv']:.3f}")
        ax.set_aspect("equal")
        ax.axis("off")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(buf.copy())
        plt.close(fig)
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT_GIF, frames, duration=0.6, loop=0)


def write_result(metrics: dict[str, Any]) -> None:
    ref = json.loads(REF.read_text())
    spec_cv = metrics.get("specimen_edge_cv")
    model_cv = metrics.get("best_edge_cv")
    r = metrics.get("pearson_r", float("nan"))
    verdict = "mixed"
    if PASS2 and spec_cv is not None and model_cv is not None:
        if spec_cv < 0.35 and (r > 0.5 or model_cv < 0.2):
            verdict = "supported"
        elif spec_cv > 0.55 or (np.isfinite(r) and r < 0.2):
            verdict = "refuted"
    elif not PASS2:
        verdict = "untestable"
    payload = {
        "claim_id": "T1917-025",
        "date_made": "1917",
        "dataset": {
            "name": "radiolaria_challenger_plate111.jpg + haeckel_radiolaria_plate_1888.jpg",
            "doi": None,
            "sha256": ref["vendored_files"]["radiolaria_challenger_plate111.jpg"]["sha256"],
            "audit_named_file_missing": True,
        },
        "verdict": verdict,
        "headline_numbers": {
            "best_geodesic_freq": metrics["best_freq"],
            "model_edge_cv": model_cv,
            "specimen_edge_cv": spec_cv,
            "pearson_r": r,
            "rmse": metrics.get("rmse"),
            "ks_p": metrics.get("ks_p"),
        },
        "script": str(Path(__file__).relative_to(ROOT)),
        "figure": str(OUT_PNG.relative_to(ROOT)),
    }
    def _clean(obj: Any) -> Any:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        return obj

    RESULT.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")


def run_self_test() -> None:
    verify_datasets()
    g = geodesic_reference(0)
    assert g["n_edges"] == 30
    assert g["edge_cv"] < 0.05
    g2 = geodesic_reference(1)
    assert g2["n_edges"] > g["n_edges"]
    lens = random_chord_network(12, RNG)
    assert len(lens) == 12
    gray = load_challenger_crop()
    assert gray.shape[0] > 100
    m = measure_morphoeval(pass2=False)
    assert "best_freq" in m
    print("[self-test] T1917-025 geodesic model OK")


def main() -> None:
    global PASS2
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pass2", action="store_true", help="Pass 2 remediation mode")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    PASS2 = args.pass2
    verify_datasets()
    metrics = measure_morphoeval(pass2=PASS2)
    render_comparison(metrics, PASS2)
    if not args.no_gif:
        render_gif(metrics["best_freq"])
    if PASS2:
        write_result(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
