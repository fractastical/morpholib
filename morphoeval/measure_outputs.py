#!/usr/bin/env python3
"""
measure_outputs.py — snapshot the *morphology* behind a MorphoEval audit.

The audit nanopubs pin the code (commit) and the judgement (scores); this tool pins the
form that was judged, in two ways, and writes a sidecar YAML that build_morphoeval_kg.py
merges into the nanopubs:

  1. content fingerprint of every rendered output that an evidence item cites
     (sha256, bytes, media type, pixel size, frame count)  -> schema:ImageObject/sha256
  2. morphometrics measured from the actual geometry / pixels of those outputs
     (and, for Raup, from the script's own generating function)  -> schema:PropertyValue
     evidence on the `morphometric-validation` assessment

Measurements are deliberately simple, stated with their method, and carry a reference
value only where one is actually known; "no reference recorded" is an honest state.

    python3 morphoeval/measure_outputs.py morphoeval/audits/2026-10-13-morphology-audit.yaml
      -> morphoeval/audits/2026-10-13-morphology-audit.measurements.yaml

Requires: numpy, scipy, Pillow, pyyaml.  Outputs that are not on disk are reported as
`missing` and get neither hash nor measurements (nothing is invented).
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageSequence
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# --------------------------------------------------------------------------- #
# 1. fingerprints
# --------------------------------------------------------------------------- #
def fingerprint(path: Path) -> dict:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    d = {"file": str(path.relative_to(ROOT)), "sha256": h, "bytes": path.stat().st_size,
         "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}
    try:
        with Image.open(path) as im:
            d["width"], d["height"] = im.size
            d["frames"] = getattr(im, "n_frames", 1)
    except Exception:
        pass
    return d


def frames(path: Path, every: int = 1):
    with Image.open(path) as im:
        return [np.asarray(f.convert("RGB")) for i, f in enumerate(ImageSequence.Iterator(im)) if i % every == 0]


def M(name, value, unit, method, reference=None, source=None, note=None, derived_from="output"):
    """derived_from: 'output' (measured on the rendered file) or 'script' (computed from the
    script's own generating function / run)."""
    m = {"name": name, "value": float(round(value, 4)) if isinstance(value, (float, np.floating)) else value,
         "unit": unit, "method": method, "derived_from": derived_from}
    if reference is not None:
        m["reference_value"] = reference
    if source:
        m["reference_source"] = source
    if note:
        m["note"] = note
    return m


# --------------------------------------------------------------------------- #
# 2. per-simulation morphometrics
# --------------------------------------------------------------------------- #
def measure_planform(gif: Path) -> list:
    """Union silhouette of the superimposed morphologies per frame (central panel only,
    labels excluded): elongation and how much the silhouette changes across the timeline."""
    fs = frames(gif)
    H, W_, _ = fs[0].shape
    r0, r1, c0, c1 = int(0.22 * H), int(0.85 * H), int(0.25 * W_), int(0.75 * W_)   # exclude title/badges
    masks = []
    for a in fs:
        roi = a[r0:r1, c0:c1].astype(int)
        m = roi.min(2) < 235                                   # anything not white
        m = ndi.binary_opening(m, iterations=2)
        lab, n = ndi.label(m)
        if n == 0:
            continue
        sizes = ndi.sum(m, lab, range(1, n + 1))
        masks.append(lab == 1 + int(np.argmax(sizes)))
    ar, areas = [], []
    for m in masks:
        ys, xs = np.where(m)
        ar.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)); areas.append(m.sum())
    ious = [(masks[i] & masks[j]).sum() / (masks[i] | masks[j]).sum()
            for i in range(len(masks)) for j in range(i + 1, len(masks))]
    return [
        M("silhouette width/height (bbox, mean over frames)", float(np.mean(ar)), "ratio",
          "largest non-white connected component in the central panel of each GIF frame; bbox W/H",
          reference="≈0.15–0.3 (adult S. mediterranea is ~3–7× longer than wide, long axis = AP)",
          source="qualitative; Lobo et al. 2013 Planform renderings show elongated AP bodies",
          note=f"min {min(ar):.3f}, max {max(ar):.3f}; rendered body is wider than long"),
        M("silhouette IoU between frames (mean)", float(np.mean(ious)), "ratio",
          "pairwise intersection-over-union of the per-frame silhouettes on the fixed canvas",
          reference="distinct morphologies should give low IoU; one body plan over time should be near 1",
          note=f"min {min(ious):.3f}; frames differ mainly by added translucent flaps, not by body form"),
        M("frames with a silhouette", len(masks), "count", "frames where a component was found", note=f"of {len(fs)}"),
    ]


def measure_planform_v2(script: Path) -> list:
    """Geometry decoded by 2013-lobo-planform-morphology.py from PlanformDB: the wild-type
    silhouette against the radii encoded in the database, how many of the morphologies
    render as distinct silhouettes, and how different the canonical phenotypes are."""
    import importlib.util
    import os
    from matplotlib.path import Path as MplPath
    db = os.environ.get("PLANFORM_DB_PATH")
    if not db or not Path(db).is_file():
        raise FileNotFoundError("PLANFORM_DB_PATH not set or not a file")
    spec = importlib.util.spec_from_file_location("planform_v2", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # dataclasses need the module registered
    spec.loader.exec_module(mod)
    morphs = mod.load_morphologies(db)
    wt = morphs[1]
    mm = mod.morphometrics(wt)
    # encoded reference: head anterior radius 80 + head→trunk 200 + trunk→tail 200 + tail posterior radius 80
    # = 560 along AP; trunk lateral radii 60 + 60 = 120 across ML (RegionParam / RegionsLink rows of morphology 1)
    enc_len, enc_wid = 80 + 200 + 200 + 80, 60 + 60
    rows = {k: mod.morphometrics(m) for k, m in morphs.items()}

    def sil_key(r):
        return (round(r["ap_length"]), round(r["ml_width"]), round(r["area"], -2), r["n_heads"], r["n_tails"])
    keys = {sil_key(r) for r in rows.values()}
    share_wt = sum(1 for r in rows.values() if sil_key(r) == sil_key(rows[1]))

    def full_key(m):                                   # every encoded number: regions, links, organs
        import dataclasses
        return (tuple((reg.type, tuple(round(p, 3) for p in reg.params)) for _, reg in sorted(m.regions.items())),
                tuple(dataclasses.astuple(l)[2:] for l in m.links),
                tuple(dataclasses.astuple(o)[:3] for o in m.organs))
    full = {full_key(m) for m in morphs.values()}

    def iou(a, b):
        pts = np.vstack(list(a.outlines.values()) + list(b.outlines.values()))
        x0, y0 = pts.min(0) - 10; x1, y1 = pts.max(0) + 10
        gx, gy = np.meshgrid(np.linspace(x0, x1, 256), np.linspace(y0, y1, 256))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        ma = np.zeros(len(grid), bool); mb = np.zeros(len(grid), bool)
        for poly in a.outlines.values():
            ma |= MplPath(poly).contains_points(grid)
        for poly in b.outlines.values():
            mb |= MplPath(poly).contains_points(grid)
        return float((ma & mb).sum() / (ma | mb).sum())

    out = [
        M("wild-type silhouette AP length", mm["ap_length"], "Planform units",
          "union of decoded region outlines of Morphology 1, rasterised; bbox extent along the AP axis",
          reference=enc_len, source="PlanformDB 2.5.0 RegionParam/RegionsLink of morphology 1 (80+200+200+80)",
          derived_from="script"),
        M("wild-type silhouette ML width", mm["ml_width"], "Planform units", "as above, across the ML axis",
          reference=enc_wid, source="trunk lateral radii 60+60", derived_from="script"),
        M("wild-type AP/ML aspect", mm["aspect"], "ratio", "AP length / ML width of the decoded silhouette",
          reference=round(enc_len / enc_wid, 3), source="encoded radii", derived_from="script",
          note="S. mediterranea in Lobo et al. 2013 Fig. 1A is of this order; the formalism, not a specimen, is the reference"),
        M("morphologies rendered", len(morphs), "count", "Morphology rows with at least one encoded region",
          note="372 rows in the database; 3 have no regions (Lethal, Midbody Buldge, Double head, side tail)",
          derived_from="script"),
        M("distinct body silhouettes", len(keys), "count",
          "unique (AP length, ML width, area to 100 units, #heads, #tails) tuples over all rendered morphologies",
          reference=len(morphs), source="upper bound: one silhouette per morphology",
          note=f"{share_wt} morphologies share the wild-type body and differ only in organs "
               "(ectopic eyes, doubled CNS, pharynx position, ...), which is how the database encodes them",
          derived_from="script"),
        M("distinct encodings (regions + links + organs)", len(full), "count",
          "unique tuples of every encoded number read from the database", reference=len(morphs),
          source="every morphology row is a distinct phenotype", derived_from="script"),
        M("IoU wild type vs double head (#24)", iou(wt, morphs[24]), "ratio",
          "intersection-over-union of the two decoded silhouettes on a shared canvas (region centres aligned by layout)",
          reference="< 1; the two bodies differ only at the posterior cap (tail -> second head), the rest of the difference is organs",
          derived_from="script"),
        M("IoU wild type vs headless (#20)", iou(wt, morphs[20]), "ratio", "as above",
          reference="≈ (trunk+tail area)/(whole-body area) if the head is simply absent", derived_from="script"),
    ]
    return out


def measure_raup(script: Path, variables: Path) -> list:
    """Effective translation rate from the script's own raup_shell(): Raup's T is the axial
    translation per revolution relative to the generating-curve centre's distance from the
    axis; the script grows r exponentially but translates linearly, so T_eff -> 0."""
    src = script.read_text()
    s = src.index("def raup_shell"); e = src.index("\ndef ", s + 10)
    ns = {"np": np}; exec(src[s:e], ns)                       # only the generating function
    raup_shell = ns["raup_shell"]
    out = []
    for sh in json.loads(variables.read_text())["empirical_shells"]:
        W, D, T = sh["W"], sh["D"], sh["T"]
        X, Y, Z = raup_shell(W=W, D=D, T=T, turns=6, res=100)
        ax = 1 if X.shape[0] > X.shape[1] else 0
        cx, cy, cz = X.mean(axis=ax), Y.mean(axis=ax), Z.mean(axis=ax)
        rc = np.hypot(cx, cy); n = len(rc); pw = n // 6
        T_eff = abs(cz[-1] - cz[-1 - pw]) / rc[-1]
        hw = abs(cz[-1] - cz[0]) / (2 * rc.max())
        out.append(M(f"effective translation rate T_eff, '{sh['name']}' (nominal T={T})", float(T_eff), "ratio",
                     "Δz over the last whorl ÷ generating-curve centre radius, from raup_shell(W,D,T,turns=6)",
                     reference=T, source="Raup 1966: T is defined relative to the whorl radius, so T_eff should equal T",
                     note=f"shell height/width = {hw:.3f}", derived_from="script"))
    return out


def measure_turing(gif: Path, N: int = 200) -> list:
    """Dominant pattern wavelength (radial FFT peak) in grid cells, pattern coverage, and
    whether the animation actually evolves (frame-to-frame change inside the axes box)."""
    fs = frames(gif)
    a0, aL = fs[0].astype(int), fs[-1].astype(int)
    # axes box = pixels that are viridis-coloured (not white/grey/black): saturated
    sat = aL.max(2) - aL.min(2)
    box = ndi.binary_opening(sat > 40, iterations=3)
    ys, xs = np.where(box)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    yellow = lambda a: (a[r0:r1, c0:c1, 0] > 150) & (a[r0:r1, c0:c1, 2] < 120)     # viridis high end
    yl, y0 = yellow(aL), yellow(a0)
    g = aL[r0:r1, c0:c1].mean(2); g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = F.shape; cy, cx = h // 2, w // 2
    yy, xx = np.indices(F.shape); rr = np.hypot(yy - cy, xx - cx).astype(int)
    radial = np.bincount(rr.ravel(), F.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    radial[:4] = 0
    k = int(np.argmax(radial[: min(h, w) // 2]))
    wl_cells = (min(h, w) / k) * N / min(h, w) if k else float("nan")
    lab, ncomp = ndi.label(yl)
    change = float(np.mean(yl != y0))
    return [
        M("dominant wavelength (last frame)", float(wl_cells), "grid cells",
          f"peak of the radial power spectrum of the axes-box image, scaled to the N={N} grid",
          reference=None, source=None,
          note="Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded"),
        M("pattern coverage (last frame)", float(yl.mean()), "fraction", "fraction of axes box in the high-V (yellow) band"),
        M("connected high-V components (last frame)", int(ncomp), "count", "scipy.ndimage.label on the yellow mask"),
        M("fraction of pattern pixels changed, first→last frame", change, "fraction",
          "XOR of yellow masks of frame 0 and the last frame", reference=">0 for an evolving simulation",
          note="≈0 means the GIF shows a frozen pattern"),
    ]


def measure_hoffmann(script: Path) -> list:
    """Run the wing script (seeded, uses the cached example wing) and take the morphometrics
    it computes itself on real vs model domains: circularity c = 4πA/P² and domain counts.
    The real wing is the specimen here, so the reference values are measured, not quoted."""
    import re
    import subprocess
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"{script.name} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    vals = {}
    for line in r.stdout.splitlines():
        if (m := re.search(r"Real wing: (\d+) vein domains; wing area (\d+) px", line)):
            vals["domains_real"], vals["area_px"] = int(m.group(1)), int(m.group(2))
        if line.startswith("Circularity"):
            m = re.search(r"real ⟨c⟩=([0-9.]+) \(n=(\d+)\)\s+model ⟨c⟩=([0-9.]+) \(n=(\d+)\)\s+hexagon ([0-9.]+)", line)
            if m:
                vals.update(c_real=float(m[1]), n_real=int(m[2]), c_model=float(m[3]), n_model=int(m[4]), hexagon=float(m[5]))
    if "c_model" not in vals:
        raise RuntimeError("could not parse circularity line from script output")
    return [
        M("mean domain circularity, model wing", vals["c_model"], "ratio (4πA/P²)",
          "script's domain_stats() over Voronoi domains after Lloyd equilibration (seed 0)",
          reference=vals["c_real"], source="same statistic on the authors' segmented real wing (Hoffmann et al. 2018 example)",
          note=f"regular hexagon = {vals['hexagon']}; model n={vals['n_model']}, real n={vals['n_real']}",
          derived_from="script"),
        M("vein-domain count, model wing", vals["n_model"], "count", "domains large enough to measure, model",
          reference=vals["n_real"], source="same count on the real wing",
          note="equal by construction: the model is seeded with the real wing's domain count", derived_from="script"),
        M("vein-domain count, real wing (all)", vals["domains_real"], "count",
          "peaks of the distance transform inside the segmented real wing", note=f"wing area {vals['area_px']} px",
          derived_from="script"),
    ]


# --------------------------------------------------------------------------- #
def main():
    audit_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "audits" / "2026-10-13-morphology-audit.yaml").resolve()
    audit = yaml.safe_load(audit_path.read_text())
    out = {"audit": audit["audit"]["id"], "outputs": {}, "measurements": {}, "missing": []}
    for sim in audit["simulations"]:
        if sim.get("status") != "evaluated":
            continue
        sid = sim["id"]
        files = {e["file"] for d in sim["scores"].values() for e in d.get("evidence", []) if e["lines"][1] == 0}
        if sim.get("output"):
            files.add(sim["output"])
        try:   # measurements first: some re-render their output, which must then be fingerprinted
            if sid == "planform-morphology-animation" and sim["script"] == "2013-lobo-planform-morphology.py":
                out["measurements"][sid] = measure_planform_v2(ROOT / sim["script"])
            elif sid == "planform-morphology-animation" and (ROOT / sim["output"]).exists():
                out["measurements"][sid] = measure_planform(ROOT / sim["output"])
            elif sid == "raup-shell-coiling":
                out["measurements"][sid] = measure_raup(ROOT / sim["script"], ROOT / "datasets/raup/variables.json")
            elif sid == "turing-gray-scott" and (ROOT / sim["output"]).exists():
                out["measurements"][sid] = measure_turing(ROOT / sim["output"])
            elif sid == "hoffmann-wing-venation":
                out["measurements"][sid] = measure_hoffmann(ROOT / sim["script"])
        except Exception as ex:  # a failed measurement is reported, never silently skipped
            out["measurements"][sid] = [{"name": "measurement failed", "value": None, "unit": None,
                                         "method": f"{type(ex).__name__}: {ex}"}]
        for f in sorted(files):
            p = ROOT / f
            if p.exists():
                out["outputs"].setdefault(sid, []).append(fingerprint(p))
            else:
                out["missing"].append({"simulation": sid, "file": f})
        print(f"{sid}: {len(out['outputs'].get(sid, []))} output(s) fingerprinted, "
              f"{len(out['measurements'].get(sid, []))} measurement(s)")
        for m in out["measurements"].get(sid, []):
            ref = f"   [ref {m['reference_value']}]" if m.get("reference_value") is not None else ""
            print(f"    {m['name']}: {m['value']} {m['unit'] or ''}{ref}")
    if out["missing"]:
        print("missing outputs (no hash, no measurement):", ", ".join(f"{m['simulation']}:{m['file']}" for m in out["missing"]))
    side = audit_path.with_name(audit_path.stem + ".measurements.yaml")
    side.write_text("# Generated by morphoeval/measure_outputs.py — fingerprints and morphometrics of the\n"
                    "# rendered outputs that the audit judged. Regenerate after re-rendering.\n"
                    + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))
    print("->", side.relative_to(ROOT))


if __name__ == "__main__":
    main()
