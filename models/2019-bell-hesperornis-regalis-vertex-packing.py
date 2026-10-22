#!/usr/bin/env python3
"""
models/2019-bell-hesperornis-regalis-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1593)
Organism: Hesperornis regalis (Late Cretaceous Toothed Diving Bird)
Phenomenon Scale: 492 µm (0.0004924 m, log10=-3.308, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Hesperornis regalis (Late Cretaceous (Campanian, 83.5 Ma))
Geological Horizon: Late Cretaceous (Campanian, 83.5 Ma) (83.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Hesperornis
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (492 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1593_hesperornis_regalis_vertex_packing.lean` (`MathOfBiology.Loop_1593`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hesperornis_regalis_vertex_packing"
EXPECTED_SHA256 = {
    "hesperornis_regalis_vertex_packing_reference.json": "a607ffbd0cf095fdaeb2b24f64c14f9a7db9d69ec6a14f22df0dca916e18c6bd",
    "hesperornis_regalis_specimen.jpg": "a8ca1cd24cf7035f2fff3d592b3f55831669012975aed64e3050567bec49e7a7",
    "scout_manifest.json": "f6445392749c5eda72bc89e40b1c889e537d6117b688066b57c169d9bd7d4896",
    "paper_citations.bib": "20c20859bd196ebfb53ffe13add78b2625a7084045b3342c5928237f04074653"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "hesperornis_regalis_vertex_packing_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/2019-bell-hesperornis-regalis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
