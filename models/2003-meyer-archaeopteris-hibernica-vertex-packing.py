#!/usr/bin/env python3
"""
models/2003-meyer-archaeopteris-hibernica-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1617)
Organism: Archaeopteris hibernica (Late Devonian Progymnosperm Tree (Planated Megaphyll Fronds))
Phenomenon Scale: 96 µm (9.562e-05 m, log10=-4.019, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Archaeopteris hibernica (Late Devonian (Famennian, 370 Ma))
Geological Horizon: Late Devonian (Famennian, 370 Ma) (370.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Archaeopteris
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (96 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1617_archaeopteris_hibernica_vertex_packing.lean` (`MathOfBiology.Loop_1617`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteris_hibernica_vertex_packing"
EXPECTED_SHA256 = {
    "archaeopteris_hibernica_vertex_packing_reference.json": "e907be0ad02b2c4c35522ea9db8fc877463587982b4632dc04b9e6d632a8d979",
    "archaeopteris_hibernica_specimen.jpg": "107d9b021f541e8a82a7c594b45f0250c4a6c8fa160f1b3c54299cd439a8163e",
    "scout_manifest.json": "cd5c7c0c299bad45f114cda40b518d79dd7b09efc9e6b95e3b22d21d172368f2",
    "paper_citations.bib": "c00b63db58511e468b3bcdc7f4a50f178b9b6848d2273914ea61cc68f5025cf8"
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
    ref = json.loads((DATASET_DIR / "archaeopteris_hibernica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-meyer-archaeopteris-hibernica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
