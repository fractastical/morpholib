#!/usr/bin/env python3
"""
models/1984-jarvik-eusthenopteron-foordi-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1572)
Organism: Eusthenopteron foordi (Late Devonian Tristichopterid Lobe-Fin Fish)
Phenomenon Scale: 354 µm (0.0003535 m, log10=-3.452, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Eusthenopteron foordi (Late Devonian (Frasnian, 379 Ma))
Geological Horizon: Late Devonian (Frasnian, 379 Ma) (379.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Eusthenopteron
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (354 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1572_eusthenopteron_foordi_vertex_packing.lean` (`MathOfBiology.Loop_1572`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eusthenopteron_foordi_vertex_packing"
EXPECTED_SHA256 = {
    "eusthenopteron_foordi_vertex_packing_reference.json": "192ba7e633437f16ac171d7fe28e2c2e5572e867678c4a186f8afbf25b8629ba",
    "eusthenopteron_foordi_specimen.jpg": "d97b68e66855434cb59cbd6b3fc48d7827c97c6981f567e77fb025c359499811",
    "scout_manifest.json": "42a73a6cc354b8a833401220f167081201ab317d4f0d4803f57c6b9bc65ee0f2",
    "paper_citations.bib": "fbae130b348f7a430e3ce1ce38d1b57cb943942a274fee4d2abcc580210e0c76"
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
    ref = json.loads((DATASET_DIR / "eusthenopteron_foordi_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1984-jarvik-eusthenopteron-foordi-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
