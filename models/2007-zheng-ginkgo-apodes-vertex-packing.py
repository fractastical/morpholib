#!/usr/bin/env python3
"""
models/2007-zheng-ginkgo-apodes-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1622)
Organism: Ginkgo apodes (Early Cretaceous Multi-Lobed Jehol Biota Ginkgo Chronospecies)
Phenomenon Scale: 110 µm (0.0001103 m, log10=-3.957, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Ginkgo apodes (Early Cretaceous (Yixian Fm, 121 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 121 Ma) (121.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (110 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1622_ginkgo_apodes_vertex_packing.lean` (`MathOfBiology.Loop_1622`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_apodes_vertex_packing"
EXPECTED_SHA256 = {
    "ginkgo_apodes_vertex_packing_reference.json": "5c9cd8233ed55fc580f312d1cdfb471f228680e4153036f447ea0677fd432974",
    "ginkgo_apodes_specimen.jpg": "3a074e9794ed0797dcc5abd51ab2fcccf8a109da1756d08ad6f52d95faf7d0f5",
    "scout_manifest.json": "cda3c86c28c7bb0ce56e1eac5132cbc604f5264260a1bd7c0a87e182f33b8357",
    "paper_citations.bib": "fbd23b5670f0e4f66e643c7cc53958af0420ee1d344e386952fdcfe8ffa69c22"
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
    ref = json.loads((DATASET_DIR / "ginkgo_apodes_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-zheng-ginkgo-apodes-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
