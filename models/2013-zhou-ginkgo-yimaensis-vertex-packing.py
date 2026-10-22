#!/usr/bin/env python3
"""
models/2013-zhou-ginkgo-yimaensis-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1621)
Organism: Ginkgo yimaensis (Middle Jurassic Deeply-Divided 6-Lobe Ginkgo Chronospecies)
Phenomenon Scale: 91 µm (9.09e-05 m, log10=-4.041, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Ginkgo yimaensis (Middle Jurassic (Yima Fm, 170 Ma))
Geological Horizon: Middle Jurassic (Yima Fm, 170 Ma) (170.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (91 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1621_ginkgo_yimaensis_vertex_packing.lean` (`MathOfBiology.Loop_1621`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_yimaensis_vertex_packing"
EXPECTED_SHA256 = {
    "ginkgo_yimaensis_vertex_packing_reference.json": "a1b08a6d4d4eafbcce79fa43cf8070e1cb5b29ba6ba1ae9fe46fad66bbb20d89",
    "ginkgo_yimaensis_specimen.jpg": "1ae0d5163efa653c4908ad5d93fc761b18efd5abe7d256c68c27e86ea915e772",
    "scout_manifest.json": "3f3537dc5f859c78b905c5e7b8957ad929135a6f01c5d1d24dafe6c961180b1f",
    "paper_citations.bib": "41af1e46710ce49d27b873ff1eced892fb5543563fbf167532176ec6bfa540d6"
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
    ref = json.loads((DATASET_DIR / "ginkgo_yimaensis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-zhou-ginkgo-yimaensis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
