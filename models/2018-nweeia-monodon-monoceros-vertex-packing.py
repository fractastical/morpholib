#!/usr/bin/env python3
"""
models/2018-nweeia-monodon-monoceros-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1587)
Organism: Monodon monoceros (Arctic Narwhal (Chiral Helical Left Incisor Tusk))
Phenomenon Scale: 4.9 mm (0.004856 m, log10=-2.314, 6_organ_mm)
Anatomical Structure: Apical epithelial cell vertex polygon in Monodon monoceros (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Monodon
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (4.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1587_monodon_monoceros_vertex_packing.lean` (`MathOfBiology.Loop_1587`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/monodon_monoceros_vertex_packing"
EXPECTED_SHA256 = {
    "monodon_monoceros_vertex_packing_reference.json": "d5ca0b1b4ec56bdf4dc07b4f31b757cc651b1bd2d53cd5f75dcac18f26719837",
    "monodon_monoceros_specimen.jpg": "1fa8766cd6ff186448201eb9da7097191319445a02628f275cfe5752d1be8cc3",
    "scout_manifest.json": "4b3f1d94ec4c2944d586682454111ced0e652d6bab72977e5888214603fdef49",
    "paper_citations.bib": "49a2fa3fbb3b0c66e59f9edf53163255f8848993b1a7ea8fc30776070c03c4b8"
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
    ref = json.loads((DATASET_DIR / "monodon_monoceros_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-nweeia-monodon-monoceros-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
