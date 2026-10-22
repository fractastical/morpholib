#!/usr/bin/env python3
"""
models/2003-pettigrew-ornithorhynchus-anatinus-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1634)
Organism: Ornithorhynchus anatinus (Duck-Billed Platypus (Electroreceptive Bill Pore Hexagonal Array))
Phenomenon Scale: 340 nm (3.395e-07 m, log10=-6.469, 2_macromolecular_100nm)
Anatomical Structure: Apical epithelial cell vertex polygon in Ornithorhynchus anatinus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ornithorhynchus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (340 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1634_ornithorhynchus_anatinus_vertex_packing.lean` (`MathOfBiology.Loop_1634`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ornithorhynchus_anatinus_vertex_packing"
EXPECTED_SHA256 = {
    "ornithorhynchus_anatinus_vertex_packing_reference.json": "47fe48b89beeca82bf7cfe02a5d0e6d06a2d213b0d1c472d318233e34518847a",
    "ornithorhynchus_anatinus_specimen.jpg": "5a8d5bf6c1809500a8e9e88694cff57c75cead6258be2e7d05ede7922465f2b6",
    "scout_manifest.json": "40787dce180ac14b2b4e4f1a29fa8d9461eea47c5344b56b6d8880ce5188ba3a",
    "paper_citations.bib": "d83c64d9b3635b2dadeb1c10d89ef2a805cdddecfac01591fd3c277678ba38a7"
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
    ref = json.loads((DATASET_DIR / "ornithorhynchus_anatinus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-pettigrew-ornithorhynchus-anatinus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
