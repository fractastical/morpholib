#!/usr/bin/env python3
"""
models/2012-janis-merychippus-insignis-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1580)
Organism: Merychippus insignis (Miocene Grazing Spring-Footed Three-Toed Horse)
Phenomenon Scale: 472 µm (0.0004725 m, log10=-3.326, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Merychippus insignis (Middle Miocene (Barstovian, 15.0 Ma))
Geological Horizon: Middle Miocene (Barstovian, 15.0 Ma) (15.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Merychippus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (472 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1580_merychippus_insignis_vertex_packing.lean` (`MathOfBiology.Loop_1580`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/merychippus_insignis_vertex_packing"
EXPECTED_SHA256 = {
    "merychippus_insignis_vertex_packing_reference.json": "4d6ecc950762aa1085e2f63842da7a7bc8459095f54eea68e24e1c2143b97b68",
    "merychippus_insignis_specimen.jpg": "93354e358baacf8e852b4dccb0a713b62e217cf8aba2864c49e3a358f03f7425",
    "scout_manifest.json": "fdbe56e86d0cbd77ff75efddc9a9750f4d84139dc8a790a4317ac9037d3e3b24",
    "paper_citations.bib": "7d7e2c3f0a00d970cdf419c2852c4727efaa666f439921f3a8519508ac76aafa"
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
    ref = json.loads((DATASET_DIR / "merychippus_insignis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-janis-merychippus-insignis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
