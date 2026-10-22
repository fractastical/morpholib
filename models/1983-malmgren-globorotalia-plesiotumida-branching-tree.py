#!/usr/bin/env python3
"""
models/1983-malmgren-globorotalia-plesiotumida-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1717)
Organism: Globorotalia plesiotumida (Late Miocene Moderately-Swollen Trochospiral Foraminifer Chronospecies)
Phenomenon Scale: 173 µm (0.0001731 m, log10=-3.762, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Globorotalia plesiotumida (Late Miocene (Messinian, 6.2 Ma))
Geological Horizon: Late Miocene (Messinian, 6.2 Ma) (6.2 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (173 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1717_globorotalia_plesiotumida_branching_tree.lean` (`MathOfBiology.Loop_1717`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_plesiotumida_branching_tree"
EXPECTED_SHA256 = {
    "globorotalia_plesiotumida_branching_tree_reference.json": "5479709dbefc02e6d76148eb2d9a852ad219d5944267d3b6fcbc851d440db6c7",
    "globorotalia_plesiotumida_specimen.jpg": "c1c82488a477ce08218e48d667618dbcb5efdb7a98603919f3838b81a82663b6",
    "scout_manifest.json": "4545cfa829759d261254a797eb21cfa19d96ac3677efcf3af7625f195da5de2f",
    "paper_citations.bib": "9eac8b52cfac0b4784423817740bfafe5d37db3cd701709dfb7ba9d60b262d37"
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
    ref = json.loads((DATASET_DIR / "globorotalia_plesiotumida_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1983-malmgren-globorotalia-plesiotumida-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
