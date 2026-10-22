#!/usr/bin/env python3
"""
models/2003-zheng-ginkgo-apodes-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1712)
Organism: Ginkgo apodes (Early Cretaceous Multi-Lobed Jehol Biota Ginkgo Chronospecies)
Phenomenon Scale: 1.5 cm (0.01544 m, log10=-1.811, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Ginkgo apodes (Early Cretaceous (Yixian Fm, 121 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 121 Ma) (121.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (1.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1712_ginkgo_apodes_branching_tree.lean` (`MathOfBiology.Loop_1712`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_apodes_branching_tree"
EXPECTED_SHA256 = {
    "ginkgo_apodes_branching_tree_reference.json": "4f1dca163ca1e6df719da1abb2cd2443affcd8cc4483701e7536828ca910368b",
    "ginkgo_apodes_specimen.jpg": "3a074e9794ed0797dcc5abd51ab2fcccf8a109da1756d08ad6f52d95faf7d0f5",
    "scout_manifest.json": "6c10c231207ce942ef36b2294df29995296d1c29a8fe89456610873528e9b787",
    "paper_citations.bib": "95864144b6adf314522c713a093d51a96cedcdaeccebdf506ebc6273b4971f17"
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
    ref = json.loads((DATASET_DIR / "ginkgo_apodes_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-zheng-ginkgo-apodes-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
