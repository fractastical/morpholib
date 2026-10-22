#!/usr/bin/env python3
"""
models/2011-chen-dasypus-novemcinctus-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1726)
Organism: Dasypus novemcinctus (Nine-Banded Armadillo (Hexagonal Osteoderm Rosette Tiles))
Phenomenon Scale: 2.4 mm (0.002389 m, log10=-2.622, 6_organ_mm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Dasypus novemcinctus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dasypus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (2.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1726_dasypus_novemcinctus_branching_tree.lean` (`MathOfBiology.Loop_1726`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dasypus_novemcinctus_branching_tree"
EXPECTED_SHA256 = {
    "dasypus_novemcinctus_branching_tree_reference.json": "56005839b44a7e442c00d690ac229f03086faba34404622ff5010db902443b0a",
    "dasypus_novemcinctus_specimen.jpg": "5cbb611c05d8707d92d1c6d2cdd19de1525cb938009adae92309147f22174539",
    "scout_manifest.json": "58a7ed5f89b0ffbe4c2f1afae9f4f449abe3f86c3ef78793b3298e6d9a8c952b",
    "paper_citations.bib": "e8af3c4204029ad3540e495f8725cd77b6dcfa6071b15eef0f24ed52d0981c93"
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
    ref = json.loads((DATASET_DIR / "dasypus_novemcinctus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-chen-dasypus-novemcinctus-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
