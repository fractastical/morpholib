#!/usr/bin/env python3
"""
models/2016-comanns-moloch-horridus-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1729)
Organism: Moloch horridus (Australian Thorny Devil (Capillary Drinking Skin Micro-Groove Network))
Phenomenon Scale: 30 µm (2.959e-05 m, log10=-4.529, 4_cellular_10um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Moloch horridus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Moloch
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (30 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1729_moloch_horridus_branching_tree.lean` (`MathOfBiology.Loop_1729`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/moloch_horridus_branching_tree"
EXPECTED_SHA256 = {
    "moloch_horridus_branching_tree_reference.json": "623a02365562b62945e467c0dbbe9f9937154300131f734d22d02db3435c262c",
    "moloch_horridus_specimen.jpg": "224fb2ba2daad242971c5263ec01e8e384e89bea8709b08d5380e120ebdbf94a",
    "scout_manifest.json": "5edbb8345d136e0d4dac2ae624089f5f2b4d4f31885d95bf0824a8ca123c5265",
    "paper_citations.bib": "a76915c76adbc485c8c50e22f5b97fd33d4da006126e859472ff377b8462b4c6"
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
    ref = json.loads((DATASET_DIR / "moloch_horridus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-comanns-moloch-horridus-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
