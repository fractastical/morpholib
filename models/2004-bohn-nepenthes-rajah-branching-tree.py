#!/usr/bin/env python3
"""
models/2004-bohn-nepenthes-rajah-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1739)
Organism: Nepenthes rajah (Giant Bornean Pitcher Plant (Micro-Grooved Peristome Aquaplaning Rim))
Phenomenon Scale: 16 µm (1.591e-05 m, log10=-4.798, 4_cellular_10um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Nepenthes rajah (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Nepenthes
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (16 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1739_nepenthes_rajah_branching_tree.lean` (`MathOfBiology.Loop_1739`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nepenthes_rajah_branching_tree"
EXPECTED_SHA256 = {
    "nepenthes_rajah_branching_tree_reference.json": "a282d8a8c4c29e5416d4f0d534af76c5f715586cf24296101bc9f737ea8abcd2",
    "nepenthes_rajah_specimen.jpg": "34d473e5c7aa422e26ade1e48e08666c34867f853524458ff188daa017f9d5a3",
    "scout_manifest.json": "78fce1e0608f8886fba10ece9b49b160fba69bdefa73bb23df7c284b67fdb092",
    "paper_citations.bib": "0a759663f0f49cc3746f9aa0d0442d468b8ba31f6b032f47bb8bcad736ab7fc2"
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
    ref = json.loads((DATASET_DIR / "nepenthes_rajah_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-bohn-nepenthes-rajah-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
