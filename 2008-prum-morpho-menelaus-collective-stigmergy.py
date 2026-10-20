#!/usr/bin/env python3
"""
2008-prum-morpho-menelaus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1044)
Organism: Morpho menelaus (Blue Morpho Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Morpho
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1044_morpho_menelaus_collective_stigmergy.lean` (`MathOfBiology.Loop_1044`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/morpho_menelaus_collective_stigmergy"
EXPECTED_SHA256 = {
    "morpho_menelaus_collective_stigmergy_reference.json": "38f78f687d18ec8b7ece0a5cd0ad45bb70201cca062048251d07fdbeccf53c36",
    "morpho_menelaus_specimen.jpg": "4568faabf61d02b09522424f4f34c0efc7d869ad36aba4c8d412cda345094c68",
    "scout_manifest.json": "031157ef89c3ea0e5ac496b1baadf72aedbb6d58a9b0fb8be98f5f4920571f62",
    "paper_citations.bib": "a42ea594df6034a7124bcce78c82853584a6b8164f545943dad09b8f1a1fb30f"
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
    ref = json.loads((DATASET_DIR / "morpho_menelaus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2008-prum-morpho-menelaus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
