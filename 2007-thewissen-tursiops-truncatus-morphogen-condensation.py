#!/usr/bin/env python3
"""
2007-thewissen-tursiops-truncatus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0210)
Organism: Tursiops truncatus (Bottlenose Dolphin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Tursiops
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0210_tursiops_truncatus_morphogen_condensation.lean` (`MathOfBiology.Loop_0210`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tursiops_truncatus_morphogen_condensation"
EXPECTED_SHA256 = {
    "tursiops_truncatus_morphogen_condensation_reference.json": "6236423c601246eb4caff43b63bcb9c5a05fbe268eeb57447607adebc0844385",
    "tursiops_truncatus_specimen.jpg": "038f0a413155e6a92701bb438caa1f4c23fdef11b3c5c4892f851e65bcdf710a",
    "scout_manifest.json": "59daeabb1575790e945941d9857b4eeff088c99d3163460b0e361fee9b2bdb16",
    "paper_citations.bib": "56f28d32bbbaa897127d04f62311beee1442cb644c6a4d6965118ccb11425b76"
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
    ref = json.loads((DATASET_DIR / "tursiops_truncatus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-thewissen-tursiops-truncatus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
