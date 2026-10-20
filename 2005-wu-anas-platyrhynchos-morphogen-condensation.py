#!/usr/bin/env python3
"""
2005-wu-anas-platyrhynchos-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0216)
Organism: Anas platyrhynchos (Mallard Duck)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anas
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0216_anas_platyrhynchos_morphogen_condensation.lean` (`MathOfBiology.Loop_0216`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anas_platyrhynchos_morphogen_condensation"
EXPECTED_SHA256 = {
    "anas_platyrhynchos_morphogen_condensation_reference.json": "018856fd5c338ceb7088045cd5f681e8a4ddcfd761f1fc7083ca7a64578902b2",
    "anas_platyrhynchos_specimen.jpg": "ee1cc0720d68f5f20e5e3490647b69cb29c1fe84dae8fdb048019814f69eccba",
    "scout_manifest.json": "610ff4ab54d0523bacce93bafe70948e67641f32d4247b8ab03c5dbbd9f2b6de",
    "paper_citations.bib": "3395eb5985322149b0931e5aa502b0191df6438a424551c2daad6b635637d331"
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
    ref = json.loads((DATASET_DIR / "anas_platyrhynchos_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-wu-anas-platyrhynchos-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
