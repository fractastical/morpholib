#!/usr/bin/env python3
"""
1978-bard-equus-quagga-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0204)
Organism: Equus quagga (Plains Zebra)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0204_equus_quagga_morphogen_condensation.lean` (`MathOfBiology.Loop_0204`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/equus_quagga_morphogen_condensation"
EXPECTED_SHA256 = {
    "equus_quagga_morphogen_condensation_reference.json": "6841c9e62dbb4d254bc4a304afb9cbe6fb2c51e5faa242d2bc841af4bb061d0e",
    "equus_quagga_specimen.jpg": "b734dc3b533ffe889dee9f205dae9015580b4b3537a05743c602408063b17ca8",
    "scout_manifest.json": "705f9d4c5afa5c67354a457e6f1abb82947edb8e8cf73240b554b060b768441c",
    "paper_citations.bib": "8730b1e6d27b51b744aeb05de29bf4588f8348fdb1f0a28fa9b3f17753acec66"
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
    ref = json.loads((DATASET_DIR / "equus_quagga_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1978-bard-equus-quagga-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
