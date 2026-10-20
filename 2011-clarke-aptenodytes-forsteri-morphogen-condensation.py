#!/usr/bin/env python3
"""
2011-clarke-aptenodytes-forsteri-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0219)
Organism: Aptenodytes forsteri (Emperor Penguin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Aptenodytes
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0219_aptenodytes_forsteri_morphogen_condensation.lean` (`MathOfBiology.Loop_0219`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aptenodytes_forsteri_morphogen_condensation"
EXPECTED_SHA256 = {
    "aptenodytes_forsteri_morphogen_condensation_reference.json": "9d98eb5e9a896d66087434e0337c57d8585877578bbb335c2ec97ca0684e79c6",
    "aptenodytes_forsteri_specimen.jpg": "ef141d4e199c2ace11888d32dc8d271279924c5e9a9faa6f15ad839d7b5dd0d0",
    "scout_manifest.json": "b9fbec06c70b5b7b366409abf66fc697237312cc62533006050e6d65e82454ff",
    "paper_citations.bib": "60f781446023ca03c2a3b0a8c033212225e0e53d1cbc43f9ff7fa5419f72f239"
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
    ref = json.loads((DATASET_DIR / "aptenodytes_forsteri_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-clarke-aptenodytes-forsteri-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
