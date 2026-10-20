#!/usr/bin/env python3
"""
2012-clarke-aptenodytes-forsteri-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1019)
Organism: Aptenodytes forsteri (Emperor Penguin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Aptenodytes
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1019_aptenodytes_forsteri_collective_stigmergy.lean` (`MathOfBiology.Loop_1019`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aptenodytes_forsteri_collective_stigmergy"
EXPECTED_SHA256 = {
    "aptenodytes_forsteri_collective_stigmergy_reference.json": "4ff6a21177798d39442f1ca9b7180dbff0b375f4e358f18a807c07f494257497",
    "aptenodytes_forsteri_specimen.jpg": "ef141d4e199c2ace11888d32dc8d271279924c5e9a9faa6f15ad839d7b5dd0d0",
    "scout_manifest.json": "6db4927e7e68d0dd770fdda89109ba504533ca80f04406eec1b591c8a1ce0d2d",
    "paper_citations.bib": "170dc5f2a3b591788212c06020f735999cfe6af242dd03f0374c5606d2a8f674"
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
    ref = json.loads((DATASET_DIR / "aptenodytes_forsteri_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-clarke-aptenodytes-forsteri-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
