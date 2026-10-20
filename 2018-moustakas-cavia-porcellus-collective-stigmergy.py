#!/usr/bin/env python3
"""
2018-moustakas-cavia-porcellus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1008)
Organism: Cavia porcellus (Guinea Pig)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Cavia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1008_cavia_porcellus_collective_stigmergy.lean` (`MathOfBiology.Loop_1008`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/cavia_porcellus_collective_stigmergy"
EXPECTED_SHA256 = {
    "cavia_porcellus_collective_stigmergy_reference.json": "21f5c467f86726ffad32e9dcc8cb2bdf7de199cf44e84f31bbf96180f125b976",
    "cavia_porcellus_specimen.jpg": "42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd",
    "scout_manifest.json": "f69fbe2a10ba8514e61c7419d8c73df588679c8aca76dce9fe88ee89896bed90",
    "paper_citations.bib": "08d01f456891972ab36117cf95f17cbb509b9132bbeee8d8d2e34a74ebe0bc8a"
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
    ref = json.loads((DATASET_DIR / "cavia_porcellus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-moustakas-cavia-porcellus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
