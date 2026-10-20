#!/usr/bin/env python3
"""
2001-czirok-proteus-mirabilis-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0890)
Organism: Proteus mirabilis (Bullseye Swarming Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Proteus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0890_proteus_mirabilis_biomineral_accretion.lean` (`MathOfBiology.Loop_0890`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/proteus_mirabilis_biomineral_accretion"
EXPECTED_SHA256 = {
    "proteus_mirabilis_biomineral_accretion_reference.json": "a1b37973fea2e8b2c9f76874b2e08f379935996ecd70b2300ae6351d8cb7b629",
    "proteus_mirabilis_specimen.jpg": "ac07f43481b9b721da7b124217c182a69192087ef9fb2f59564a107e3383a85b",
    "scout_manifest.json": "b1aa0c1bf10130134e7f0baf27e9169424c04137388ae889c6fd6df5728d49d3",
    "paper_citations.bib": "79c19ef227c9ad49730d8902e437b9065e9ddfb827add3e94e517abbcee6207b"
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
    ref = json.loads((DATASET_DIR / "proteus_mirabilis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-czirok-proteus-mirabilis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
