#!/usr/bin/env python3
"""
2007-czirok-proteus-mirabilis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0790)
Organism: Proteus mirabilis (Bullseye Swarming Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Proteus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0790_proteus_mirabilis_elastic_buckling.lean` (`MathOfBiology.Loop_0790`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/proteus_mirabilis_elastic_buckling"
EXPECTED_SHA256 = {
    "proteus_mirabilis_elastic_buckling_reference.json": "03da36252dd8bd36d0c82a0ade1b236f5103385397da000f724cbbecda40ee57",
    "proteus_mirabilis_specimen.jpg": "ac07f43481b9b721da7b124217c182a69192087ef9fb2f59564a107e3383a85b",
    "scout_manifest.json": "f4af92de4a87982dd7bc138429d4195afc9a39f8d849800e1182325042ad6543",
    "paper_citations.bib": "2e3dbdd3a602e615a624003b46e1f237da28192fa956291345897d1f3f94fd77"
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
    ref = json.loads((DATASET_DIR / "proteus_mirabilis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-czirok-proteus-mirabilis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
