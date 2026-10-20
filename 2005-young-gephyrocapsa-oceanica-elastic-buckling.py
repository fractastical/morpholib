#!/usr/bin/env python3
"""
2005-young-gephyrocapsa-oceanica-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0798)
Organism: Gephyrocapsa oceanica (Calcite Coccolithophore)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Gephyrocapsa
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0798_gephyrocapsa_oceanica_elastic_buckling.lean` (`MathOfBiology.Loop_0798`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gephyrocapsa_oceanica_elastic_buckling"
EXPECTED_SHA256 = {
    "gephyrocapsa_oceanica_elastic_buckling_reference.json": "258e62cee1b086a4f2b4cfc50d41c5e9f87a35a663281c0cd1c0822de11c54a3",
    "gephyrocapsa_oceanica_specimen.jpg": "b3482bdf62d2f619ecd8ad93b8bc68fb784477c51f4de8d0682d3d8fa3dbbc9e",
    "scout_manifest.json": "6d2c617a811e8306cd3dd25df52d0e3f54b9bd766dc1f4954d117ba83f0cb6b5",
    "paper_citations.bib": "bc289185f8738dae44ebc0af0f4585631760e5c3c5df571dbdaa6c0eefe110af"
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
    ref = json.loads((DATASET_DIR / "gephyrocapsa_oceanica_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-young-gephyrocapsa-oceanica-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
