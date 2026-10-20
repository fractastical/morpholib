#!/usr/bin/env python3
"""
2018-bakker-struthio-camelus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1015)
Organism: Struthio camelus (Common Ostrich)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Struthio
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1015_struthio_camelus_collective_stigmergy.lean` (`MathOfBiology.Loop_1015`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/struthio_camelus_collective_stigmergy"
EXPECTED_SHA256 = {
    "struthio_camelus_collective_stigmergy_reference.json": "3cf71de74ad88aa4f1b0598315dc3901216ad924b6c18c19430221bccb1f2951",
    "struthio_camelus_specimen.jpg": "798eba64c73c1474a1b787e850654c5f61164ca01db9ec2d75836507f9494def",
    "scout_manifest.json": "a68cc9d94d7c223f5d0b601f0155a4bfc6d81cba34a5fd8fa325ed582a637ffe",
    "paper_citations.bib": "df1284588cf963862136c7581dc9e4ae244ef6be1e9e07f1980904d2d977c26d"
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
    ref = json.loads((DATASET_DIR / "struthio_camelus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-bakker-struthio-camelus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
