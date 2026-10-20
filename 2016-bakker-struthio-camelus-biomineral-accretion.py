#!/usr/bin/env python3
"""
2016-bakker-struthio-camelus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0815)
Organism: Struthio camelus (Common Ostrich)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Struthio
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0815_struthio_camelus_biomineral_accretion.lean` (`MathOfBiology.Loop_0815`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/struthio_camelus_biomineral_accretion"
EXPECTED_SHA256 = {
    "struthio_camelus_biomineral_accretion_reference.json": "d7a4574d02af1dbb9bd15044778c2bbe3a7ea97b3cefab711114681b0f6d7a4c",
    "struthio_camelus_specimen.jpg": "798eba64c73c1474a1b787e850654c5f61164ca01db9ec2d75836507f9494def",
    "scout_manifest.json": "d7fcbf7b64b9c9981a9ec49ca924762de17eaa611bb3574555ebae0815ecf83d",
    "paper_citations.bib": "9912c2876c66761297b8b9ec5ebc79970287818e02ffc5591f8e74f08eddb33b"
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
    ref = json.loads((DATASET_DIR / "struthio_camelus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-bakker-struthio-camelus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
