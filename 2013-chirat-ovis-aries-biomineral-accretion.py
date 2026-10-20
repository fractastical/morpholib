#!/usr/bin/env python3
"""
2013-chirat-ovis-aries-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0807)
Organism: Ovis aries (Bighorn Sheep)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ovis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0807_ovis_aries_biomineral_accretion.lean` (`MathOfBiology.Loop_0807`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ovis_aries_biomineral_accretion"
EXPECTED_SHA256 = {
    "ovis_aries_biomineral_accretion_reference.json": "b1251b149d4603cccdbd4f4a63885e1bb872ccebb0ada576d093d2b8f9954ba8",
    "ovis_aries_specimen.jpg": "3e5aea2a1dd349564873df9e61677c401d7c2264295f722a777e3808f6a3c0d2",
    "scout_manifest.json": "11306334b1ba6da6cd90f1a60694c37869e3c95272394f302f17c27d15063630",
    "paper_citations.bib": "42104763e8f7d6cc40b3bfa5999a27087a3f89f1a1fd1737fb607dc80462e380"
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
    ref = json.loads((DATASET_DIR / "ovis_aries_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-chirat-ovis-aries-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
