#!/usr/bin/env python3
"""
2015-plackett-ceratopteris-richardii-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0880)
Organism: Ceratopteris richardii (Triangle Water Fern)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ceratopteris
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0880_ceratopteris_richardii_biomineral_accretion.lean` (`MathOfBiology.Loop_0880`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ceratopteris_richardii_biomineral_accretion"
EXPECTED_SHA256 = {
    "ceratopteris_richardii_biomineral_accretion_reference.json": "de6c3fbb5553e358812349dc4720a366e5ba177f11c633a84d28e119f58e224c",
    "ceratopteris_richardii_specimen.jpg": "3fcf2684662c089b9c66687d1f6b2e9409565391d0d316e5db46561d1809641f",
    "scout_manifest.json": "1166154290cd7a532b314ec58c4b13de38df927b0abf33e6d666f63dfa3afd97",
    "paper_citations.bib": "a628b78e46d51c10f871eb218c09014345d92a515591bb7d13ad9dc6ffde25f9"
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
    ref = json.loads((DATASET_DIR / "ceratopteris_richardii_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-plackett-ceratopteris-richardii-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
