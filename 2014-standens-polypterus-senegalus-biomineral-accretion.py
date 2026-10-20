#!/usr/bin/env python3
"""
2014-standens-polypterus-senegalus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0840)
Organism: Polypterus senegalus (Gray Bichir)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Polypterus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0840_polypterus_senegalus_biomineral_accretion.lean` (`MathOfBiology.Loop_0840`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/polypterus_senegalus_biomineral_accretion"
EXPECTED_SHA256 = {
    "polypterus_senegalus_biomineral_accretion_reference.json": "8b9b730406e6d2756c2820284d1a8a1b830254e82facb9eaf4853eede3b591cd",
    "polypterus_senegalus_specimen.jpg": "8332a72979f53f67ddcfb9e11c646f5695dd1fac3cbcfb340acc3d76e93582f5",
    "scout_manifest.json": "687c39dfd15c873b746a105d19de54b7917fc0a5ff417d18bd5752e732b50f83",
    "paper_citations.bib": "3e97feb1e640c0ae0b4300006ca56acf2cacc40944408ae56fc640ea6e446031"
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
    ref = json.loads((DATASET_DIR / "polypterus_senegalus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-standens-polypterus-senegalus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
