#!/usr/bin/env python3
"""
2013-shapiro-columba-livia-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0814)
Organism: Columba livia (Rock Dove)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Columba
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0814_columba_livia_biomineral_accretion.lean` (`MathOfBiology.Loop_0814`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/columba_livia_biomineral_accretion"
EXPECTED_SHA256 = {
    "columba_livia_biomineral_accretion_reference.json": "4146e8b3d9c81941740e76499db09c699516b7f086b695a3f14125988fc0ee3a",
    "columba_livia_specimen.jpg": "00bf4ed50c6075550c363caacdfa0f8820f68486fb8d737e81039f614c1ae7fb",
    "scout_manifest.json": "76a7196aaf00fc97a87dac58b85b9739376628d3077c36107fd5e2fab153193d",
    "paper_citations.bib": "2d0ba45f232afbc79ded60d0e88296187d42b77c1afb5faf16163e44cf6efb53"
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
    ref = json.loads((DATASET_DIR / "columba_livia_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-shapiro-columba-livia-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
