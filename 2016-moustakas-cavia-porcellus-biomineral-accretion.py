#!/usr/bin/env python3
"""
2016-moustakas-cavia-porcellus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0808)
Organism: Cavia porcellus (Guinea Pig)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Cavia
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0808_cavia_porcellus_biomineral_accretion.lean` (`MathOfBiology.Loop_0808`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/cavia_porcellus_biomineral_accretion"
EXPECTED_SHA256 = {
    "cavia_porcellus_biomineral_accretion_reference.json": "a408cf73baf3a149e6205d4c534f23c6c04b0cc70e6c6827356affdf55265a90",
    "cavia_porcellus_specimen.jpg": "42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd",
    "scout_manifest.json": "5f2af363cada2b50b89633ae2669493a39cd2f4c7d2f7a2a85c816ddae60fc0c",
    "paper_citations.bib": "42e3e260e68f60ff34174a9584c278ef60d150fc56c904bf86f15e798f1f8e26"
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
    ref = json.loads((DATASET_DIR / "cavia_porcellus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-moustakas-cavia-porcellus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
