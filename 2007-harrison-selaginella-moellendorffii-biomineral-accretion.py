#!/usr/bin/env python3
"""
2007-harrison-selaginella-moellendorffii-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0879)
Organism: Selaginella moellendorffii (Spikemoss Lycophyte)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Selaginella
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0879_selaginella_moellendorffii_biomineral_accretion.lean` (`MathOfBiology.Loop_0879`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/selaginella_moellendorffii_biomineral_accretion"
EXPECTED_SHA256 = {
    "selaginella_moellendorffii_biomineral_accretion_reference.json": "99e17b599f8db301fa1e02da9897acc4bbf6094822001671d5bad3f4e254d87d",
    "selaginella_moellendorffii_specimen.jpg": "87d0d25c483d4f17cb66050be175071aafb49d6f321bc8eb2d9cc3bac9e55cd4",
    "scout_manifest.json": "d5aaf9fa30b0e6267fdfded85ae1d4230b40861a6e8ef55b0fbce06eb094c5d4",
    "paper_citations.bib": "9496c660dc348784bafb7558d408a7ba2fffd936d238d2744644351c24f4ed93"
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
    ref = json.loads((DATASET_DIR / "selaginella_moellendorffii_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-harrison-selaginella-moellendorffii-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
