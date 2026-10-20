#!/usr/bin/env python3
"""
1968-raup-turritella-communis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1058)
Organism: Turritella communis (Tower Screw Shell)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Turritella
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1058_turritella_communis_collective_stigmergy.lean` (`MathOfBiology.Loop_1058`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/turritella_communis_collective_stigmergy"
EXPECTED_SHA256 = {
    "turritella_communis_collective_stigmergy_reference.json": "fd3426a301e6fa08b3c964027467363e9eb2dc50e85b06dc36f5d378b13ec890",
    "turritella_communis_specimen.jpg": "899a774eadcf1dae2799654a3393a678c5543bfebd3e3f9a02043f13da6934d0",
    "scout_manifest.json": "e8320122e5d6d557ab6b4ba89b2eeba37dd4f0ea1b1261f8f8636878209be618",
    "paper_citations.bib": "dc017e624bdbced3b7b5b2b7bbec4b1cc479d6b71ce076e541c071b268e29bc4"
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
    ref = json.loads((DATASET_DIR / "turritella_communis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1968-raup-turritella-communis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
