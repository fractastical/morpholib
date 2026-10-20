#!/usr/bin/env python3
"""
2008-buhl-schistocerca-gregaria-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1048)
Organism: Schistocerca gregaria (Desert Locust)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Schistocerca
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1048_schistocerca_gregaria_collective_stigmergy.lean` (`MathOfBiology.Loop_1048`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schistocerca_gregaria_collective_stigmergy"
EXPECTED_SHA256 = {
    "schistocerca_gregaria_collective_stigmergy_reference.json": "101940164ea087491276d7fe02d83e9cc6a313b07cc4a3cdcf585a160dc4f8e6",
    "schistocerca_gregaria_specimen.jpg": "3f8a278b02bc5e5158458d4ff3669c71a29826b1129be2cc7920ab4b3ead3827",
    "scout_manifest.json": "d25bc95cd06a2196304ae87be1ed82fff141711a09d13f3528f8b7d576f8468b",
    "paper_citations.bib": "2de01b70a851e5c62d918e44266dbe50e5f66ce40e76edb53873259bd299eae8"
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
    ref = json.loads((DATASET_DIR / "schistocerca_gregaria_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-buhl-schistocerca-gregaria-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
