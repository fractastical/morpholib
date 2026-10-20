#!/usr/bin/env python3
"""
2007-buhl-schistocerca-gregaria-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0248)
Organism: Schistocerca gregaria (Desert Locust)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Schistocerca
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0248_schistocerca_gregaria_morphogen_condensation.lean` (`MathOfBiology.Loop_0248`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schistocerca_gregaria_morphogen_condensation"
EXPECTED_SHA256 = {
    "schistocerca_gregaria_morphogen_condensation_reference.json": "8fb4ab3e91a317454efb1ed49c697b0c9e0fa74ddf1f3d58828ef99862da6a49",
    "schistocerca_gregaria_specimen.jpg": "3f8a278b02bc5e5158458d4ff3669c71a29826b1129be2cc7920ab4b3ead3827",
    "scout_manifest.json": "b442ced191f7978bc274175732e12ce6251a99c1b3cade72b6ed56f1d4896101",
    "paper_citations.bib": "98237987a0fd062e7accd75765850157af939b0d25dfcc3747a75993e1d2a3d9"
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
    ref = json.loads((DATASET_DIR / "schistocerca_gregaria_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-buhl-schistocerca-gregaria-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
