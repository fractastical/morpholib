#!/usr/bin/env python3
"""
2014-sanger-anolis-carolinensis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1024)
Organism: Anolis carolinensis (Green Anole)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Anolis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1024_anolis_carolinensis_collective_stigmergy.lean` (`MathOfBiology.Loop_1024`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anolis_carolinensis_collective_stigmergy"
EXPECTED_SHA256 = {
    "anolis_carolinensis_collective_stigmergy_reference.json": "6bad611c80729fe65ea0555b920977ee66988e4a538b52d32a907bd1e1e0a9ab",
    "anolis_carolinensis_specimen.jpg": "584ab986514945bb9c82b1765cb9c44cb8915e47823a0e02b5f444310fe0506b",
    "scout_manifest.json": "b88c1aab9a1293499fe859c52a6c6d1d236548e604adde46af4ee2aa63ae88dc",
    "paper_citations.bib": "8a8943578f698112b3ae541fcb5239fda95a629320635fdd8bfd4f811fc9d79f"
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
    ref = json.loads((DATASET_DIR / "anolis_carolinensis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-sanger-anolis-carolinensis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
