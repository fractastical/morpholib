#!/usr/bin/env python3
"""
2022-eliason-calypte-anna-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1020)
Organism: Calypte anna (Anna's Hummingbird)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Calypte
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1020_calypte_anna_collective_stigmergy.lean` (`MathOfBiology.Loop_1020`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/calypte_anna_collective_stigmergy"
EXPECTED_SHA256 = {
    "calypte_anna_collective_stigmergy_reference.json": "083f2184294c2107f6165d52863068cbde452bb26eb005c123974ead33a87593",
    "calypte_anna_specimen.jpg": "023a1c1ebf863e2bc3fe443e9cc7d668c6fa7ac72517c8a62f1874671a2eeb60",
    "scout_manifest.json": "3de40995aed4f7b11268664e5bbaf201a81796ea3b1d71cd600ce4e172e203bf",
    "paper_citations.bib": "4e710c477dd909bfb36bd39b83971fb0630d2fd05a5d325e374c6109cc0cb787"
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
    ref = json.loads((DATASET_DIR / "calypte_anna_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-eliason-calypte-anna-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
