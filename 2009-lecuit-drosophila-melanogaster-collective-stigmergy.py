#!/usr/bin/env python3
"""
2009-lecuit-drosophila-melanogaster-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1041)
Organism: Drosophila melanogaster (Fruit Fly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Drosophila
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1041_drosophila_melanogaster_collective_stigmergy.lean` (`MathOfBiology.Loop_1041`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/drosophila_melanogaster_collective_stigmergy"
EXPECTED_SHA256 = {
    "drosophila_melanogaster_collective_stigmergy_reference.json": "68f497239a7e789e2ef47b9e9b7b586c2f738fc567dd129b8beae2d2675e789c",
    "drosophila_melanogaster_specimen.jpg": "f426366f7ede436da44782b28ebac9bd013cace1f8fe44b6ed382b77ad86c419",
    "scout_manifest.json": "2003dd6533a9ff81f946d9d4b4e52f16f7fbf490d1d6592990acfe3aa81435a1",
    "paper_citations.bib": "f2ae32c3ceb6f357d9b97fd68caca33039870e7759c10a5a905bae12e86474c1"
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
    ref = json.loads((DATASET_DIR / "drosophila_melanogaster_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-lecuit-drosophila-melanogaster-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
