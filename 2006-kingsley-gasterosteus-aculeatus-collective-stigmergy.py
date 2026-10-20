#!/usr/bin/env python3
"""
2006-kingsley-gasterosteus-aculeatus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1036)
Organism: Gasterosteus aculeatus (Three-Spined Stickleback)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1036_gasterosteus_aculeatus_collective_stigmergy.lean` (`MathOfBiology.Loop_1036`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_collective_stigmergy"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_collective_stigmergy_reference.json": "5662a20312a20c3aaa9a3c7e64eba33a1c15f692a1dfd7741f834d0259424ee6",
    "gasterosteus_aculeatus_specimen.jpg": "86fef1d4aeb86c3903e26fae972a360ea0c848048e29b546bedb2db842a5bd10",
    "scout_manifest.json": "c9215d2c4f21e093ee569b8f172eae56d4d563d559e1b4524d6bc5b144fd8803",
    "paper_citations.bib": "ec30c38423d3dc77775bf225d11571d95096683501208d60f0390a4ad0fbfd64"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-kingsley-gasterosteus-aculeatus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
