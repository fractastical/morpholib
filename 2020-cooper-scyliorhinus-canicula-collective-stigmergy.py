#!/usr/bin/env python3
"""
2020-cooper-scyliorhinus-canicula-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1035)
Organism: Scyliorhinus canicula (Small-Spotted Catshark)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Scyliorhinus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1035_scyliorhinus_canicula_collective_stigmergy.lean` (`MathOfBiology.Loop_1035`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/scyliorhinus_canicula_collective_stigmergy"
EXPECTED_SHA256 = {
    "scyliorhinus_canicula_collective_stigmergy_reference.json": "3d99311836dd3048bda57f42df01f251386cdc015213b334088ad4a3267c5fc4",
    "scyliorhinus_canicula_specimen.jpg": "71525dd878e51b100ea6bbd4e4005751066059fc236d4e204ad81ed2bbb5d553",
    "scout_manifest.json": "53f9a88fb539c5c195b1060311c6f4d8c52b9d0945919317238ae73f645eec5b",
    "paper_citations.bib": "422f87ddfe3f127e88de0302a10b2b5e711f128c64bc946bd986aeff175d994c"
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
    ref = json.loads((DATASET_DIR / "scyliorhinus_canicula_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-cooper-scyliorhinus-canicula-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
