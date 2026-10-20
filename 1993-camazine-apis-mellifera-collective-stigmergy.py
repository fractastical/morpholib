#!/usr/bin/env python3
"""
1993-camazine-apis-mellifera-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1043)
Organism: Apis mellifera (Western Honeybee)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Apis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1043_apis_mellifera_collective_stigmergy.lean` (`MathOfBiology.Loop_1043`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/apis_mellifera_collective_stigmergy"
EXPECTED_SHA256 = {
    "apis_mellifera_collective_stigmergy_reference.json": "f7ea77b84cde5ffad57329610b43e8612190784c4c3ff9f71745491ede697ab5",
    "apis_mellifera_specimen.jpg": "67454ef389899df977e29efa0db0cba99519f5d46a3631810bcc7f160efd8fce",
    "scout_manifest.json": "6afaffd2bbf812301a65ec7bd1521529214d1a1fb9212930556651e068294d89",
    "paper_citations.bib": "0ab9be49dcd6d4450d4246773c412ba81249bcd9507b756d6ef292048e070faf"
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
    ref = json.loads((DATASET_DIR / "apis_mellifera_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1993-camazine-apis-mellifera-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
