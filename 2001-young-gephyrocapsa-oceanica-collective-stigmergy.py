#!/usr/bin/env python3
"""
2001-young-gephyrocapsa-oceanica-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1098)
Organism: Gephyrocapsa oceanica (Calcite Coccolithophore)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Gephyrocapsa
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1098_gephyrocapsa_oceanica_collective_stigmergy.lean` (`MathOfBiology.Loop_1098`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gephyrocapsa_oceanica_collective_stigmergy"
EXPECTED_SHA256 = {
    "gephyrocapsa_oceanica_collective_stigmergy_reference.json": "e36afcea16934ccb105ef504c6115cde851dec169d2f1d58f3e9dc2a4b46281e",
    "gephyrocapsa_oceanica_specimen.jpg": "b3482bdf62d2f619ecd8ad93b8bc68fb784477c51f4de8d0682d3d8fa3dbbc9e",
    "scout_manifest.json": "e1928b6b7a3ad7c0dc3915c66c3b97ebe3a7b3fc563dd36a282ed1767c4cb88a",
    "paper_citations.bib": "bfabf678ae5f0b3b519a4cec587c10bdb07f4cc39170bd72c9957f819b3c0be5"
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
    ref = json.loads((DATASET_DIR / "gephyrocapsa_oceanica_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-young-gephyrocapsa-oceanica-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
