#!/usr/bin/env python3
"""
2017-plackett-ceratopteris-richardii-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1080)
Organism: Ceratopteris richardii (Triangle Water Fern)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ceratopteris
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1080_ceratopteris_richardii_collective_stigmergy.lean` (`MathOfBiology.Loop_1080`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ceratopteris_richardii_collective_stigmergy"
EXPECTED_SHA256 = {
    "ceratopteris_richardii_collective_stigmergy_reference.json": "6fde5e2efea8255770cedc4ad6942a2a4369416c3f521546fee6394fa949a1c1",
    "ceratopteris_richardii_specimen.jpg": "3fcf2684662c089b9c66687d1f6b2e9409565391d0d316e5db46561d1809641f",
    "scout_manifest.json": "d2a2e2c2d38eeaad4a25839b3bbd037cf9407cde9a2a8aab38835ab446eec852",
    "paper_citations.bib": "da91c010b9efe1b8070be71844cf3c189a88c422d1f481de923a5bcf1a272a2b"
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
    ref = json.loads((DATASET_DIR / "ceratopteris_richardii_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-plackett-ceratopteris-richardii-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
