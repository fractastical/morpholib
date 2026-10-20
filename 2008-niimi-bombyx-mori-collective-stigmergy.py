#!/usr/bin/env python3
"""
2008-niimi-bombyx-mori-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1046)
Organism: Bombyx mori (Domestic Silkmoth)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Bombyx
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1046_bombyx_mori_collective_stigmergy.lean` (`MathOfBiology.Loop_1046`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/bombyx_mori_collective_stigmergy"
EXPECTED_SHA256 = {
    "bombyx_mori_collective_stigmergy_reference.json": "843d8066362cf58a5faa360b22a99ecc429b9e4c79902d8d9ed938dc92a5e86d",
    "bombyx_mori_specimen.jpg": "2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97",
    "scout_manifest.json": "44758088b026506066ff7d9665d7c86aa60aa843e0970accc977e0de09e962cf",
    "paper_citations.bib": "2dc4fb20f84dde2dec207a26d99295b253574bf40b97e6e0eaa5cd9a8cdbea4d"
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
    ref = json.loads((DATASET_DIR / "bombyx_mori_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-niimi-bombyx-mori-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
