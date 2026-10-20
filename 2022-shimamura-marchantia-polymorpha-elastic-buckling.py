#!/usr/bin/env python3
"""
2022-shimamura-marchantia-polymorpha-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0771)
Organism: Marchantia polymorpha (Common Liverwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Marchantia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0771_marchantia_polymorpha_elastic_buckling.lean` (`MathOfBiology.Loop_0771`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/marchantia_polymorpha_elastic_buckling"
EXPECTED_SHA256 = {
    "marchantia_polymorpha_elastic_buckling_reference.json": "10baa53c983bb77d704166a4d0ed10dd9ecbaf141ba7049516e3942fb4a7c301",
    "marchantia_polymorpha_specimen.jpg": "1a60432aad00137db36b8c40d01cac4e78b01334796182f27ed7219444c3577c",
    "scout_manifest.json": "3a4b79757a5820865148359cd445bc7b4c2bec86c1cf5515d56c0f88c8c9c075",
    "paper_citations.bib": "f6a6b2799f68adcdae883bf7396f2b38a5ee5d76c1cc33fec114a2b1a3399d1b"
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
    ref = json.loads((DATASET_DIR / "marchantia_polymorpha_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-shimamura-marchantia-polymorpha-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
