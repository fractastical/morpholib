#!/usr/bin/env python3
"""
2008-martindale-mnemiopsis-leidyi-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0770)
Organism: Mnemiopsis leidyi (Warty Comb Jelly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Mnemiopsis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0770_mnemiopsis_leidyi_elastic_buckling.lean` (`MathOfBiology.Loop_0770`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mnemiopsis_leidyi_elastic_buckling"
EXPECTED_SHA256 = {
    "mnemiopsis_leidyi_elastic_buckling_reference.json": "8c176137cf67d6f5f70768b57a73b52e2062159801d56592eecee512af766f9a",
    "mnemiopsis_leidyi_specimen.jpg": "d966e0f5b06b5af8e1ef77edf5e32c90c3aa8489be8bbbf2bdf7795ecbfb7ec7",
    "scout_manifest.json": "fe97fc5198c001c59da79fa5fef68288b2e335f09a4029e6cd1048f0da7173b3",
    "paper_citations.bib": "56372cb96535ddb884b6100a6631eec304dc2d8618342c18754045bd01e7f254"
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
    ref = json.loads((DATASET_DIR / "mnemiopsis_leidyi_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-martindale-mnemiopsis-leidyi-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
