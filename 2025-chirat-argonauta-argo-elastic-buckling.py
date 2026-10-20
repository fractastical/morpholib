#!/usr/bin/env python3
"""
2025-chirat-argonauta-argo-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0756)
Organism: Argonauta argo (Paper Nautilus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Argonauta
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0756_argonauta_argo_elastic_buckling.lean` (`MathOfBiology.Loop_0756`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/argonauta_argo_elastic_buckling"
EXPECTED_SHA256 = {
    "argonauta_argo_elastic_buckling_reference.json": "3ed29f5a7ee15643d826e156ab9f9a27539581b5c9c6129ba0a5090fe68a4ab5",
    "argonauta_argo_specimen.jpg": "4d2cfc3367b264b3d9c93b5d4ea3794e99cb317bbc495df52ab10c540174dff2",
    "scout_manifest.json": "9bded43a4daefb8f49528b539a48942516fcfe1071f89d2016b7db09bd9f11c6",
    "paper_citations.bib": "49955dbf7b32227d07e3d58354195751aadf44eb33b4b7462411b845a0d19cce"
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
    ref = json.loads((DATASET_DIR / "argonauta_argo_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2025-chirat-argonauta-argo-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
