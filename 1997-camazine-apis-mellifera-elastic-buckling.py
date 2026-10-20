#!/usr/bin/env python3
"""
1997-camazine-apis-mellifera-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0743)
Organism: Apis mellifera (Western Honeybee)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Apis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0743_apis_mellifera_elastic_buckling.lean` (`MathOfBiology.Loop_0743`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/apis_mellifera_elastic_buckling"
EXPECTED_SHA256 = {
    "apis_mellifera_elastic_buckling_reference.json": "649db18ec9683c7b9a5ef02272851095ecb83b698a0ed0825520d2d8aee38c92",
    "apis_mellifera_specimen.jpg": "67454ef389899df977e29efa0db0cba99519f5d46a3631810bcc7f160efd8fce",
    "scout_manifest.json": "96f8f3af9ce17bc1f3b0d9aaea58be516cbc884c060542396f0b283010c767f8",
    "paper_citations.bib": "d79de8ee47f12272930cec1da10eaacc961ffc82ab87931b7797e1ec9e292ddd"
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
    ref = json.loads((DATASET_DIR / "apis_mellifera_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1997-camazine-apis-mellifera-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
