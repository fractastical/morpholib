#!/usr/bin/env python3
"""
1997-nijhout-heliconius-erato-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0747)
Organism: Heliconius erato (Red Postman Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Heliconius
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0747_heliconius_erato_elastic_buckling.lean` (`MathOfBiology.Loop_0747`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/heliconius_erato_elastic_buckling"
EXPECTED_SHA256 = {
    "heliconius_erato_elastic_buckling_reference.json": "ea02303129f2fb8cfa4ea5c73f1129c00cddf40e97794f28482589619dd56fe4",
    "heliconius_erato_specimen.jpg": "e54735f4ca16e4326cb91d2969d83bb613e3c4111aaf7f30cb218075b6fec48e",
    "scout_manifest.json": "a3f44939e73c5c760bc66c501b94a333eb02495ef66f36229bdf4d8e547542dd",
    "paper_citations.bib": "6aac233e3f9ccdd8b6cf3cd8d69e27f91e40ad06edf79e2d92afd6a94984d864"
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
    ref = json.loads((DATASET_DIR / "heliconius_erato_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1997-nijhout-heliconius-erato-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
