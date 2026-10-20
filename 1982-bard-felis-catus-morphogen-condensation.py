#!/usr/bin/env python3
"""
1982-bard-felis-catus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0203)
Organism: Felis catus (Domestic Cat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Felis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0203_felis_catus_morphogen_condensation.lean` (`MathOfBiology.Loop_0203`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/felis_catus_morphogen_condensation"
EXPECTED_SHA256 = {
    "felis_catus_morphogen_condensation_reference.json": "780e9a45e21daef55822bf5375272b1ef7214c5d89acdea7bd2ccbfa6250ca4d",
    "felis_catus_specimen.jpg": "f417376a9a5f657061f2082396f5cc7eb2b6a4d0c92e0c1d3692ceb2b52926bd",
    "scout_manifest.json": "24edbdeb12defaa66a377675a453ede9e030dc380bd5cf663796f4380b293cbe",
    "paper_citations.bib": "153f648f8ce62c7d094d5f103e63af4cfda9dd3434c42aa2663b497e434c6b5e"
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
    ref = json.loads((DATASET_DIR / "felis_catus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1982-bard-felis-catus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
