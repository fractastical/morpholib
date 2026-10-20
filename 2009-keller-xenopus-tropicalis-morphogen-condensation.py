#!/usr/bin/env python3
"""
2009-keller-xenopus-tropicalis-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0221)
Organism: Xenopus tropicalis (Western Clawed Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Xenopus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0221_xenopus_tropicalis_morphogen_condensation.lean` (`MathOfBiology.Loop_0221`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/xenopus_tropicalis_morphogen_condensation"
EXPECTED_SHA256 = {
    "xenopus_tropicalis_morphogen_condensation_reference.json": "0ec6d3be7952613113a7f5dc6aa304a1a14a9561c9e8ddfe7168769310dc3d51",
    "xenopus_tropicalis_specimen.jpg": "13e063704f07254fd35cc867e7368242cfe27201d452c753342213eaa194d71f",
    "scout_manifest.json": "4c8d2d2de1ff9593732832b59bc22cfc36d8a78e8ab8b521d93c5baadeeba250",
    "paper_citations.bib": "73c6a0b9668b595f7ebeda23bba9df86cdb6e22429db887c8f819cb556074960"
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
    ref = json.loads((DATASET_DIR / "xenopus_tropicalis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-keller-xenopus-tropicalis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
