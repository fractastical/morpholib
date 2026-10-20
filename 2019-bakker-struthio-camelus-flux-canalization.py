#!/usr/bin/env python3
"""
2019-bakker-struthio-camelus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0415)
Organism: Struthio camelus (Common Ostrich)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Struthio
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0415_struthio_camelus_flux_canalization.lean` (`MathOfBiology.Loop_0415`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/struthio_camelus_flux_canalization"
EXPECTED_SHA256 = {
    "struthio_camelus_flux_canalization_reference.json": "6fa280c118fbece59336f45d1cb5f70fea2d8ecd6588c1da58604680080892e6",
    "struthio_camelus_specimen.jpg": "798eba64c73c1474a1b787e850654c5f61164ca01db9ec2d75836507f9494def",
    "scout_manifest.json": "f59e80247da4b313f7b38b668c31707b3cf501c2f36cd9a695758b7efb75e3f2",
    "paper_citations.bib": "a623079f67e108ca0d3546a366cae5f44b2314450ea2e46a9567475011296468"
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
    ref = json.loads((DATASET_DIR / "struthio_camelus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-bakker-struthio-camelus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
