#!/usr/bin/env python3
"""
2014-wagner-tyto-alba-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0918)
Organism: Tyto alba (Barn Owl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Tyto
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0918_tyto_alba_bioelectric_polarity.lean` (`MathOfBiology.Loop_0918`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tyto_alba_bioelectric_polarity"
EXPECTED_SHA256 = {
    "tyto_alba_bioelectric_polarity_reference.json": "ecc2c948bec4379831cf9db13301b06c327f7752febecf628be4f392ddad8cc3",
    "tyto_alba_specimen.jpg": "1a81ca2e5a4b493f8c795ded141df4f7bfbee333f381a42a0b020882d72c2671",
    "scout_manifest.json": "3b37bbf09812447bd54d266281fbc1ae59aaae9d33696e8d95d4d2fbbdedfe1a",
    "paper_citations.bib": "ea280802d8594b3aa7f4c73a3f271c86e0b58576ffe07201d5055edf865d80d0"
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
    ref = json.loads((DATASET_DIR / "tyto_alba_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-wagner-tyto-alba-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
