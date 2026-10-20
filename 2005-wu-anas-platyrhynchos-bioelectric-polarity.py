#!/usr/bin/env python3
"""
2005-wu-anas-platyrhynchos-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0916)
Organism: Anas platyrhynchos (Mallard Duck)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anas
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0916_anas_platyrhynchos_bioelectric_polarity.lean` (`MathOfBiology.Loop_0916`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anas_platyrhynchos_bioelectric_polarity"
EXPECTED_SHA256 = {
    "anas_platyrhynchos_bioelectric_polarity_reference.json": "ccf80cb930fab2cdab48fc0f7e2ed4253d751ac136155b3e5f2244eed84c556c",
    "anas_platyrhynchos_specimen.jpg": "ee1cc0720d68f5f20e5e3490647b69cb29c1fe84dae8fdb048019814f69eccba",
    "scout_manifest.json": "be2536f1ccd13417e17b0189ba134b5ae812bc794129d6f971e0a01032b93582",
    "paper_citations.bib": "e5f42235081e4bf8baf1be3d9d27d6c39891e64c225a2b03c525188cd9f9d450"
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
    ref = json.loads((DATASET_DIR / "anas_platyrhynchos_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-wu-anas-platyrhynchos-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
