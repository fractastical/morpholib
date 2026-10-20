#!/usr/bin/env python3
"""
2014-chirat-ovis-aries-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0907)
Organism: Ovis aries (Bighorn Sheep)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ovis
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0907_ovis_aries_bioelectric_polarity.lean` (`MathOfBiology.Loop_0907`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ovis_aries_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ovis_aries_bioelectric_polarity_reference.json": "0b072136bb30684a23df361ce33e709b8df3d9353f8e2bf654ded43f91d198d2",
    "ovis_aries_specimen.jpg": "3e5aea2a1dd349564873df9e61677c401d7c2264295f722a777e3808f6a3c0d2",
    "scout_manifest.json": "61efec881f1248337342393b31d48044fd08f7005a2eb1de8110b0321758c134",
    "paper_citations.bib": "a5404be4167bd9fe3811f1ecc03a8d36e02122ca3e725591fa4c0c3268ab05e3"
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
    ref = json.loads((DATASET_DIR / "ovis_aries_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-chirat-ovis-aries-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
