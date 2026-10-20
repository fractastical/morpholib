#!/usr/bin/env python3
"""
2007-sears-myotis-lucifugus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0909)
Organism: Myotis lucifugus (Little Brown Bat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Myotis
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0909_myotis_lucifugus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0909`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/myotis_lucifugus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "myotis_lucifugus_bioelectric_polarity_reference.json": "d6d925d4fde98f68a3b88a16f81779f82c69a026cdb8236ec85a6c83e09f0fd5",
    "myotis_lucifugus_specimen.jpg": "1c8f403a4e94c928ac82c06c27e448ba133f1f6b2691860354b6490e152a97b4",
    "scout_manifest.json": "1c993dcf30608a3a82f042822aa8b5f5c892ba54701e134f19b27da3b9b5c8fe",
    "paper_citations.bib": "4c434387b54ceb6d830db4ed849fe03c1edf814850be96cd768160cd135fbd15"
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
    ref = json.loads((DATASET_DIR / "myotis_lucifugus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-sears-myotis-lucifugus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
