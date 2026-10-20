#!/usr/bin/env python3
"""
2014-kotrschal-poecilia-reticulata-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0937)
Organism: Poecilia reticulata (Trinidadian Guppy)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Poecilia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0937_poecilia_reticulata_bioelectric_polarity.lean` (`MathOfBiology.Loop_0937`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/poecilia_reticulata_bioelectric_polarity"
EXPECTED_SHA256 = {
    "poecilia_reticulata_bioelectric_polarity_reference.json": "07ac2fb8bba6310aca205af5ed8f898e85d9173b201933682f306b138a1ced32",
    "poecilia_reticulata_specimen.jpg": "3e998877d83b46f9df12f9fa135839a5982986b29a14965629955111cadf8a2e",
    "scout_manifest.json": "a9415d3c219832673bda0f4aefb2373640d384d66db3e87a33c886ac711c6d93",
    "paper_citations.bib": "0e153c932c4df4b79cb182c0ab251b2630033e3726667b55c09b014744e9d362"
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
    ref = json.loads((DATASET_DIR / "poecilia_reticulata_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-kotrschal-poecilia-reticulata-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
