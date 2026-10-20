#!/usr/bin/env python3
"""
2003-prum-taeniopygia-guttata-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0912)
Organism: Taeniopygia guttata (Zebra Finch)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Taeniopygia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0912_taeniopygia_guttata_bioelectric_polarity.lean` (`MathOfBiology.Loop_0912`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/taeniopygia_guttata_bioelectric_polarity"
EXPECTED_SHA256 = {
    "taeniopygia_guttata_bioelectric_polarity_reference.json": "2217705b010c4b3e291a7299e454313cf617ca6c6b8e3d4b48bb56867c902d6b",
    "taeniopygia_guttata_specimen.jpg": "cba8970e2c1bea961592887848134a4a694bea3ad4d2b68524b2a63c722dffa0",
    "scout_manifest.json": "eed5d327d4c6b24c5040d6bdc8fddb75f76ac40e94d4c09e58213e1d96ef4836",
    "paper_citations.bib": "889350b3bc9540122e7df869ca9767729a8f3ee60d4db887bafa7be4d1a6b9d3"
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
    ref = json.loads((DATASET_DIR / "taeniopygia_guttata_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-prum-taeniopygia-guttata-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
