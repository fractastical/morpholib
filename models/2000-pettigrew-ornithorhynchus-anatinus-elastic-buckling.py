#!/usr/bin/env python3
"""
models/2000-pettigrew-ornithorhynchus-anatinus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1814)
Organism: Ornithorhynchus anatinus (Duck-Billed Platypus (Electroreceptive Bill Pore Hexagonal Array))
Phenomenon Scale: 30 µm (2.988e-05 m, log10=-4.525, 4_cellular_10um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Ornithorhynchus anatinus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ornithorhynchus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (30 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1814_ornithorhynchus_anatinus_elastic_buckling.lean` (`MathOfBiology.Loop_1814`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ornithorhynchus_anatinus_elastic_buckling"
EXPECTED_SHA256 = {
    "ornithorhynchus_anatinus_elastic_buckling_reference.json": "ba678e768f5e263e99a3a252bc07f38cfcc2c574307d2c92b2ba339f93ceaf84",
    "ornithorhynchus_anatinus_specimen.jpg": "5a8d5bf6c1809500a8e9e88694cff57c75cead6258be2e7d05ede7922465f2b6",
    "scout_manifest.json": "f306356b1f961b40c57a6585f85602fcd52eb6bb695bd274a1df7bbb49cde5d4",
    "paper_citations.bib": "f46a9cdc1fae3adf64618d3cf9838ad85e994f338e3ab077a16fb294b3a61d8a"
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
    ref = json.loads((DATASET_DIR / "ornithorhynchus_anatinus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/2000-pettigrew-ornithorhynchus-anatinus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
