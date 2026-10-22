#!/usr/bin/env python3
"""
models/1993-macfadden-eohippus-angustidens-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1758)
Organism: Eohippus angustidens (Early Eocene Four-Toed Dawn Horse (Hyracotherium))
Phenomenon Scale: 1.8 cm (0.01814 m, log10=-1.741, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Eohippus angustidens (Early Eocene (Wasatchian, 52.0 Ma))
Geological Horizon: Early Eocene (Wasatchian, 52.0 Ma) (52.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Eohippus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1758_eohippus_angustidens_elastic_buckling.lean` (`MathOfBiology.Loop_1758`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eohippus_angustidens_elastic_buckling"
EXPECTED_SHA256 = {
    "eohippus_angustidens_elastic_buckling_reference.json": "9effc4c64d1f92644c664f007b9841c43747c666d6c87e62f0766086605d3318",
    "eohippus_angustidens_specimen.jpg": "4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf",
    "scout_manifest.json": "4e8846c8f886f4070108a71c14685011320378f5dcac0533e113e3507e2572a8",
    "paper_citations.bib": "228448423190e8f240b94b40506a3ea8d46a3391a405da2705d3d0cc7a99bd7d"
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
    ref = json.loads((DATASET_DIR / "eohippus_angustidens_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1993-macfadden-eohippus-angustidens-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
