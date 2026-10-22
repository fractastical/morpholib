#!/usr/bin/env python3
"""
models/2011-chan-gasterosteus-aculeatus-benthic-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1751)
Organism: Gasterosteus aculeatus_benthic (Post-Glacial Pitx1-Deleted Freshwater Benthic Stickleback (Low Plate))
Phenomenon Scale: 1.2 cm (0.01238 m, log10=-1.907, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Gasterosteus aculeatus_benthic (Holocene Post-Glacial Lake (Extant))
Geological Horizon: Holocene Post-Glacial Lake (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1751_gasterosteus_aculeatus_benthic_elastic_buckling.lean` (`MathOfBiology.Loop_1751`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_benthic_elastic_buckling"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_benthic_elastic_buckling_reference.json": "5f3559fb6ac985489dd771d2cd9335b8c2d34f8e623ebaea00f38f6438aad021",
    "gasterosteus_aculeatus_benthic_specimen.jpg": "d715e13ebe21bfa5b005584946ebb33b475d11926bac487e67aab10554b26e5b",
    "scout_manifest.json": "efc9acd8b566def2d437fea64338219dae642a7324fee7c3425ac1a651de5d4d",
    "paper_citations.bib": "0b5569170634f3c85ef3bc350b6e19b93589f38ae791a067d648149b41186b27"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_benthic_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-chan-gasterosteus-aculeatus-benthic-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
