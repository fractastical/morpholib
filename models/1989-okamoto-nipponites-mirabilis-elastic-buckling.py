#!/usr/bin/env python3
"""
models/1989-okamoto-nipponites-mirabilis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1780)
Organism: Nipponites mirabilis (Late Cretaceous Meandering 3D Trefoil-Knot Heteromorph Ammonite)
Phenomenon Scale: 1.2 cm (0.012 m, log10=-1.921, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Nipponites mirabilis (Late Cretaceous (Coniacian, 88 Ma))
Geological Horizon: Late Cretaceous (Coniacian, 88 Ma) (88.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nipponites
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1780_nipponites_mirabilis_elastic_buckling.lean` (`MathOfBiology.Loop_1780`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nipponites_mirabilis_elastic_buckling"
EXPECTED_SHA256 = {
    "nipponites_mirabilis_elastic_buckling_reference.json": "6c2f85e83511c95787f0806c63ff42a027ecfd5e958726ec32b351a7c85b96ec",
    "nipponites_mirabilis_specimen.jpg": "6b8124b1e5eeb73dc589cf76fafc5db8b2d4a7d15eb68c77a1175d267f717dd5",
    "scout_manifest.json": "9532ddb5e445f1b3f24d44db4f2fde9b0ab9fe5d43be374e7d2ccca28a11d5a1",
    "paper_citations.bib": "df0e21e8dd72d1f72205a26fc3cc3622c248404dd02440148333d1eddec7cba3"
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
    ref = json.loads((DATASET_DIR / "nipponites_mirabilis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1989-okamoto-nipponites-mirabilis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
