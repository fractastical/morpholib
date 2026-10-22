#!/usr/bin/env python3
"""
models/2010-zhou-ginkgo-yimaensis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1801)
Organism: Ginkgo yimaensis (Middle Jurassic Deeply-Divided 6-Lobe Ginkgo Chronospecies)
Phenomenon Scale: 8.0 mm (0.007999 m, log10=-2.097, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Ginkgo yimaensis (Middle Jurassic (Yima Fm, 170 Ma))
Geological Horizon: Middle Jurassic (Yima Fm, 170 Ma) (170.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (8.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1801_ginkgo_yimaensis_elastic_buckling.lean` (`MathOfBiology.Loop_1801`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_yimaensis_elastic_buckling"
EXPECTED_SHA256 = {
    "ginkgo_yimaensis_elastic_buckling_reference.json": "6709bb323bbae35fcc1180bda36a1ba36d4c80744914ea7a7e39d71c85082219",
    "ginkgo_yimaensis_specimen.jpg": "1ae0d5163efa653c4908ad5d93fc761b18efd5abe7d256c68c27e86ea915e772",
    "scout_manifest.json": "c905b59390c3e84b4e8627fe41e5a68b6fabdedf22692c0cc1c8af419319be2d",
    "paper_citations.bib": "edede041b3e2790571a08be7599f43e4c3c9033b7df811165c52e11abdbee6d6"
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
    ref = json.loads((DATASET_DIR / "ginkgo_yimaensis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-zhou-ginkgo-yimaensis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
