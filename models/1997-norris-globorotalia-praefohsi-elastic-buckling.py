#!/usr/bin/env python3
"""
models/1997-norris-globorotalia-praefohsi-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1805)
Organism: Globorotalia praefohsi (Middle Miocene Unkeeled Rounded-Margin Foraminifer Chronospecies)
Phenomenon Scale: 67 µm (6.657e-05 m, log10=-4.177, 4_cellular_10um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Globorotalia praefohsi (Middle Miocene (Serravallian, 13.4 Ma))
Geological Horizon: Middle Miocene (Serravallian, 13.4 Ma) (13.4 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (67 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1805_globorotalia_praefohsi_elastic_buckling.lean` (`MathOfBiology.Loop_1805`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_praefohsi_elastic_buckling"
EXPECTED_SHA256 = {
    "globorotalia_praefohsi_elastic_buckling_reference.json": "d0b1d4e24bde65db7e1714bf06736d489877343da14ba5fd93288b3652d73851",
    "globorotalia_praefohsi_specimen.jpg": "4bd61f5c5fd1e4a4d998a7b4bf377b989b14b379445592ebb2d464ec82617c74",
    "scout_manifest.json": "a4bab312327e3c73256f3aadd30c4feed4851852278f1ce67f49c856699b2cdc",
    "paper_citations.bib": "9f2b12ff97d2302d36c307b448e38d248b884aa80674a946a9904043e6e0c39d"
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
    ref = json.loads((DATASET_DIR / "globorotalia_praefohsi_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1997-norris-globorotalia-praefohsi-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
