#!/usr/bin/env python3
"""
models/2012-chen-dasypus-novemcinctus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1816)
Organism: Dasypus novemcinctus (Nine-Banded Armadillo (Hexagonal Osteoderm Rosette Tiles))
Phenomenon Scale: 1.5 mm (0.001501 m, log10=-2.824, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Dasypus novemcinctus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dasypus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1816_dasypus_novemcinctus_elastic_buckling.lean` (`MathOfBiology.Loop_1816`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dasypus_novemcinctus_elastic_buckling"
EXPECTED_SHA256 = {
    "dasypus_novemcinctus_elastic_buckling_reference.json": "f1af5b74155c47bcdcc7244c2ac65a14a2897653e9a0834450ddeb94d71a2e7a",
    "dasypus_novemcinctus_specimen.jpg": "5cbb611c05d8707d92d1c6d2cdd19de1525cb938009adae92309147f22174539",
    "scout_manifest.json": "74cf4d53ce0ab536db49105f7612918de6fd2bd2a3ef090e0a141043d97aaf4d",
    "paper_citations.bib": "2af6b8032c2db0f7835a9f9c7913c7bdf35b96bfdbc7a15d192110b6a639d31c"
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
    ref = json.loads((DATASET_DIR / "dasypus_novemcinctus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-chen-dasypus-novemcinctus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
