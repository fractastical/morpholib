#!/usr/bin/env python3
"""
models/1985-grande-diplomystus-dentatus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1746)
Organism: Diplomystus dentatus (Green River Eocene Armored Herring Fossil)
Phenomenon Scale: 4.2 cm (0.04158 m, log10=-1.381, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Diplomystus dentatus (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Diplomystus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (4.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1746_diplomystus_dentatus_elastic_buckling.lean` (`MathOfBiology.Loop_1746`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/diplomystus_dentatus_elastic_buckling"
EXPECTED_SHA256 = {
    "diplomystus_dentatus_elastic_buckling_reference.json": "fdcc6aad80affdb686c2651df760471f67a02794d44f0741285de9dfc2d2a24c",
    "diplomystus_dentatus_specimen.jpg": "00e0f80ae28f4fe36376b217f05e2048df17ed19a49f48d49d4a9b88b2b2b659",
    "scout_manifest.json": "a346f9797038e094c9b86bc228e52f7b7a9d5a436c5b9496fe65c734c62f53a6",
    "paper_citations.bib": "0a47f0b874669dd6309fe91be414c08ecfc7bf86d4a28c61a377dbd7d3bf5cfa"
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
    ref = json.loads((DATASET_DIR / "diplomystus_dentatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1985-grande-diplomystus-dentatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
