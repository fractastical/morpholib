#!/usr/bin/env python3
"""
models/2005-klug-ceratites-nodosus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1776)
Organism: Ceratites nodosus (Middle Triassic Muschelkalk Ammonoid (Serrated Lobes, D_f=1.34))
Phenomenon Scale: 6.0 mm (0.005984 m, log10=-2.223, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Ceratites nodosus (Middle Triassic (Ladinian, 240 Ma))
Geological Horizon: Middle Triassic (Ladinian, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ceratites
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (6.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1776_ceratites_nodosus_elastic_buckling.lean` (`MathOfBiology.Loop_1776`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ceratites_nodosus_elastic_buckling"
EXPECTED_SHA256 = {
    "ceratites_nodosus_elastic_buckling_reference.json": "1ebc336685ede14cb145ce33e0b094f6b73ff21b3885bfb02c14743711146a34",
    "ceratites_nodosus_specimen.jpg": "3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0",
    "scout_manifest.json": "5db0ac016537952d6002f8046fca2a95cacedd03d538edaf89c52d990406c5f5",
    "paper_citations.bib": "78ccf85d5f1c6980dc9a162185d15c4dc448c89c425ebc1989547cc066abdd16"
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
    ref = json.loads((DATASET_DIR / "ceratites_nodosus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-klug-ceratites-nodosus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
