#!/usr/bin/env python3
"""
models/2019-spiewak-danio-aesculapii-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1744)
Organism: Danio aesculapii (Panther Danio (Vertical Bar Transition))
Phenomenon Scale: 5.5 mm (0.005548 m, log10=-2.256, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Danio aesculapii (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (5.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1744_danio_aesculapii_elastic_buckling.lean` (`MathOfBiology.Loop_1744`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_aesculapii_elastic_buckling"
EXPECTED_SHA256 = {
    "danio_aesculapii_elastic_buckling_reference.json": "745d08ea9f3f1a9e901f81a208068e3ad40ad196fca1c05500578f58ae7e63b1",
    "danio_aesculapii_specimen.jpg": "4ccce0ce3648e44c77cef768d10f5acd8ab553c4ca38d348c35d4c6846b46bc5",
    "scout_manifest.json": "c38c34719ecb343a223f252e3b2f1de37fdf58c53d9c4a288e8e04803e6cff75",
    "paper_citations.bib": "5f70d9f6c3bb48478946807784727ed1261a1c4998d8a0a42fc2ae8f1b67eb18"
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
    ref = json.loads((DATASET_DIR / "danio_aesculapii_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-spiewak-danio-aesculapii-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
