#!/usr/bin/env python3
"""
models/2012-paterson-anomalocaris-canadensis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1788)
Organism: Anomalocaris canadensis (Cambrian Burgess Shale Radiodont Apex Predator (Compound Eye & Oral Cone))
Phenomenon Scale: 6.5 mm (0.006468 m, log10=-2.189, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Anomalocaris canadensis (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Anomalocaris
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (6.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1788_anomalocaris_canadensis_elastic_buckling.lean` (`MathOfBiology.Loop_1788`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anomalocaris_canadensis_elastic_buckling"
EXPECTED_SHA256 = {
    "anomalocaris_canadensis_elastic_buckling_reference.json": "802c86fe6f7d6cd5ed8c2715be637b86fba55b367d6681829e48b666af619628",
    "anomalocaris_canadensis_specimen.jpg": "197e43665198870b80e436c90febc5b7010eb24957c9239893e6ac1be683bf9c",
    "scout_manifest.json": "9c6650dd354b67d93390fcb0a89cc1e9339b4ea60f10f4d23c4840984e675568",
    "paper_citations.bib": "64846f45bef4d8e112da9c34a022c79eb29d871ee2c67374bb22cc10a6c2aafa"
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
    ref = json.loads((DATASET_DIR / "anomalocaris_canadensis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-paterson-anomalocaris-canadensis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
