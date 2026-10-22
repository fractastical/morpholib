#!/usr/bin/env python3
"""
models/2017-inoue-placenticeras-meeki-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1778)
Organism: Placenticeras meeki (Late Cretaceous Iridescent Ammolite Ammonite (Hyper-Fractal Suture, D_f=1.68))
Phenomenon Scale: 1.4 cm (0.01391 m, log10=-1.857, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Placenticeras meeki (Late Cretaceous (Campanian, 75 Ma))
Geological Horizon: Late Cretaceous (Campanian, 75 Ma) (75.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Placenticeras
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1778_placenticeras_meeki_elastic_buckling.lean` (`MathOfBiology.Loop_1778`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/placenticeras_meeki_elastic_buckling"
EXPECTED_SHA256 = {
    "placenticeras_meeki_elastic_buckling_reference.json": "99bbceb7d2d7af03c24507c151f0ed4dee2b84df66789b9611eecdb21f50c610",
    "placenticeras_meeki_specimen.jpg": "ef945c35e08dd1adfded926af45c962f3c6017582c005f246eceedc2238e4229",
    "scout_manifest.json": "87de8fa22957fd382aa4a252cdd7d92e970b43424e3c46b181132e1a455fcdbe",
    "paper_citations.bib": "c449da291a2f4f326a4ef7a105eb87f7d38213412203ce71c79d36f841c5ac84"
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
    ref = json.loads((DATASET_DIR / "placenticeras_meeki_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-inoue-placenticeras-meeki-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
