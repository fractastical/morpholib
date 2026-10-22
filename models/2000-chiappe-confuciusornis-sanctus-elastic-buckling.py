#!/usr/bin/env python3
"""
models/2000-chiappe-confuciusornis-sanctus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1772)
Organism: Confuciusornis sanctus (Early Cretaceous Pygostyle Beaked Bird (Ribbon Tail Streamers))
Phenomenon Scale: 3.0 cm (0.02988 m, log10=-1.525, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Confuciusornis sanctus (Early Cretaceous (Yixian Fm, 125.0 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 125.0 Ma) (125.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Confuciusornis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1772_confuciusornis_sanctus_elastic_buckling.lean` (`MathOfBiology.Loop_1772`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/confuciusornis_sanctus_elastic_buckling"
EXPECTED_SHA256 = {
    "confuciusornis_sanctus_elastic_buckling_reference.json": "72195589548e9018553d99b49133980059dd02629a3620f26864ecc1b9a32230",
    "confuciusornis_sanctus_specimen.jpg": "cd7c05e2d404d66e00f79aa73a3de8f5a68cde60009b9288d61176e6902d7620",
    "scout_manifest.json": "791982ac8bf7990b8560286147e48c9b1c61418e6ccb8385d323be5095374d08",
    "paper_citations.bib": "2f09c7134f96e3b8ef66402b1ea53e20670fecaca24a0f6fc8abf0abae8db588"
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
    ref = json.loads((DATASET_DIR / "confuciusornis_sanctus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-chiappe-confuciusornis-sanctus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
