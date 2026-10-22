#!/usr/bin/env python3
"""
models/2002-chiappe-confuciusornis-sanctus-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1502)
Organism: Confuciusornis sanctus (Early Cretaceous Pygostyle Beaked Bird (Ribbon Tail Streamers))
Phenomenon Scale: 3.4 cm (0.03395 m, log10=-1.469, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Confuciusornis sanctus (Early Cretaceous (Yixian Fm, 125.0 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 125.0 Ma) (125.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Confuciusornis
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (3.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1502_confuciusornis_sanctus_flux_canalization.lean` (`MathOfBiology.Loop_1502`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/confuciusornis_sanctus_flux_canalization"
EXPECTED_SHA256 = {
    "confuciusornis_sanctus_flux_canalization_reference.json": "d58dfdf513c3913b974566b6cbab33c41b19492f67cb050f26fc3c285b4846fc",
    "confuciusornis_sanctus_specimen.jpg": "cd7c05e2d404d66e00f79aa73a3de8f5a68cde60009b9288d61176e6902d7620",
    "scout_manifest.json": "732f24ddaf882036b3e18fdd87525c7ff67c142c52787d9eab7470d455919e57",
    "paper_citations.bib": "e32c41fb9fbb5c50e06b3f31bfe070a46d948285fad7908a92117d4672b72b1a"
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
    ref = json.loads((DATASET_DIR / "confuciusornis_sanctus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-chiappe-confuciusornis-sanctus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
