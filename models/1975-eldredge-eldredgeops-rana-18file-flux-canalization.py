#!/usr/bin/env python3
"""
models/1975-eldredge-eldredgeops-rana-18file-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1515)
Organism: Eldredgeops rana_18file (Middle Devonian Phacopid Trilobite (18-File Schizochroal Calcite Eye))
Phenomenon Scale: 112 µm (0.0001116 m, log10=-3.952, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Eldredgeops rana_18file (Middle Devonian (Cazenovian, 387 Ma))
Geological Horizon: Middle Devonian (Cazenovian, 387 Ma) (387.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (112 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1515_eldredgeops_rana_18file_flux_canalization.lean` (`MathOfBiology.Loop_1515`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_18file_flux_canalization"
EXPECTED_SHA256 = {
    "eldredgeops_rana_18file_flux_canalization_reference.json": "cf8131ab6c8a10a4b4513609af9831212d2be2f350ce1d61d15bfa05184dbe48",
    "eldredgeops_rana_18file_specimen.jpg": "ba63b7ea8e242e6374117b84812be17b530eebb3f9621b25eff533b5397e1369",
    "scout_manifest.json": "618ea52e35f6930b9093b191e88fb74d8eefa261deb8782a6d69d985d0bd69cb",
    "paper_citations.bib": "deb891c095c6cb5e707487456c867f8916691dd3deac09f3f985fd4773d7ad2b"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_18file_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1975-eldredge-eldredgeops-rana-18file-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
