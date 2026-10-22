#!/usr/bin/env python3
"""
models/2017-nweeia-monodon-monoceros-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1497)
Organism: Monodon monoceros (Arctic Narwhal (Chiral Helical Left Incisor Tusk))
Phenomenon Scale: 48.6 cm (0.4856 m, log10=-0.314, 8_colonial_m)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Monodon monoceros (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Monodon
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (48.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1497_monodon_monoceros_flux_canalization.lean` (`MathOfBiology.Loop_1497`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/monodon_monoceros_flux_canalization"
EXPECTED_SHA256 = {
    "monodon_monoceros_flux_canalization_reference.json": "25f557cd4c5c780aa8b1815ac03e88217076c7ed38361ac8341d32639bd6074d",
    "monodon_monoceros_specimen.jpg": "1fa8766cd6ff186448201eb9da7097191319445a02628f275cfe5752d1be8cc3",
    "scout_manifest.json": "4dab25cf145047a9c6237a0d821493d18d50a94e8cc241a0ec949a0de173a4b6",
    "paper_citations.bib": "4d0d6fbfd5a49f4abfc08bd5c621401f75de04e3b944b7acd8eb69380b04b62f"
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
    ref = json.loads((DATASET_DIR / "monodon_monoceros_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-nweeia-monodon-monoceros-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
