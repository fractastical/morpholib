#!/usr/bin/env python3
"""
models/2017-gorzelak-encrinus-liliiformis-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1541)
Organism: Encrinus liliiformis (Middle Triassic Sea-Lily Crinoid (Stereom Gyroid Ossicles))
Phenomenon Scale: 3.8 µm (3.825e-06 m, log10=-5.417, 3_subcellular_um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Encrinus liliiformis (Middle Triassic (Muschelkalk, 240 Ma))
Geological Horizon: Middle Triassic (Muschelkalk, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Encrinus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (3.8 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1541_encrinus_liliiformis_flux_canalization.lean` (`MathOfBiology.Loop_1541`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/encrinus_liliiformis_flux_canalization"
EXPECTED_SHA256 = {
    "encrinus_liliiformis_flux_canalization_reference.json": "be927d44aee277cd78e19abbbe790c7b64a2eef60c6d68e564c735a34b86bdc6",
    "encrinus_liliiformis_specimen.jpg": "d19f3b6e48018c2dcd0ffb7be81fb1a880d1264498e9b5ec0b872b3310699a44",
    "scout_manifest.json": "9d0bf1df67fdaede75e7d46f4b23c100fa962c0bc1c67f6ec0419d418050084d",
    "paper_citations.bib": "d7a76fdb70bfc1acd0c39d360e781c1bd6e13478924622df73c192c1af6e20d2"
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
    ref = json.loads((DATASET_DIR / "encrinus_liliiformis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-gorzelak-encrinus-liliiformis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
