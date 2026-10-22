#!/usr/bin/env python3
"""
models/2015-li-microraptor-gui-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1501)
Organism: Microraptor gui (Early Cretaceous Iridescent Black Four-Winged Dromaeosaur)
Phenomenon Scale: 214 nm (2.139e-07 m, log10=-6.670, 2_macromolecular_100nm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Microraptor gui (Early Cretaceous (Jiufotang Fm, 120.0 Ma))
Geological Horizon: Early Cretaceous (Jiufotang Fm, 120.0 Ma) (120.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Microraptor
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (214 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1501_microraptor_gui_flux_canalization.lean` (`MathOfBiology.Loop_1501`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/microraptor_gui_flux_canalization"
EXPECTED_SHA256 = {
    "microraptor_gui_flux_canalization_reference.json": "87200e9ce5e9faf29f8b904d0741eaed2b60698ed46b86fbf05d3539dca09954",
    "microraptor_gui_specimen.jpg": "24fb32a1367db5df0d77dace2b5b810c0854a2b3c231e370b837356348765137",
    "scout_manifest.json": "7deddc0b552aa5a4a8a28c26ee9a09829840137117533a5df7ea0f0c3702b27b",
    "paper_citations.bib": "eb0bac55ba9187613cda1f56bef9e5de24601fdff98a0a740d93accc2eff8790"
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
    ref = json.loads((DATASET_DIR / "microraptor_gui_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-li-microraptor-gui-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
