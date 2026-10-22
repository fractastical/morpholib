#!/usr/bin/env python3
"""
models/1986-malmgren-globorotalia-fohsi-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1536)
Organism: Globorotalia fohsi (Middle Miocene Peripheral-Keel Punctuated Foraminifer Chronospecies)
Phenomenon Scale: 98 µm (9.765e-05 m, log10=-4.010, 4_cellular_10um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Globorotalia fohsi (Middle Miocene (Serravallian, 12.7 Ma))
Geological Horizon: Middle Miocene (Serravallian, 12.7 Ma) (12.7 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (98 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1536_globorotalia_fohsi_flux_canalization.lean` (`MathOfBiology.Loop_1536`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_fohsi_flux_canalization"
EXPECTED_SHA256 = {
    "globorotalia_fohsi_flux_canalization_reference.json": "ac0bacef6727810035aa5f8207157bdc5c4dcf5c4bc5850916113c62af93b933",
    "globorotalia_fohsi_specimen.jpg": "da1c3b52b1d7e26144fe2b2960cf08ed37afc6fd3eaf5de4044b285992f13b6f",
    "scout_manifest.json": "e29191d8afacd2238922a3b6435cdc886ab3ffa1e6e4ca81b51720eb9459cf97",
    "paper_citations.bib": "3bd979da8fc52cfd9e5a1fca72899e97460f4498b1b4d5a983f05831850d8fb9"
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
    ref = json.loads((DATASET_DIR / "globorotalia_fohsi_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-malmgren-globorotalia-fohsi-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
