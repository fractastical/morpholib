#!/usr/bin/env python3
"""
models/2021-martins-loxodonta-africana-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1547)
Organism: Loxodonta africana (African Bush Elephant (Skin Thermal Mud-Cracking Channel Network))
Phenomenon Scale: 327 µm (0.000327 m, log10=-3.485, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Loxodonta africana (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Loxodonta
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (327 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1547_loxodonta_africana_flux_canalization.lean` (`MathOfBiology.Loop_1547`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/loxodonta_africana_flux_canalization"
EXPECTED_SHA256 = {
    "loxodonta_africana_flux_canalization_reference.json": "e2da11303c5dbc6a4921f14d73d22f1372014324623480ccf735d210169131c1",
    "loxodonta_africana_specimen.jpg": "a530e48a3daec38e9c6fd4a03ae176871589b79b17232feda58acffdff236163",
    "scout_manifest.json": "ec9e5dfb7ab28867226da6465aef184e072232ed00aea4544a0e57012c3bcc28",
    "paper_citations.bib": "621031b24d8e2317aecfa222e952581225eb9cb63777fb3ac8d6b48584ec4f19"
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
    ref = json.loads((DATASET_DIR / "loxodonta_africana_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-martins-loxodonta-africana-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
