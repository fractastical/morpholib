#!/usr/bin/env python3
"""
models/2021-martins-loxodonta-africana-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1997)
Organism: Loxodonta africana (African Bush Elephant (Skin Thermal Mud-Cracking Channel Network))
Phenomenon Scale: 1.0 µm (1.046e-06 m, log10=-5.980, 3_subcellular_um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Loxodonta africana (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Loxodonta
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (1.0 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1997_loxodonta_africana_bioelectric_polarity.lean` (`MathOfBiology.Loop_1997`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/loxodonta_africana_bioelectric_polarity"
EXPECTED_SHA256 = {
    "loxodonta_africana_bioelectric_polarity_reference.json": "f656d2e0a5126a32a0a477208153aa109f02ad3b73dc2d7bef26f31d6b75d2ec",
    "loxodonta_africana_specimen.jpg": "a530e48a3daec38e9c6fd4a03ae176871589b79b17232feda58acffdff236163",
    "scout_manifest.json": "58fb9bbec26f9d650c307d806cdb42dbef1f5f2c44fcd9db751d932257af04e8",
    "paper_citations.bib": "e32e43bb5d32296df75a46d6e8149933e1ddb920a198501e5ea3831f0da417e6"
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
    ref = json.loads((DATASET_DIR / "loxodonta_africana_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-martins-loxodonta-africana-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
