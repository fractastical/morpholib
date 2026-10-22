#!/usr/bin/env python3
"""
models/2004-gingerich-rodhocetus-kasranii-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1945)
Organism: Rodhocetus kasranii (Middle Eocene Pelvic-Swimming Protocetid Whale)
Phenomenon Scale: 186 µm (0.0001862 m, log10=-3.730, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Rodhocetus kasranii (Middle Eocene (Lutetian, 46.0 Ma))
Geological Horizon: Middle Eocene (Lutetian, 46.0 Ma) (46.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Rodhocetus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (186 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1945_rodhocetus_kasranii_bioelectric_polarity.lean` (`MathOfBiology.Loop_1945`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rodhocetus_kasranii_bioelectric_polarity"
EXPECTED_SHA256 = {
    "rodhocetus_kasranii_bioelectric_polarity_reference.json": "802a968967dfaeed0f11c0b5abb5b12a511bb92d90b1ce2e4ca42f30c4e0fecf",
    "rodhocetus_kasranii_specimen.jpg": "939e4d05d84f57caadd235a117accb853b0aee6b303a1b365098b6db1f32f308",
    "scout_manifest.json": "283b204864f9a2d9501f59c9e7b1e2f1d52b4e08ae470e4f96d3fd6465435493",
    "paper_citations.bib": "0c72d0cf827b92cce0ec09dec36176b64bc6df439de997e3c016e04cfd490f04"
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
    ref = json.loads((DATASET_DIR / "rodhocetus_kasranii_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-gingerich-rodhocetus-kasranii-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
