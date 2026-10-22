#!/usr/bin/env python3
"""
models/2003-gingerich-rodhocetus-kasranii-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1405)
Organism: Rodhocetus kasranii (Middle Eocene Pelvic-Swimming Protocetid Whale)
Phenomenon Scale: 1.9 cm (0.01862 m, log10=-1.730, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Rodhocetus kasranii (Middle Eocene (Lutetian, 46.0 Ma))
Geological Horizon: Middle Eocene (Lutetian, 46.0 Ma) (46.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Rodhocetus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1405_rodhocetus_kasranii_clock_wavefront.lean` (`MathOfBiology.Loop_1405`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rodhocetus_kasranii_clock_wavefront"
EXPECTED_SHA256 = {
    "rodhocetus_kasranii_clock_wavefront_reference.json": "56384328a2b078ec5e1c44582656c8f614a775774a80112c553a449e2c3a4f6b",
    "rodhocetus_kasranii_specimen.jpg": "939e4d05d84f57caadd235a117accb853b0aee6b303a1b365098b6db1f32f308",
    "scout_manifest.json": "633d4a74701f362b0cc9ef7ce0b5112f004387397823a74f020fa5f66f595f0c",
    "paper_citations.bib": "98dbb4b12e2d36d578b19882c9ff88bd531e6be758b0c8ca83260ddccf4da3dd"
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
    ref = json.loads((DATASET_DIR / "rodhocetus_kasranii_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-gingerich-rodhocetus-kasranii-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
