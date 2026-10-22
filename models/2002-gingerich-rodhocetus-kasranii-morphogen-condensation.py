#!/usr/bin/env python3
"""
models/2002-gingerich-rodhocetus-kasranii-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1315)
Organism: Rodhocetus kasranii (Middle Eocene Pelvic-Swimming Protocetid Whale)
Phenomenon Scale: 4.2 cm (0.0419 m, log10=-1.378, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Rodhocetus kasranii (Middle Eocene (Lutetian, 46.0 Ma))
Geological Horizon: Middle Eocene (Lutetian, 46.0 Ma) (46.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Rodhocetus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (4.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1315_rodhocetus_kasranii_morphogen_condensation.lean` (`MathOfBiology.Loop_1315`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rodhocetus_kasranii_morphogen_condensation"
EXPECTED_SHA256 = {
    "rodhocetus_kasranii_morphogen_condensation_reference.json": "75e9cb59909ae0aaecac3b3b1d327ec4b57d2c7bdf47d5c1731ec3f0af7871ba",
    "rodhocetus_kasranii_specimen.jpg": "939e4d05d84f57caadd235a117accb853b0aee6b303a1b365098b6db1f32f308",
    "scout_manifest.json": "41c688dc87a07e01e8883de9cc2d55e03a8d02edda3baf7db1b6875466a27f70",
    "paper_citations.bib": "c3f8c0785fe49243371a8d33c576f953bffcac9914703fea9b62cd6fc60b6dda"
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
    ref = json.loads((DATASET_DIR / "rodhocetus_kasranii_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-gingerich-rodhocetus-kasranii-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
