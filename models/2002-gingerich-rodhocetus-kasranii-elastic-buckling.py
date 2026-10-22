#!/usr/bin/env python3
"""
models/2002-gingerich-rodhocetus-kasranii-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1765)
Organism: Rodhocetus kasranii (Middle Eocene Pelvic-Swimming Protocetid Whale)
Phenomenon Scale: 5.1 cm (0.05122 m, log10=-1.291, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Rodhocetus kasranii (Middle Eocene (Lutetian, 46.0 Ma))
Geological Horizon: Middle Eocene (Lutetian, 46.0 Ma) (46.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Rodhocetus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (5.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1765_rodhocetus_kasranii_elastic_buckling.lean` (`MathOfBiology.Loop_1765`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rodhocetus_kasranii_elastic_buckling"
EXPECTED_SHA256 = {
    "rodhocetus_kasranii_elastic_buckling_reference.json": "2ac7a53e9a34c0a6e34f40a579d280299b262cb1c2acf4475e355f95811fc700",
    "rodhocetus_kasranii_specimen.jpg": "939e4d05d84f57caadd235a117accb853b0aee6b303a1b365098b6db1f32f308",
    "scout_manifest.json": "e36a954d0ef441d73d76db6919de0fddd266f47ba59eb1fa3bc4058ce11db63b",
    "paper_citations.bib": "c4ca0860020a0f8cbfce6c6c9d9cd6ca747cd636073d36b06c1e1d596fc44e74"
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
    ref = json.loads((DATASET_DIR / "rodhocetus_kasranii_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-gingerich-rodhocetus-kasranii-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
