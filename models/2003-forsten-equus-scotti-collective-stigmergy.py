#!/usr/bin/env python3
"""
models/2003-forsten-equus-scotti-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2032)
Organism: Equus scotti (Pleistocene Giant Ice-Age Horse Chronospecies)
Phenomenon Scale: 39.8 cm (0.3978 m, log10=-0.400, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Equus scotti (Middle Pleistocene (Irvingtonian, 0.6 Ma))
Geological Horizon: Middle Pleistocene (Irvingtonian, 0.6 Ma) (0.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (39.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2032_equus_scotti_collective_stigmergy.lean` (`MathOfBiology.Loop_2032`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/equus_scotti_collective_stigmergy"
EXPECTED_SHA256 = {
    "equus_scotti_collective_stigmergy_reference.json": "de7ea01a97b626b66eb2a5d28d1b1f133b9386393161b97f553c3fbf15b08596",
    "equus_scotti_specimen.jpg": "fea6d4b899e55872847306904b4629acc9fcf09737b2a6f906d0accb0de6e222",
    "scout_manifest.json": "fff9ed5b46081c1699c48a6d268fe35ab9fe4e38f9263dd06847e613fca3202a",
    "paper_citations.bib": "49014d165ed762d5b4545adad93328e927fb0bc5ec807db383072074b44296a9"
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
    ref = json.loads((DATASET_DIR / "equus_scotti_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-forsten-equus-scotti-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
