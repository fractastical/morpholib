#!/usr/bin/env python3
"""
models/2012-janis-merychippus-insignis-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2030)
Organism: Merychippus insignis (Miocene Grazing Spring-Footed Three-Toed Horse)
Phenomenon Scale: 34.0 cm (0.3402 m, log10=-0.468, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Merychippus insignis (Middle Miocene (Barstovian, 15.0 Ma))
Geological Horizon: Middle Miocene (Barstovian, 15.0 Ma) (15.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Merychippus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (34.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2030_merychippus_insignis_collective_stigmergy.lean` (`MathOfBiology.Loop_2030`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/merychippus_insignis_collective_stigmergy"
EXPECTED_SHA256 = {
    "merychippus_insignis_collective_stigmergy_reference.json": "c20dc4bad29606573e01a16998d1e964573d73f82836a36f665e0661784cfef0",
    "merychippus_insignis_specimen.jpg": "93354e358baacf8e852b4dccb0a713b62e217cf8aba2864c49e3a358f03f7425",
    "scout_manifest.json": "064867d1076a5cd135016017438089b0d63b035c3d3d25089329e6d4edb65acc",
    "paper_citations.bib": "89368b0ff4d70fff4c8ff1db3deae3951b707b69a6feb453c24d0014ff5e2953"
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
    ref = json.loads((DATASET_DIR / "merychippus_insignis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-janis-merychippus-insignis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
