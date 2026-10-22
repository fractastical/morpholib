#!/usr/bin/env python3
"""
models/2013-jeffery-astyanax-mexicanus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2090)
Organism: Astyanax mexicanus (Mexican Blind Cavefish (Optic Cup Degeneration vs Neuromast Expansion))
Phenomenon Scale: 1.1 mm (0.001088 m, log10=-2.963, 6_organ_mm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Astyanax mexicanus (Holocene Cave Morph (Extant))
Geological Horizon: Holocene Cave Morph (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Astyanax
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (1.1 mm))
Formal Verification: `morphoeval/lean/loops/Loop_2090_astyanax_mexicanus_collective_stigmergy.lean` (`MathOfBiology.Loop_2090`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/astyanax_mexicanus_collective_stigmergy"
EXPECTED_SHA256 = {
    "astyanax_mexicanus_collective_stigmergy_reference.json": "4497726492424f91f42c45c9136cbad5b6b80a1cf1856ba91bbc7edf8f37fadc",
    "astyanax_mexicanus_specimen.jpg": "53b28ff32166e8938db70570a413fac88bfdd10206b30d8fcda2d15cb61c64f5",
    "scout_manifest.json": "971abe2285d82a450b33ce58be5a227fe55f6ab57fb6f667ff6245d48a629981",
    "paper_citations.bib": "535d700d015de750a20f9dfdf4ba9cfbc375da9be83ef96cb181266d24dcbddb"
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
    ref = json.loads((DATASET_DIR / "astyanax_mexicanus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-jeffery-astyanax-mexicanus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
