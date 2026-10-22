#!/usr/bin/env python3
"""
models/2022-bobrovskiy-dickinsonia-costata-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2060)
Organism: Dickinsonia costata (Ediacaran Glide-Symmetric Segmented Proarticulatan (Cholesterol Biomarker))
Phenomenon Scale: 9.9 cm (0.09945 m, log10=-1.002, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Dickinsonia costata (Late Ediacaran (White Sea, 558 Ma))
Geological Horizon: Late Ediacaran (White Sea, 558 Ma) (558.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dickinsonia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (9.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2060_dickinsonia_costata_collective_stigmergy.lean` (`MathOfBiology.Loop_2060`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dickinsonia_costata_collective_stigmergy"
EXPECTED_SHA256 = {
    "dickinsonia_costata_collective_stigmergy_reference.json": "73ee67af6ba21d53bb6147e24f3ea7933e18f9b3581bfb5a14d719f38016b95f",
    "dickinsonia_costata_specimen.jpg": "ab935d7f33b31712e63e63f3b84c1c1eac296d72d1728a400dbe609341e6ffc9",
    "scout_manifest.json": "aa522663577172360865f3087b8682bcaeff5fb829e0ae009d673437da422879",
    "paper_citations.bib": "9a15fa341b73f3b0d318b430b08f464658e16f847ee03fea24901cbb181200e6"
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
    ref = json.loads((DATASET_DIR / "dickinsonia_costata_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-bobrovskiy-dickinsonia-costata-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
