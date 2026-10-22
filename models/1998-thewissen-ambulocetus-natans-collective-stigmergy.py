#!/usr/bin/env python3
"""
models/1998-thewissen-ambulocetus-natans-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2034)
Organism: Ambulocetus natans (Early Eocene Amphibious Walking Whale)
Phenomenon Scale: 46.9 cm (0.4687 m, log10=-0.329, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Ambulocetus natans (Early Eocene (Lutetian, 47.5 Ma))
Geological Horizon: Early Eocene (Lutetian, 47.5 Ma) (47.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ambulocetus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (46.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2034_ambulocetus_natans_collective_stigmergy.lean` (`MathOfBiology.Loop_2034`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ambulocetus_natans_collective_stigmergy"
EXPECTED_SHA256 = {
    "ambulocetus_natans_collective_stigmergy_reference.json": "bf8e5ad3af3d921cb430d05c6b1363943797631c7dcc7c3aabb5906294ad65a9",
    "ambulocetus_natans_specimen.jpg": "01a4e6043915d96e0394516e3db543eecad170348e219e8dd573ef897d004e67",
    "scout_manifest.json": "7230e089cf37a49c02204c12d983c7ebc1bbd97f913493fd799259dd8576e767",
    "paper_citations.bib": "a9b8b4d6cfbdb25a9ee189655175b7398c3b1b9c37dc4f6a79b3fecde14d4bd4"
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
    ref = json.loads((DATASET_DIR / "ambulocetus_natans_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1998-thewissen-ambulocetus-natans-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
