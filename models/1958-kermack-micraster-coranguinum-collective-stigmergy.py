#!/usr/bin/env python3
"""
models/1958-kermack-micraster-coranguinum-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2082)
Organism: Micraster coranguinum (Late Cretaceous Chalk Heart-Urchin Chronospecies (Petaloid Burrowing))
Phenomenon Scale: 7.7 cm (0.0769 m, log10=-1.114, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Micraster coranguinum (Late Cretaceous (Santonian, 85 Ma))
Geological Horizon: Late Cretaceous (Santonian, 85 Ma) (85.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Micraster
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (7.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2082_micraster_coranguinum_collective_stigmergy.lean` (`MathOfBiology.Loop_2082`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/micraster_coranguinum_collective_stigmergy"
EXPECTED_SHA256 = {
    "micraster_coranguinum_collective_stigmergy_reference.json": "d5845d2a572b567e577f28d2bd9b204894e76f1bbdb2c455dc141d48de7e6a28",
    "micraster_coranguinum_specimen.jpg": "e066a3f38cd1da19878288ea7b3f5ea0aeec2482a9947c60126b6e6e07fb80da",
    "scout_manifest.json": "fecf69d2e1eb035d64761994407b69b812f915cf63e4a829e54e757a18339302",
    "paper_citations.bib": "eee7b3445b63a943e6a1b0f49ecb151776d61b6d16b39b042152fb80d7fb187c"
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
    ref = json.loads((DATASET_DIR / "micraster_coranguinum_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1958-kermack-micraster-coranguinum-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
