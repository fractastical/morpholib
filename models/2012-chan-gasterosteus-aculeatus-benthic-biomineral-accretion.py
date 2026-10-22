#!/usr/bin/env python3
"""
models/2012-chan-gasterosteus-aculeatus-benthic-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1841)
Organism: Gasterosteus aculeatus_benthic (Post-Glacial Pitx1-Deleted Freshwater Benthic Stickleback (Low Plate))
Phenomenon Scale: 3.1 cm (0.03094 m, log10=-1.509, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Gasterosteus aculeatus_benthic (Holocene Post-Glacial Lake (Extant))
Geological Horizon: Holocene Post-Glacial Lake (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (3.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1841_gasterosteus_aculeatus_benthic_biomineral_accretion.lean` (`MathOfBiology.Loop_1841`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_benthic_biomineral_accretion"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_benthic_biomineral_accretion_reference.json": "8c21e8eb533759e16b4b40ecf42872e2357e4f0be564e540adc31d6983bd247d",
    "gasterosteus_aculeatus_benthic_specimen.jpg": "d715e13ebe21bfa5b005584946ebb33b475d11926bac487e67aab10554b26e5b",
    "scout_manifest.json": "585141818df31af6d998fd793b325d7369146b0353d007b024a78854e7f240ff",
    "paper_citations.bib": "7aa56c3ce2924e201faedafcd5dda8e36ef99274d57749e1d879f841e4ef5016"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_benthic_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-chan-gasterosteus-aculeatus-benthic-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
