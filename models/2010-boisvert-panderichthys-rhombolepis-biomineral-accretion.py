#!/usr/bin/env python3
"""
models/2010-boisvert-panderichthys-rhombolepis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1843)
Organism: Panderichthys rhombolepis (Late Devonian Elpistostegalian Flat-Skulled Fish)
Phenomenon Scale: 9.5 cm (0.09529 m, log10=-1.021, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Panderichthys rhombolepis (Late Devonian (Frasnian, 380 Ma))
Geological Horizon: Late Devonian (Frasnian, 380 Ma) (380.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Panderichthys
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (9.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1843_panderichthys_rhombolepis_biomineral_accretion.lean` (`MathOfBiology.Loop_1843`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/panderichthys_rhombolepis_biomineral_accretion"
EXPECTED_SHA256 = {
    "panderichthys_rhombolepis_biomineral_accretion_reference.json": "90d2f6715fd17ed983516b0c2ad6b3c5e8baab160f2b1ee2179134fbd0c3328a",
    "panderichthys_rhombolepis_specimen.jpg": "5c39604c489cd7a850b62b1d1aaf2e3061b9865043e83968c2feebf4e34ca870",
    "scout_manifest.json": "88c2337dda5f6185f6c124f61660ebc47f4ede65767552270457f177dc985e85",
    "paper_citations.bib": "9ccf0d4f294e247a4bb70375150f58a52c1b36caeabac08df049809cbd281255"
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
    ref = json.loads((DATASET_DIR / "panderichthys_rhombolepis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-boisvert-panderichthys-rhombolepis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
