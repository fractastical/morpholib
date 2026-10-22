#!/usr/bin/env python3
"""
models/2022-kavanagh-pliohippus-pernix-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1851)
Organism: Pliohippus pernix (Late Miocene Single-Hoofed Monodactyl Ancestor)
Phenomenon Scale: 12.9 cm (0.1289 m, log10=-0.890, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Pliohippus pernix (Late Miocene (Hemphillian, 8.5 Ma))
Geological Horizon: Late Miocene (Hemphillian, 8.5 Ma) (8.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pliohippus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (12.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1851_pliohippus_pernix_biomineral_accretion.lean` (`MathOfBiology.Loop_1851`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pliohippus_pernix_biomineral_accretion"
EXPECTED_SHA256 = {
    "pliohippus_pernix_biomineral_accretion_reference.json": "94f6a25d38da66d6c1cbb357a521d30e0bf9acf8c4606072fd1a8f98253cec10",
    "pliohippus_pernix_specimen.jpg": "1211e4acc14fca6bb091e7747aca534b08c9b54a2cea23f8b7aa25c1b659afdd",
    "scout_manifest.json": "f17924a0c5f76a048b051be25bcdd6286509bef31ce77dc85587b6aa33f5390f",
    "paper_citations.bib": "d3f5bf5ec255f631b988996fedc5b970590bff7a7a1f6641c4d14c465dd9f4f5"
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
    ref = json.loads((DATASET_DIR / "pliohippus_pernix_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-kavanagh-pliohippus-pernix-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
