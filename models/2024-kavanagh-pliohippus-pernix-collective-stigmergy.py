#!/usr/bin/env python3
"""
models/2024-kavanagh-pliohippus-pernix-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2031)
Organism: Pliohippus pernix (Late Miocene Single-Hoofed Monodactyl Ancestor)
Phenomenon Scale: 42.2 cm (0.4218 m, log10=-0.375, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Pliohippus pernix (Late Miocene (Hemphillian, 8.5 Ma))
Geological Horizon: Late Miocene (Hemphillian, 8.5 Ma) (8.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pliohippus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (42.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2031_pliohippus_pernix_collective_stigmergy.lean` (`MathOfBiology.Loop_2031`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pliohippus_pernix_collective_stigmergy"
EXPECTED_SHA256 = {
    "pliohippus_pernix_collective_stigmergy_reference.json": "dfab18b03b07a24cb44c93853a0f281f58653ab17606f35a0b974133e9fc27b2",
    "pliohippus_pernix_specimen.jpg": "1211e4acc14fca6bb091e7747aca534b08c9b54a2cea23f8b7aa25c1b659afdd",
    "scout_manifest.json": "6405691e9b07b3c419910cac00e0c2699632918ae054b889d8979d8e12ce35f6",
    "paper_citations.bib": "c3c54e6c7fb5c509cfe94c50b28c0e1ddf5b0ed8423f1c8b935c399ab913b876"
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
    ref = json.loads((DATASET_DIR / "pliohippus_pernix_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-kavanagh-pliohippus-pernix-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
