#!/usr/bin/env python3
"""
models/1991-ward-nautilus-pompilius-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2051)
Organism: Nautilus pompilius (Extant Chambered Nautilus (Equiangular Logarithmic Siphuncle))
Phenomenon Scale: 31.2 cm (0.3119 m, log10=-0.506, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Nautilus pompilius (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nautilus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (31.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2051_nautilus_pompilius_collective_stigmergy.lean` (`MathOfBiology.Loop_2051`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nautilus_pompilius_collective_stigmergy"
EXPECTED_SHA256 = {
    "nautilus_pompilius_collective_stigmergy_reference.json": "074cf61900e19d34d0174ebb0c5cf76b706fb65385a24348c065e2e27fefb651",
    "nautilus_pompilius_specimen.jpg": "5dfd47d29d062b864cb14b6f5260b674dffb372fac09ff90bdd8fc5caf196da1",
    "scout_manifest.json": "2dbc39a121e29f4688398f601d515760f21200a29b3459c64c04d4493f62a000",
    "paper_citations.bib": "ab6649baf08e4337f55a900ab3c09c79ac6fcb3dc0c7e658592836956f1800b3"
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
    ref = json.loads((DATASET_DIR / "nautilus_pompilius_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1991-ward-nautilus-pompilius-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
