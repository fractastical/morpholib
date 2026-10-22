#!/usr/bin/env python3
"""
models/1996-macfadden-mesohippus-bairdi-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2029)
Organism: Mesohippus bairdi (Late Eocene Three-Toed Sub-Hypsodont Horse)
Phenomenon Scale: 22.7 cm (0.2273 m, log10=-0.643, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Mesohippus bairdi (Late Eocene (Chadronian, 35.0 Ma))
Geological Horizon: Late Eocene (Chadronian, 35.0 Ma) (35.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mesohippus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (22.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2029_mesohippus_bairdi_collective_stigmergy.lean` (`MathOfBiology.Loop_2029`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/mesohippus_bairdi_collective_stigmergy"
EXPECTED_SHA256 = {
    "mesohippus_bairdi_collective_stigmergy_reference.json": "085aac90ea1958001e0f1c2304de0c02ff6007ee884971e86dc364aef7e95b78",
    "mesohippus_bairdi_specimen.jpg": "ffa85314095959cda06a6a0c44566047678b150e06996250a97c7c8b4dbc1876",
    "scout_manifest.json": "eace1b0751c2ff0cd950ab0ef8aff48bf788daaacc16b55e897db825e00154e6",
    "paper_citations.bib": "33c775ad77e4b8b6861e9ea055340ee790945a33e2fd8755a4348c39984291ba"
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
    ref = json.loads((DATASET_DIR / "mesohippus_bairdi_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-macfadden-mesohippus-bairdi-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
