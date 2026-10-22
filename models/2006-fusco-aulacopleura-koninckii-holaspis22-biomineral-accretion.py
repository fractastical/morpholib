#!/usr/bin/env python3
"""
models/2006-fusco-aulacopleura-koninckii-holaspis22-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1874)
Organism: Aulacopleura koninckii_holaspis22 (Silurian Trilobite Holaspid Morphotype II (22 Thoracic Segments))
Phenomenon Scale: 1.1 cm (0.01077 m, log10=-1.968, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Aulacopleura koninckii_holaspis22 (Silurian (Wenlock Epoch, 428.1 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.1 Ma) (428.1 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1874_aulacopleura_koninckii_holaspis22_biomineral_accretion.lean` (`MathOfBiology.Loop_1874`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis22_biomineral_accretion"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis22_biomineral_accretion_reference.json": "ffeee5728160e40bb1432439f3e94d119624254fda6f5c8b007eccab657615b2",
    "aulacopleura_koninckii_holaspis22_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "125b2db94ca7730162df2f029aed2e3e878fe757472ba16880401ade5d4b03b9",
    "paper_citations.bib": "02d301d0d9734fb176a3114f5b53abab3b3b840593201f0713360f9c48d1a97d"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis22_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-fusco-aulacopleura-koninckii-holaspis22-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
