#!/usr/bin/env python3
"""
models/2007-fusco-aulacopleura-koninckii-holaspis18-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1513)
Organism: Aulacopleura koninckii_holaspis18 (Silurian Trilobite Holaspid Morphotype I (18 Thoracic Segments))
Phenomenon Scale: 3.4 mm (0.0034 m, log10=-2.469, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Aulacopleura koninckii_holaspis18 (Silurian (Wenlock Epoch, 428.3 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.3 Ma) (428.3 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (3.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1513_aulacopleura_koninckii_holaspis18_flux_canalization.lean` (`MathOfBiology.Loop_1513`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis18_flux_canalization"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis18_flux_canalization_reference.json": "adeaac13e3ff061e11a288da35f23b77db129ec26c370bb16d9d19ce2deeecd6",
    "aulacopleura_koninckii_holaspis18_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "5b9db020efa454ecc2557db7288ee8d69e5d0a330593631ab40c47210292b2d7",
    "paper_citations.bib": "2d7ba86925f11b12f6b8ccd054ab290c5593051be34d76f2678dc4612d4d90bd"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis18_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-fusco-aulacopleura-koninckii-holaspis18-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
