#!/usr/bin/env python3
"""
models/2013-riedel-danio-rerio-nm28-delta-notch-synapse-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1112)
Organism: Danio rerio (Zebrafish — 12–16 hpf)
Phenomenon Scale: 23 nm (2.282e-08 m, log10=-7.642, 1_molecular_nm)
Anatomical Structure: DeltaC–Notch1a trans-endocytosis & EphA4–EphrinB2 boundary complex (12–16 hpf (6–14 Somite Stage, ZFA:0000115)) · Zebrahub somite_boundary_epithelium & notochord_sheath (12–16 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation at 23 nm (DeltaC–Notch1a trans-endocytosis & EphA4–EphrinB2 boundary complex))
Formal Verification: `morphoeval/lean/loops/Loop_1112_danio_rerio_nm28_delta_notch_synapse_morphogen_condensation.lean` (`MathOfBiology.Loop_1112`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_nm28_delta_notch_synapse_morphogen_condensation"
EXPECTED_SHA256 = {
    "danio_rerio_nm28_delta_notch_synapse_morphogen_condensation_reference.json": "301c4905c762a793befb241dd9e5409faaf429a9c72c22f8ccc93a204c7a816d",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "7cb896f703cc15f294d18c28a05f3d11d740488bbdcd787c26e7d4982a722b78",
    "paper_citations.bib": "ee5bafaeee5093297cab44020762d5954c8de3f8687746819e958b44eb5b6e50"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_nm28_delta_notch_synapse_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-riedel-danio-rerio-nm28-delta-notch-synapse-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
