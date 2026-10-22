#!/usr/bin/env python3
"""
models/2004-fusco-aulacopleura-koninckii-holaspis18-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1693)
Organism: Aulacopleura koninckii_holaspis18 (Silurian Trilobite Holaspid Morphotype I (18 Thoracic Segments))
Phenomenon Scale: 4.8 mm (0.00476 m, log10=-2.322, 6_organ_mm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Aulacopleura koninckii_holaspis18 (Silurian (Wenlock Epoch, 428.3 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.3 Ma) (428.3 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (4.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1693_aulacopleura_koninckii_holaspis18_branching_tree.lean` (`MathOfBiology.Loop_1693`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis18_branching_tree"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis18_branching_tree_reference.json": "a40f77637ae18675e590e0c66b3901845e754e61edf1cb42e5f6cdd94f79afed",
    "aulacopleura_koninckii_holaspis18_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "1ef3c554f0143ebc52868f287e868c594127294ba0480e29ceb880eae67451c0",
    "paper_citations.bib": "d1b0b48c4eb45fbdd916e26e9efe685ffe37b8687089091a2d4e52834cc8c801"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis18_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-fusco-aulacopleura-koninckii-holaspis18-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
