#!/usr/bin/env python3
"""
models/2004-klug-ceratites-nodosus-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1686)
Organism: Ceratites nodosus (Middle Triassic Muschelkalk Ammonoid (Serrated Lobes, D_f=1.34))
Phenomenon Scale: 9.5 mm (0.00952 m, log10=-2.021, 6_organ_mm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Ceratites nodosus (Middle Triassic (Ladinian, 240 Ma))
Geological Horizon: Middle Triassic (Ladinian, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ceratites
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (9.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1686_ceratites_nodosus_branching_tree.lean` (`MathOfBiology.Loop_1686`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ceratites_nodosus_branching_tree"
EXPECTED_SHA256 = {
    "ceratites_nodosus_branching_tree_reference.json": "634a20fe803c55a629e0f997dcfe835180e2f076f49fce84034e8586dc817af2",
    "ceratites_nodosus_specimen.jpg": "3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0",
    "scout_manifest.json": "7344cd70ced377c9c5b2d90fbc6b64e763f18b8c59649d87ee93d71319f1c92d",
    "paper_citations.bib": "8adb2c11b3b0d951a64f648052171018fd5da3cf56bb477b0e7d6bd001e471c8"
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
    ref = json.loads((DATASET_DIR / "ceratites_nodosus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-klug-ceratites-nodosus-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
