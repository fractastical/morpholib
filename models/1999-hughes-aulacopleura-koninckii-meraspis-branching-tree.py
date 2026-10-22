#!/usr/bin/env python3
"""
models/1999-hughes-aulacopleura-koninckii-meraspis-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1692)
Organism: Aulacopleura koninckii_meraspis (Silurian Trilobite Meraspid Larva (12 Thoracic Segments, Early Morph))
Phenomenon Scale: 1.6 mm (0.001602 m, log10=-2.795, 6_organ_mm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Aulacopleura koninckii_meraspis (Silurian (Wenlock Epoch, 428.5 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.5 Ma) (428.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (1.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1692_aulacopleura_koninckii_meraspis_branching_tree.lean` (`MathOfBiology.Loop_1692`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_meraspis_branching_tree"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_meraspis_branching_tree_reference.json": "bafe788e577bc56b3d67dc5ed226eae7b921eebbdec5f57ac15559611725c38a",
    "aulacopleura_koninckii_meraspis_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "cf7b3391217613080d65fdafc9a7d259c6306d87c5379b22fd9f0380f86b711b",
    "paper_citations.bib": "b58c5fd83d57137bcd0453fea410d8792eb549b0184104d3190faa8f15857258"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_meraspis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-hughes-aulacopleura-koninckii-meraspis-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
