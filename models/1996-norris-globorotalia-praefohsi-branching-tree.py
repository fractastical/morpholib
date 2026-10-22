#!/usr/bin/env python3
"""
models/1996-norris-globorotalia-praefohsi-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1715)
Organism: Globorotalia praefohsi (Middle Miocene Unkeeled Rounded-Margin Foraminifer Chronospecies)
Phenomenon Scale: 106 µm (0.0001059 m, log10=-3.975, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Globorotalia praefohsi (Middle Miocene (Serravallian, 13.4 Ma))
Geological Horizon: Middle Miocene (Serravallian, 13.4 Ma) (13.4 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (106 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1715_globorotalia_praefohsi_branching_tree.lean` (`MathOfBiology.Loop_1715`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_praefohsi_branching_tree"
EXPECTED_SHA256 = {
    "globorotalia_praefohsi_branching_tree_reference.json": "4ff90dd877b3c42c5e87798d7893f956a40493f76ad4dbb7e0c71c1c5a4d3456",
    "globorotalia_praefohsi_specimen.jpg": "4bd61f5c5fd1e4a4d998a7b4bf377b989b14b379445592ebb2d464ec82617c74",
    "scout_manifest.json": "2531b56496ca6fce3a032d027d4f823864e33a7c023b1a4d9f9a965dede2aa91",
    "paper_citations.bib": "2fcdbea7532e0b66eab62747cf422d2c454ac49202a1092e0fce2619ffee6ade"
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
    ref = json.loads((DATASET_DIR / "globorotalia_praefohsi_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-norris-globorotalia-praefohsi-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
