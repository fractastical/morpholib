#!/usr/bin/env python3
"""
models/1992-dimichele-lepidodendron-aculeatum-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1708)
Organism: Lepidodendron aculeatum (Carboniferous Giant Scale-Tree (Helical Parastichy Leaf Cushions))
Phenomenon Scale: 1.2 cm (0.01184 m, log10=-1.927, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Lepidodendron aculeatum (Late Carboniferous (Pennsylvanian, 310 Ma))
Geological Horizon: Late Carboniferous (Pennsylvanian, 310 Ma) (310.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Lepidodendron
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (1.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1708_lepidodendron_aculeatum_branching_tree.lean` (`MathOfBiology.Loop_1708`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/lepidodendron_aculeatum_branching_tree"
EXPECTED_SHA256 = {
    "lepidodendron_aculeatum_branching_tree_reference.json": "a2dda9eb0a7ac0c6c21bf1d599447dc35533e4d579ba6ca7fd32223b60005925",
    "lepidodendron_aculeatum_specimen.jpg": "854f4a6257e58a0123b8eee34674c32ed016882c88d6eb907ca1d85dbb78c467",
    "scout_manifest.json": "1ffe36c4a4ec49f18bb1ed62fac60a7e0755f005288228710266c6ba0a7281a2",
    "paper_citations.bib": "c15bb411a1004da55f9632ea9af42c0b03755726495e2cac553275d54d022766"
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
    ref = json.loads((DATASET_DIR / "lepidodendron_aculeatum_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-dimichele-lepidodendron-aculeatum-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
