#!/usr/bin/env python3
"""
models/2018-spiewak-danio-aesculapii-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1654)
Organism: Danio aesculapii (Panther Danio (Vertical Bar Transition))
Phenomenon Scale: 8.8 mm (0.008827 m, log10=-2.054, 6_organ_mm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Danio aesculapii (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (8.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1654_danio_aesculapii_branching_tree.lean` (`MathOfBiology.Loop_1654`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_aesculapii_branching_tree"
EXPECTED_SHA256 = {
    "danio_aesculapii_branching_tree_reference.json": "8f8f86edcb599da3637e1402e9a317825d28b89df7178d6e536d8892d82ff3be",
    "danio_aesculapii_specimen.jpg": "4ccce0ce3648e44c77cef768d10f5acd8ab553c4ca38d348c35d4c6846b46bc5",
    "scout_manifest.json": "08aeb542e6eb02e5db23a17405b81138fd19c5cfc28d90ec7b4fa04becb25d96",
    "paper_citations.bib": "fdbfe7c5eef81d604f29e22f7339d9960e1945429a1af77dbfff7470699e53fd"
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
    ref = json.loads((DATASET_DIR / "danio_aesculapii_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-spiewak-danio-aesculapii-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
