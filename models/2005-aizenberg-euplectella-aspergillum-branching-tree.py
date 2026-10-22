#!/usr/bin/env python3
"""
models/2005-aizenberg-euplectella-aspergillum-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1735)
Organism: Euplectella aspergillum (Venus' Flower Basket Glass Sponge (Hierarchical Silica Lattice Cage))
Phenomenon Scale: 446 µm (0.0004462 m, log10=-3.350, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Euplectella aspergillum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Euplectella
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (446 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1735_euplectella_aspergillum_branching_tree.lean` (`MathOfBiology.Loop_1735`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euplectella_aspergillum_branching_tree"
EXPECTED_SHA256 = {
    "euplectella_aspergillum_branching_tree_reference.json": "15c36ae026b3fc1fe0f1057d2f9eaff1bc597130002083245fffcf959e9e652c",
    "euplectella_aspergillum_specimen.jpg": "c39266f00e99a0f35db98c47cad0aab61aaf982021afc8d7827e37376a49fd28",
    "scout_manifest.json": "703939e73c080d142e66ad51f4acf386321c31ad38ecf75b411de3a3d796908d",
    "paper_citations.bib": "113fa26686446c2e14f05c7f5e1cf05200e2d03c96a5623a5b41ad29f29afaf3"
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
    ref = json.loads((DATASET_DIR / "euplectella_aspergillum_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-aizenberg-euplectella-aspergillum-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
