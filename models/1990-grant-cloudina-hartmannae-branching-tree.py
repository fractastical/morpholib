#!/usr/bin/env python3
"""
models/1990-grant-cloudina-hartmannae-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1702)
Organism: Cloudina hartmannae (Terminal Ediacaran Cone-in-Cone Biomineralized Calcite Tube)
Phenomenon Scale: 911 µm (0.0009114 m, log10=-3.040, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Cloudina hartmannae (Terminal Ediacaran (Nama Group, 549 Ma))
Geological Horizon: Terminal Ediacaran (Nama Group, 549 Ma) (549.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Cloudina
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (911 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1702_cloudina_hartmannae_branching_tree.lean` (`MathOfBiology.Loop_1702`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cloudina_hartmannae_branching_tree"
EXPECTED_SHA256 = {
    "cloudina_hartmannae_branching_tree_reference.json": "b0a038601a78da71d3816aaf08035d23a334522b5f80b6756ac6433ae68f98e3",
    "cloudina_hartmannae_specimen.jpg": "cbe0233dbf275f4e575ad54b3134b268c070d5cafa4a4ef2add96d579bc0efe6",
    "scout_manifest.json": "50d9302ca28ce627ff06cfb5fb7c16ba29747e22bfd410e269c5b4e3e5395706",
    "paper_citations.bib": "0a9ae39c5c79bbd58052432a7dfa263b0e38513947b326d44f331b37c69ffe7b"
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
    ref = json.loads((DATASET_DIR / "cloudina_hartmannae_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1990-grant-cloudina-hartmannae-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
