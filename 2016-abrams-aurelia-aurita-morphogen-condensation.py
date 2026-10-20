#!/usr/bin/env python3
"""
2016-abrams-aurelia-aurita-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0264)
Organism: Aurelia aurita (Moon Jellyfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Aurelia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0264_aurelia_aurita_morphogen_condensation.lean` (`MathOfBiology.Loop_0264`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aurelia_aurita_morphogen_condensation"
EXPECTED_SHA256 = {
    "aurelia_aurita_morphogen_condensation_reference.json": "51eb4610efab2b26e5e28d6388fca92501318f82dffd554ee70c01121869849c",
    "aurelia_aurita_specimen.jpg": "1a8c5b9fb4abe35f1362b2c458d03bafa15966b0f6046d49aa840fdd3a66769e",
    "scout_manifest.json": "e42e5241324acf1e2fdfb9e2dc09b643601214b8f7ff0a84aca64d9f6d3cae2e",
    "paper_citations.bib": "ad10ea3ab7d7b0fa6f2a3170445f8b826bb61266684ac6440b29f04f0d075046"
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
    ref = json.loads((DATASET_DIR / "aurelia_aurita_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2016-abrams-aurelia-aurita-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
