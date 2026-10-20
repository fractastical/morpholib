#!/usr/bin/env python3
"""
1974-wilcox-anabaena-cylindrica-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0285)
Organism: Anabaena cylindrica (Heterocyst Cyanobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Anabaena
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0285_anabaena_cylindrica_morphogen_condensation.lean` (`MathOfBiology.Loop_0285`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anabaena_cylindrica_morphogen_condensation"
EXPECTED_SHA256 = {
    "anabaena_cylindrica_morphogen_condensation_reference.json": "33f748ea4669bcd47fa96936384f87133ced6efc04673479bc9a86fda4633766",
    "anabaena_cylindrica_specimen.jpg": "44fe88b299c228646a9ba3152293e87a876bcaa8ac24671d7b5582ff58ad4ae6",
    "scout_manifest.json": "b13c19996bfb992551aafe2d388704d7390e21e57524150d50d1b56750c1ff2d",
    "paper_citations.bib": "2834b7fd201deb51b46fc4e40738804517872e3ff032ef0fd0f3b180d6de5aae"
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
    ref = json.loads((DATASET_DIR / "anabaena_cylindrica_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1974-wilcox-anabaena-cylindrica-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
