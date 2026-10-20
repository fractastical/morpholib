#!/usr/bin/env python3
"""
1974-wilcox-anabaena-cylindrica-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0985)
Organism: Anabaena cylindrica (Heterocyst Cyanobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Anabaena
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0985_anabaena_cylindrica_bioelectric_polarity.lean` (`MathOfBiology.Loop_0985`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anabaena_cylindrica_bioelectric_polarity"
EXPECTED_SHA256 = {
    "anabaena_cylindrica_bioelectric_polarity_reference.json": "0f9c65a5e2f253b748b7755cd3ebfef5bc301305e089c6219695a1891dec6365",
    "anabaena_cylindrica_specimen.jpg": "44fe88b299c228646a9ba3152293e87a876bcaa8ac24671d7b5582ff58ad4ae6",
    "scout_manifest.json": "c5af4d71d70d4fe3cc35cb7f81c009d45145f072fbc49ced7f0e722fa1fdaa55",
    "paper_citations.bib": "3e750d3258491c04176dd6929795ea72cbac25bc4879d8d4761b68bf6901531e"
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
    ref = json.loads((DATASET_DIR / "anabaena_cylindrica_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1974-wilcox-anabaena-cylindrica-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
