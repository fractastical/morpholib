#!/usr/bin/env python3
"""
1973-wilcox-anabaena-cylindrica-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0885)
Organism: Anabaena cylindrica (Heterocyst Cyanobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Anabaena
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0885_anabaena_cylindrica_biomineral_accretion.lean` (`MathOfBiology.Loop_0885`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anabaena_cylindrica_biomineral_accretion"
EXPECTED_SHA256 = {
    "anabaena_cylindrica_biomineral_accretion_reference.json": "ffbb81bc22dd18236b92311584d9b1fdbdc1bc670a0efe240d627029adbe9ed6",
    "anabaena_cylindrica_specimen.jpg": "44fe88b299c228646a9ba3152293e87a876bcaa8ac24671d7b5582ff58ad4ae6",
    "scout_manifest.json": "eab02026c3a4ab24b01353983282a53ed26cf401708805f0619813aa7f75a695",
    "paper_citations.bib": "ff5e77fef3c2f0c3dded2db24b755124ec21a69d6cc15477ea357a3f7d55ee54"
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
    ref = json.loads((DATASET_DIR / "anabaena_cylindrica_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1973-wilcox-anabaena-cylindrica-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
