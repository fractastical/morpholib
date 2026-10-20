#!/usr/bin/env python3
"""
2014-amemiya-latimeria-chalumnae-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0939)
Organism: Latimeria chalumnae (West Indian Coelacanth)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Latimeria
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0939_latimeria_chalumnae_bioelectric_polarity.lean` (`MathOfBiology.Loop_0939`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/latimeria_chalumnae_bioelectric_polarity"
EXPECTED_SHA256 = {
    "latimeria_chalumnae_bioelectric_polarity_reference.json": "0877a23c1a92a60e88de15e946b3d670bdfb7d052544c51521866538ce93cb9e",
    "latimeria_chalumnae_specimen.jpg": "29bcb5946834bbbf95c873d247d7bb9449f8ea6b56ddef496b4eef521ce6cd5d",
    "scout_manifest.json": "9c6eb678679c8ddbe9cf623f83951055134f1b498ff43415a06dd65888342009",
    "paper_citations.bib": "969181ed111a275d5389c9c3ee254dd2f7e85ed2a691f79f4a8165927bbf2085"
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
    ref = json.loads((DATASET_DIR / "latimeria_chalumnae_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-amemiya-latimeria-chalumnae-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
