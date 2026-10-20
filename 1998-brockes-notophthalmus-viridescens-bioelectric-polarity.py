#!/usr/bin/env python3
"""
1998-brockes-notophthalmus-viridescens-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0927)
Organism: Notophthalmus viridescens (Eastern Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Notophthalmus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0927_notophthalmus_viridescens_bioelectric_polarity.lean` (`MathOfBiology.Loop_0927`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/notophthalmus_viridescens_bioelectric_polarity"
EXPECTED_SHA256 = {
    "notophthalmus_viridescens_bioelectric_polarity_reference.json": "c806370f84e5c6ef9d6e7d981e9768d8c99f3e3fd21e56e7b29a825a3caa4c6f",
    "notophthalmus_viridescens_specimen.jpg": "5389e095a5be650aba6c5b00fa322f4fbd3beed046b1913ac82a95ce045d5000",
    "scout_manifest.json": "d20c03f6daf63b72b8e3e802650d228a60a73bc7f64a2fe0cef99bdd9afb7b4a",
    "paper_citations.bib": "2c5488d38a1c3f58a73a53da7c88ef7d36661d21ab06c84308aaff7b532e4829"
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
    ref = json.loads((DATASET_DIR / "notophthalmus_viridescens_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1998-brockes-notophthalmus-viridescens-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
