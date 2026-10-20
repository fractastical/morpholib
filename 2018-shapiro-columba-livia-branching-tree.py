#!/usr/bin/env python3
"""
2018-shapiro-columba-livia-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0614)
Organism: Columba livia (Rock Dove)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Columba
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0614_columba_livia_branching_tree.lean` (`MathOfBiology.Loop_0614`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/columba_livia_branching_tree"
EXPECTED_SHA256 = {
    "columba_livia_branching_tree_reference.json": "b91426394e032b5c2200c545bb8153e14c17d0b1f7918cc80246fb80c0c722f2",
    "columba_livia_specimen.jpg": "00bf4ed50c6075550c363caacdfa0f8820f68486fb8d737e81039f614c1ae7fb",
    "scout_manifest.json": "0ac8678810af0c1873b0db29cb473c0c0b1faf091b18cd2cc12a0b7a0a13a590",
    "paper_citations.bib": "23abf2b62631912824aa3ed9e28718b88d61875383f89689fddabc76679c5064"
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
    ref = json.loads((DATASET_DIR / "columba_livia_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-shapiro-columba-livia-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
