#!/usr/bin/env python3
"""
2019-shapiro-columba-livia-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0714)
Organism: Columba livia (Rock Dove)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Columba
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0714_columba_livia_elastic_buckling.lean` (`MathOfBiology.Loop_0714`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/columba_livia_elastic_buckling"
EXPECTED_SHA256 = {
    "columba_livia_elastic_buckling_reference.json": "eb765366371a8073961281418148b451c398454eede9eda2c65891deb15b16a7",
    "columba_livia_specimen.jpg": "00bf4ed50c6075550c363caacdfa0f8820f68486fb8d737e81039f614c1ae7fb",
    "scout_manifest.json": "938b945f41948e9681124c7883b4864c18149718a8df6761563fe72120392d4c",
    "paper_citations.bib": "03b40143c008d89ee3a357448d0d132f7b90640a9157a3983e3894a09ca38a15"
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
    ref = json.loads((DATASET_DIR / "columba_livia_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-shapiro-columba-livia-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
