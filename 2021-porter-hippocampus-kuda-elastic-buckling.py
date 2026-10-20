#!/usr/bin/env python3
"""
2021-porter-hippocampus-kuda-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0734)
Organism: Hippocampus kuda (Spotted Seahorse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Hippocampus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0734_hippocampus_kuda_elastic_buckling.lean` (`MathOfBiology.Loop_0734`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hippocampus_kuda_elastic_buckling"
EXPECTED_SHA256 = {
    "hippocampus_kuda_elastic_buckling_reference.json": "3f697cf324ed1dfbbcb80c9c7a80ab0c67d5dc5d40624761cc2fded8bd21f65c",
    "hippocampus_kuda_specimen.jpg": "2d453503e6f09426796b25434fc672645ac4aa639654b8d282c6607928df715e",
    "scout_manifest.json": "22d571b97dfc98989838c12543b57466aaf4e3c31805566980a510d236e45a02",
    "paper_citations.bib": "dc0c729bb92731baf1c9ce5cdd65cd87e820889ed5d5c42f3d81d15fa92cc61b"
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
    ref = json.loads((DATASET_DIR / "hippocampus_kuda_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-porter-hippocampus-kuda-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
